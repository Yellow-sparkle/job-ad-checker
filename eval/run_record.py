"""Run record for Job Ad Checker eval scripts (fixes 1 and 3, built 14 Nov 2026).

Every eval run gets its own folder, eval/runs/<run_id>/, holding:
  run_info.json  what was run: golden-set version and hash, tool-output files and
                 hashes, judge model, temperature, prompt versions and hashes,
                 thresholds, checker versions, warnings, status, usage summary
  usage.csv      one row per judge call or checker result: case_id, source,
                 metric, input_tokens, output_tokens, cost_usd, seconds, error
  <the script's own result files, unchanged in content>

Numbers and ids only: no ad text, no reasons, no prompts, no API key is written
by this module. Nothing is sent anywhere (no Opik cloud, track=False stays).

Used by faithfulness_check.py, extra_checks.py and stability_check.py:

    from run_record import start_run, parse_args
    pos, opts = parse_args(sys.argv[1:])
    with start_run("faithfulness_check", gold_path, [out_path], judges={...},
                   model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE,
                   thresholds={...}, run_dir=opts["run_dir"],
                   dry_run=opts["dry_run"]) as run:
        res = run.timed(case_id, "faithfulness", metric.score, input=..., ...)
        json.dump(rows, open(run.path("faithfulness_results.json"), "w"))

Compare two runs with compare_runs.py.
"""
import csv, datetime, hashlib, json, os, platform, re, statistics, sys, time

# Opik reports errors to Sentry and usage analytics to Comet by default. Neither
# carries ad text, but Kate's rule is that eval checks stay local, so both are
# switched off here. Import run_record before opik for this to take effect.
os.environ.setdefault("OPIK_SENTRY_ENABLE", "false")
os.environ.setdefault("OPIK_ANALYTICS_ENABLE", "false")

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
USAGE_FIELDS = ["case_id", "source", "metric", "input_tokens", "output_tokens",
                "cost_usd", "seconds", "error"]
PROMPT_NAME = re.compile(r"(PROMPT|EXAMPLES|TEMPLATE)$")


# ---- small helpers ---------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def golden_version(path):
    """'golden-set-v0.13.jsonl' -> 'v0.13'; None if the name has no version."""
    m = re.search(r"golden-set-(v\d+(?:\.\d+)*)\.jsonl$", os.path.basename(path))
    return m.group(1) if m else None


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def prompt_hash(module):
    """Hash every module-level string whose name ends in PROMPT, EXAMPLES or TEMPLATE."""
    parts = [f"{name}\n{value}" for name, value in sorted(vars(module).items())
             if PROMPT_NAME.search(name) and isinstance(value, str)]
    return sha256_text("\n\n".join(parts))


def parse_args(argv):
    """Split argv into positional args and the shared options.
    Options: --run-dir <folder>, --dry-run. Other --flags are kept in opts['flags']."""
    pos, flags, run_dir, dry = [], set(), None, False
    it = iter(argv)
    for a in it:
        if a == "--run-dir":
            run_dir = next(it, None)
            if run_dir is None:
                sys.exit("--run-dir needs a folder name")
        elif a.startswith("--run-dir="):
            run_dir = a.split("=", 1)[1]
        elif a == "--dry-run":
            dry = True
        elif a.startswith("--"):
            flags.add(a)
        else:
            pos.append(a)
    return pos, {"run_dir": run_dir, "dry_run": dry, "flags": flags}


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _p90(values):
    if not values:
        return None
    values = sorted(values)
    k = max(0, int(round(0.9 * len(values) + 0.5)) - 1)
    return round(values[min(k, len(values) - 1)], 3)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


# ---- the run ---------------------------------------------------------------
class Run:
    def __init__(self, folder, info, dry_run):
        self.folder = folder
        self.info = info
        self.dry_run = dry_run
        self.usage = []          # rows for usage.csv
        self._checker_versions = {}
        self._checker_prompt_versions = {}
        self._token_warning_given = False
        self._finished = False

    # where the script writes its result files
    def path(self, name):
        return os.path.join(self.folder, name)

    def warn(self, text):
        self.info["warnings"].append(text)
        print(f"WARNING: {text}", file=sys.stderr)

    # -- judge calls --
    def timed(self, case_id, metric_name, fn, *args, **kwargs):
        """Call a judge's score(...) and record seconds, tokens and cost (numbers only).

        Tokens are read from the model responses by briefly wrapping the call the
        Opik model makes: LiteLLM (model._litellm_completion) or the Anthropic SDK
        (model._client.messages.create / .parse). Only the usage numbers are kept."""
        if self.dry_run:
            raise RuntimeError("judge called during --dry-run")
        calls = []
        model = getattr(getattr(fn, "__self__", None), "_model", None)
        patched = []
        for owner, attr in ((model, "_litellm_completion"),
                            (getattr(getattr(model, "_client", None), "messages", None), "create"),
                            (getattr(getattr(model, "_client", None), "messages", None), "parse")):
            original = getattr(owner, attr, None) if owner is not None else None
            if not callable(original):
                continue
            had_own = attr in getattr(owner, "__dict__", {})

            def capture(*a, _orig=original, **kw):
                resp = _orig(*a, **kw)
                calls.append(resp)
                return resp
            try:
                setattr(owner, attr, capture)
                patched.append((owner, attr, original, had_own))
            except Exception:
                continue
        row = {"case_id": case_id, "source": "judge", "metric": metric_name,
               "input_tokens": None, "output_tokens": None, "cost_usd": None,
               "seconds": None, "error": ""}
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            row["error"] = type(e).__name__
            raise
        finally:
            row["seconds"] = round(time.perf_counter() - t0, 3)
            for owner, attr, original, had_own in patched:
                if had_own:
                    setattr(owner, attr, original)
                else:
                    try:
                        delattr(owner, attr)
                    except AttributeError:
                        setattr(owner, attr, original)
            if calls:
                row.update(self._tokens_and_cost(calls, getattr(model, "model_name", None)))
            elif not self._token_warning_given and not row["error"]:
                self._token_warning_given = True
                self.warn("tokens and cost could not be captured for judge calls; "
                          "recording seconds only")
            self.usage.append(row)

    @staticmethod
    def _tokens_and_cost(responses, model_name=None):
        tin = tout = 0
        cost, cost_known = 0.0, True
        for r in responses:
            u = getattr(r, "usage", None)
            if u is None and isinstance(r, dict):
                u = r.get("usage")
            get = u.get if isinstance(u, dict) else (lambda k: getattr(u, k, None))
            i = _num(get("prompt_tokens"))
            o = _num(get("completion_tokens"))
            if i is None:
                i = _num(get("input_tokens"))
            if o is None:
                o = _num(get("output_tokens"))
            tin += i or 0
            tout += o or 0
            c = None
            try:
                import litellm
                try:
                    c = litellm.completion_cost(completion_response=r)
                except Exception:
                    if model_name:
                        p, q = litellm.cost_per_token(model=model_name, prompt_tokens=i or 0,
                                                      completion_tokens=o or 0)
                        c = p + q
            except Exception:
                c = None
            if _num(c) is None:
                cost_known = False
            else:
                cost += c
        return {"input_tokens": tin, "output_tokens": tout,
                "cost_usd": round(cost, 6) if cost_known else None}

    # -- the checker's own usage, from tool_outputs.jsonl --
    def add_checker_usage(self, case_id, out):
        """out: one tool-output line (dict). Reads the optional usage and version fields."""
        for key, store in (("checker_version", self._checker_versions),
                           ("checker_prompt_version", self._checker_prompt_versions)):
            v = out.get(key)
            if v is not None:
                store[str(v)] = store.get(str(v), 0) + 1
        u = out.get("usage")
        if not isinstance(u, dict):
            return
        self.usage.append({"case_id": case_id, "source": "checker", "metric": "check",
                           "input_tokens": _num(u.get("input_tokens")),
                           "output_tokens": _num(u.get("output_tokens")),
                           "cost_usd": _num(u.get("cost_usd")),
                           "seconds": _num(u.get("latency_s")), "error": ""})

    # -- end of run --
    def summary(self):
        out = {}
        for source in ("judge", "checker"):
            rows = [r for r in self.usage if r["source"] == source]
            secs = [r["seconds"] for r in rows if _num(r["seconds"]) is not None]
            costs = [r["cost_usd"] for r in rows if _num(r["cost_usd"]) is not None]
            out[source] = {
                "calls": len(rows),
                "errors": sum(1 for r in rows if r["error"]),
                "seconds_total": round(sum(secs), 3) if secs else None,
                "seconds_mean": round(statistics.mean(secs), 3) if secs else None,
                "seconds_p90": _p90(secs),
                "input_tokens": sum(r["input_tokens"] or 0 for r in rows) if any(
                    r["input_tokens"] is not None for r in rows) else None,
                "output_tokens": sum(r["output_tokens"] or 0 for r in rows) if any(
                    r["output_tokens"] is not None for r in rows) else None,
                "cost_usd": round(sum(costs), 6) if costs else None,
                "calls_without_cost": len(rows) - len(costs),
            }
        return out

    def finish(self, status="completed", error=None):
        if self._finished:
            return
        self._finished = True
        with open(self.path("usage.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=USAGE_FIELDS)
            w.writeheader()
            for r in self.usage:
                w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in USAGE_FIELDS})
        self.info.update(status=status, error=error, finished_at=_now().isoformat(timespec="seconds"),
                         usage_summary=self.summary())
        self.info["checker"] = {"versions": self._checker_versions,
                                "prompt_versions": self._checker_prompt_versions}
        self._write_info()
        print(self.summary_line())

    def summary_line(self):
        s = self.info.get("usage_summary") or self.summary()
        j, c = s["judge"], s["checker"]

        def part(name, d):
            if not d["calls"]:
                return f"{name}: none"
            bits = [f"{d['calls']} calls"]
            if d["seconds_total"] is not None:
                bits.append(f"{d['seconds_total']:.1f}s (p90 {d['seconds_p90']:.1f}s)")
            if d["input_tokens"] is not None:
                bits.append(f"{d['input_tokens']} in / {d['output_tokens']} out tokens")
            if d["cost_usd"] is not None:
                bits.append(f"${d['cost_usd']:.4f}" + (f" ({d['calls_without_cost']} without price)"
                                                      if d["calls_without_cost"] else ""))
            return f"{name}: " + ", ".join(bits)

        return (f"Run {self.info['run_id']} {self.info['status']}{' (dry run)' if self.dry_run else ''}"
                f" | {part('judge', j)} | {part('checker', c)} | folder: {self.folder}")

    def _write_info(self):
        with open(self.path("run_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.info, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.finish("completed")
        elif exc_type is SystemExit and not exc.code:
            self.finish("completed")
        else:
            self.finish("failed", f"{exc_type.__name__}: {str(exc)[:200]}")
        return False


def _previous_infos(runs_dir, current):
    infos = []
    if not os.path.isdir(runs_dir):
        return infos
    for name in os.listdir(runs_dir):
        p = os.path.join(runs_dir, name, "run_info.json")
        if os.path.join(runs_dir, name) == current or not os.path.isfile(p):
            continue
        try:
            infos.append((os.path.getmtime(p), json.load(open(p, encoding="utf-8"))))
        except Exception:
            continue
    return [i for _, i in sorted(infos, key=lambda x: x[0], reverse=True)]


def describe_judges(judges, model, temperature):
    """judges: {name: module} or {name: {"prompt_version": ..., "text": ...}}."""
    out = {}
    for name, j in (judges or {}).items():
        if isinstance(j, dict):
            version, h = j.get("prompt_version"), sha256_text(j.get("text", ""))
        else:
            version, h = getattr(j, "PROMPT_VERSION", None), prompt_hash(j)
        out[name] = {"model": model, "temperature": temperature,
                     "temperature_applied": temperature_applied(model, temperature),
                     "prompt_version": version, "prompt_hash": h}
    return out


def temperature_applied(model, temperature):
    """False when Opik drops the temperature for this model (Claude 5.x and some
    others accept no sampling settings). None when it can't be told."""
    if temperature is None or not model:
        return None
    try:
        from opik.evaluation.models.litellm import util
        return not util.rejects_sampling_params(model)
    except Exception:
        return None


def start_run(script, golden_path, output_paths, judges=None, model=None, temperature=None,
              thresholds=None, run_dir=None, dry_run=False, runs_dir=None):
    runs_dir = runs_dir or RUNS_DIR
    started = _now()
    gv = golden_version(golden_path) or "unversioned"
    run_id = f"{started.strftime('%Y%m%d-%H%M%S')}-{script}-{gv}" + ("-dry" if dry_run else "")
    folder = os.path.abspath(run_dir) if run_dir else os.path.join(runs_dir, run_id)
    if run_dir:
        run_id = os.path.basename(folder.rstrip(os.sep))
    if os.path.exists(os.path.join(folder, "run_info.json")):
        sys.exit(f"Run folder already used: {folder}. Pick another --run-dir.")
    os.makedirs(folder, exist_ok=True)

    info = {"run_id": run_id, "script": script, "status": "running", "error": None,
            "dry_run": dry_run, "started_at": started.isoformat(timespec="seconds"),
            "finished_at": None, "golden_set": None, "outputs": [],
            "judges": describe_judges(judges, model, temperature),
            "thresholds": thresholds or {}, "checker": {}, "warnings": [],
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "opik": _pkg_version("opik"), "litellm": _pkg_version("litellm")},
            "usage_summary": None}
    run = Run(folder, info, dry_run)

    missing = [p for p in [golden_path, *output_paths] if not os.path.isfile(p)]
    if missing:
        run.finish("failed", "missing file: " + ", ".join(os.path.basename(p) for p in missing))
        sys.exit(f"Missing file(s): {', '.join(missing)}")

    info["golden_set"] = {"file": os.path.basename(golden_path), "version": gv,
                          "sha256": sha256_file(golden_path), "cases": count_lines(golden_path)}
    info["outputs"] = [{"file": os.path.basename(p), "sha256": sha256_file(p),
                        "lines": count_lines(p)} for p in output_paths]

    # warnings: temperature, and a prompt change without a new PROMPT_VERSION
    if info["judges"] and temperature not in (None, 0, 0.0):
        run.warn(f"judge temperature is {temperature}, not 0: scores may not repeat between runs")
    if any(j.get("temperature_applied") is False for j in info["judges"].values()):
        run.warn(f"the judge model {model} accepts no temperature setting, so temperature "
                 f"{temperature} is not applied: judge scores may vary a little between runs")
    previous = _previous_infos(runs_dir, folder)
    for name, j in info["judges"].items():
        prev = next((p["judges"][name] for p in previous if name in (p.get("judges") or {})), None)
        if prev and prev.get("prompt_hash") != j["prompt_hash"] \
                and prev.get("prompt_version") == j["prompt_version"]:
            run.warn(f"the {name} prompt changed but PROMPT_VERSION is still "
                     f"{j['prompt_version']!r}: update PROMPT_VERSION in its module")
    run._write_info()
    return run
