"""Batch mode for the AI judge: half price, same scores (Kate, 14 Nov: cost plan, points 1 and 5).

Used by `faithfulness_check.py ... --batch`. Without --batch nothing here runs and
the judge works exactly as before.

How it keeps the scores the same as a live run:
1. Collect: runs faithfulness_check's own score_all() once with the judge calls
   intercepted. Each call is built by Opik's own request builder (same model,
   prompt, JSON format, max_tokens and settings as a live call), so the only
   differences are the transport (Message Batches API) and prompt caching.
2. Submit all judge calls as one batch (50% off) and wait for it to finish.
   The judge's system prompt is marked for caching. Caching doesn't change what
   the judge sees, only what the repeated part costs.
3. Replay: runs score_all() again, handing each judge call its batch answer.
   Every request is checked against the one sent, so an answer can't end up on
   the wrong reason. All results files are written by the same code as a live run.

The batch id is saved in the run folder (batch_requests.json). If the batch
isn't finished by --batch-wait minutes (default 90), the script stops and
prints the command to pick the same batch up later (--batch-id <id>), or you
run the check live without --batch.

Needs: pip install anthropic opik, and ANTHROPIC_API_KEY.
"""
import contextlib, copy, hashlib, io, json, os, sys, tempfile, time

# Batch prices: 50% of the standard rate per million tokens (from the claude-api
# reference, 3 Nov 2026). Cache writes cost 1.25x input, cache reads 0.1x input.
BATCH_DISCOUNT = 0.5
PRICES = {  # model: (input, output) $ per million tokens, standard rate
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
POLL_SECONDS = 30


def _key(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def _placeholder(response_format):
    """A valid-looking JSON answer for the collect pass (never saved anywhere)."""
    fields = getattr(response_format, "model_fields", {}) if response_format else {}
    out = {}
    for name, f in fields.items():
        t = str(getattr(f, "annotation", ""))
        out[name] = [] if "List" in t or "list" in t else ("" if "str" in t else 0.0)
    return json.dumps(out or {"score": 0.0, "reason": []})


def _with_cache(params):
    """Mark the judge's system prompt for prompt caching (same text, cheaper re-reads)."""
    p = copy.deepcopy(params)
    if isinstance(p.get("system"), str):
        p["system"] = [{"type": "text", "text": p["system"],
                        "cache_control": {"type": "ephemeral"}}]
    return p


def _build_params(model, messages, response_format, kwargs):
    """The exact request a live call would send, built by Opik's own code."""
    if not hasattr(model, "_build_call_kwargs"):
        sys.exit("Batch mode needs Opik's native Anthropic model (pip install anthropic). "
                 "Run without --batch instead.")
    params = model._build_call_kwargs(list(messages), dict(kwargs))
    if response_format is not None:
        # A live call goes through messages.parse(output_format=...), and the SDK turns the
        # pydantic model into a strict schema (adds "additionalProperties": false and more).
        # Use the SDK's own transform so the batch request is the same as the live one.
        # (Opik's pydantic_to_output_config skips that step; the API rejected all 241
        # requests of batch msgbatch_01QMydmC8fdRJFHfEx3H9avg on 14 Nov.)
        from anthropic.lib._parse._transform import transform_schema
        oc = dict(params.get("output_config") or {})
        oc["format"] = {"type": "json_schema", "schema": transform_schema(response_format)}
        params["output_config"] = oc
    return params


class _Patch:
    """Swap generate_chat_completion on each judge model for the collect or replay pass."""

    def __init__(self, metrics, fn):
        self.metrics, self.fn, self.saved = metrics, fn, []

    def __enter__(self):
        for m in self.metrics:
            model = m._model
            self.saved.append((model, model.__dict__.get("generate_chat_completion")))
            model.generate_chat_completion = (
                lambda messages, response_format=None, _model=model, **kw:
                self.fn(_model, messages, response_format, kw))
        return self

    def __exit__(self, *exc):
        for model, own in self.saved:
            if own is None:
                del model.generate_chat_completion
            else:
                model.generate_chat_completion = own
        return False


class _CollectRun:
    """Stand-in for the run during the collect pass: records nothing, writes to a temp folder."""

    def __init__(self):
        self.dry_run = False
        self.calls = []  # (case_id, metric_name) in call order
        self.tmp = tempfile.mkdtemp(prefix="judge-batch-collect-")

    def add_checker_usage(self, *a, **k):
        pass

    def timed(self, case_id, metric_name, fn, *args, **kwargs):
        self.calls.append((case_id, metric_name))
        return fn(*args, **kwargs)

    def path(self, name):
        return os.path.join(self.tmp, name)


class _ReplayRun:
    """Wraps the real run: same files and checker usage, judge usage taken from the batch."""

    def __init__(self, run, answers):
        self._run, self._answers, self._i = run, answers, 0

    def __getattr__(self, name):
        return getattr(self._run, name)

    def timed(self, case_id, metric_name, fn, *args, **kwargs):
        ans = self._answers[self._i]
        self._i += 1
        row = {"case_id": case_id, "source": "judge", "metric": metric_name,
               "input_tokens": ans["usage"].get("input_tokens"),
               "output_tokens": ans["usage"].get("output_tokens"),
               "cost_usd": ans.get("cost_usd"), "seconds": None, "error": ""}
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            row["error"] = type(e).__name__
            raise
        finally:
            self._run.usage.append(row)


def _cost(model_name, usage):
    price = PRICES.get(model_name)
    if not price:
        return None
    pin, pout = price
    inp = usage.get("input_tokens") or 0
    cw = usage.get("cache_creation_input_tokens") or 0
    cr = usage.get("cache_read_input_tokens") or 0
    out = usage.get("output_tokens") or 0
    dollars = (inp * pin + cw * pin * 1.25 + cr * pin * 0.1 + out * pout) / 1e6
    return round(dollars * BATCH_DISCOUNT, 6)


def _error_text(result):
    """The API's error message for a failed batch request, as plain text."""
    if result.type != "errored":
        return f"request {result.type}"
    err = getattr(result, "error", None)
    inner = getattr(err, "error", err)
    kind = getattr(inner, "type", "") or ""
    msg = getattr(inner, "message", "") or str(inner)
    return f"{kind}: {msg}".strip(": ")


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("Batch mode needs the Anthropic SDK: pip install anthropic")
    return anthropic.Anthropic()


def _wait(client, batch_id, wait_minutes, run):
    deadline = time.time() + wait_minutes * 60
    while True:
        b = client.messages.batches.retrieve(batch_id)
        c = b.request_counts
        print(f"Batch {batch_id}: {b.processing_status} | done {c.succeeded}, "
              f"errors {c.errored}, still running {c.processing}")
        if b.processing_status == "ended":
            return
        if time.time() > deadline:
            run.warn(f"batch {batch_id} not finished after {wait_minutes} min")
            raise SystemExit(
                f"Batch not finished after {wait_minutes} min. Pick it up later with:\n"
                f"  python faithfulness_check.py <golden> <outputs> --batch --batch-id {batch_id}\n"
                f"or run the check live (without --batch).")
        time.sleep(POLL_SECONDS)


def score_all_batch(fc, gold_path, out_path, run, batch_id=None, wait_minutes=90, use_cache=True):
    """fc: the faithfulness_check module. Same results as fc.score_all, via one batch."""
    metrics = [fc.metric, fc.relevance, fc.precision]
    requests, keys = [], []

    # 1. Collect every judge call, built exactly as a live call would be.
    def collect(model, messages, response_format, kw):
        params = _build_params(model, messages, response_format, kw)
        keys.append(_key(params))
        sent = _with_cache(params) if use_cache else params
        requests.append({"custom_id": f"j{len(requests):05d}", "params": sent})
        return {"role": "assistant", "content": _placeholder(response_format)}

    collect_run = _CollectRun()
    with _Patch(metrics, collect), contextlib.redirect_stdout(io.StringIO()):
        fc.score_all(gold_path, out_path, collect_run)  # placeholder answers: output hidden
    if not requests:
        print("No judge calls to make.")
        return fc.score_all(gold_path, out_path, run)
    model_name = requests[0]["params"]["model"]

    # 2. Submit (or pick up an earlier batch) and wait.
    client = _client()
    saved = run.path("batch_requests.json")
    if batch_id is None:
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"Submitted batch {batch_id}: {len(requests)} judge calls, 50% price"
              f"{', judge prompt cached' if use_cache else ''}.")
    else:
        print(f"Picking up batch {batch_id} ({len(requests)} judge calls expected).")
    with open(saved, "w", encoding="utf-8") as f:
        json.dump({"batch_id": batch_id, "model": model_name, "prompt_cache": use_cache,
                   "calls": [{"custom_id": r["custom_id"], "case_id": c, "metric": m, "request_sha256": k}
                             for r, (c, m), k in zip(requests, collect_run.calls, keys)]}, f, indent=2)
    run.info["judge_mode"] = {"mode": "batch", "batch_id": batch_id, "prompt_cache": use_cache,
                              "calls": len(requests), "price": "50% batch rate"}
    _wait(client, batch_id, wait_minutes, run)

    # 3. Read the answers, in the order the calls were made.
    by_id, failed, first_error = {}, [], None
    for res in client.messages.batches.results(batch_id):
        if res.result.type != "succeeded":
            failed.append(f"{res.custom_id} ({res.result.type})")
            if first_error is None:
                first_error = _error_text(res.result)
            continue
        msg = res.result.message
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = {k: getattr(msg.usage, k, None) for k in
                 ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        by_id[res.custom_id] = {"text": text, "usage": usage, "cost_usd": _cost(model_name, usage),
                                "stop_reason": getattr(msg, "stop_reason", None)}
    failed_ids = {f.split(" ")[0] for f in failed}
    missing = [r["custom_id"] for r in requests if r["custom_id"] not in by_id and r["custom_id"] not in failed_ids]
    if failed or missing:
        if first_error:
            print(f"First error from the batch: {first_error}")
        raise RuntimeError(f"{len(failed) + len(missing)} of {len(requests)} judge calls did not succeed in batch "
                           f"{batch_id} ({len(failed)} errored or expired, {len(missing)} missing; for example "
                           f"{(failed or missing)[:3]}). Run the check live without --batch.")
    answers = [by_id[r["custom_id"]] for r in requests]
    cut = [r["custom_id"] for r, a in zip(requests, answers) if a["stop_reason"] == "max_tokens"]
    if cut:
        run.warn(f"{len(cut)} judge answers were cut off by max_tokens: {cut[:5]}")

    # 4. Replay: the normal scoring code, fed the batch answers in order.
    pos = {"i": 0}

    def replay(model, messages, response_format, kw):
        i = pos["i"]
        pos["i"] += 1
        if i >= len(keys) or _key(_build_params(model, messages, response_format, kw)) != keys[i]:
            raise RuntimeError(f"judge call {i} doesn't match the request sent in the batch; "
                               "the input files changed. Run again without --batch-id.")
        return {"role": "assistant", "content": answers[i]["text"]}

    with _Patch(metrics, replay):
        fc.score_all(gold_path, out_path, _ReplayRun(run, answers))
    if pos["i"] != len(answers):
        raise RuntimeError(f"replay used {pos['i']} of {len(answers)} batch answers")
