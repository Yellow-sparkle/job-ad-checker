"""Compare two eval runs (fix 1, built 14 Nov 2026).

Shows what changed in the setup (golden set, judges, prompts, thresholds,
checker versions), in time, tokens and cost, and in the results.

Usage:
  python compare_runs.py <run A> <run B>
A run is a folder under eval/runs/ (its full path, or just its name).
Plain code: reads local files only, no API key, nothing is sent anywhere.
"""
import collections, json, os, sys

from run_record import RUNS_DIR


def find(run):
    for p in (run, os.path.join(RUNS_DIR, run)):
        if os.path.isfile(os.path.join(p, "run_info.json")):
            return p
    sys.exit(f"No run_info.json found for {run!r}")


def load(folder, name):
    p = os.path.join(folder, name)
    return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def setup_view(info):
    g = info.get("golden_set") or {}
    view = {"script": info.get("script"), "status": info.get("status"),
            "dry_run": info.get("dry_run"),
            "golden_set.version": g.get("version"), "golden_set.sha256": (g.get("sha256") or "")[:12],
            "golden_set.cases": g.get("cases")}
    for name, j in (info.get("judges") or {}).items():
        for k in ("model", "temperature", "prompt_version"):
            view[f"judge.{name}.{k}"] = j.get(k)
        view[f"judge.{name}.prompt_hash"] = (j.get("prompt_hash") or "")[:12]
    view.update({f"threshold.{k}": v for k, v in (info.get("thresholds") or {}).items()})
    view.update(flat(info.get("checker"), "checker."))
    return view


def pct(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a == 0:
        return ""
    return f" ({(b - a) / a:+.0%})"


def results_view(folder):
    """A few comparable numbers from whichever result files the run has."""
    out = {}
    extra = load(folder, "extra_check_results.json")
    if extra:
        rows = extra
        out["extra_checks.rows"] = len(rows)
        out["extra_checks.rows_all_ok"] = sum(all(v is not False for v in r.values()) for r in rows)
        fails = collections.Counter(k for r in rows for k, v in r.items() if v is False)
        for k, n in sorted(fails.items()):
            out[f"extra_checks.failed.{k}"] = n
    faith = load(folder, "faithfulness_results.json")
    if faith:
        judged = [r for r in faith if "judge_score" in r]
        rel = [r["relevance_score"] for r in faith if "relevance_score" in r]
        pre = [r["precision_score"] for r in faith if "precision_score" in r]
        out["faithfulness.reasons_judged"] = len(judged)
        out["faithfulness.invented"] = sum(r.get("unfaithful_kind") == "invented" for r in judged)
        out["faithfulness.distorted"] = sum(r.get("unfaithful_kind") == "distorted" for r in judged)
        out["relevance.mean"] = round(sum(rel) / len(rel), 3) if rel else None
        out["extraction.mean"] = round(sum(pre) / len(pre), 3) if pre else None
    stab = os.path.join(folder, "stability_report.csv")
    if os.path.isfile(stab):
        import csv
        counts = collections.Counter(s for r in csv.DictReader(open(stab, encoding="utf-8"))
                                     for s in r["status"].split(" + "))
        out.update({f"stability.{k}": v for k, v in sorted(counts.items())})
    return out


def case_flips(a_folder, b_folder):
    """Checks that passed in A and fail in B, or the other way round (extra_checks only)."""
    a, b = load(a_folder, "extra_check_results.json"), load(b_folder, "extra_check_results.json")
    if not a or not b:
        return []
    key = lambda r: (r["case_id"], r.get("run", 1))
    bmap = {key(r): r for r in b}
    flips = []
    for r in a:
        other = bmap.get(key(r))
        if not other:
            continue
        for k, v in r.items():
            w = other.get(k)
            if isinstance(v, bool) and isinstance(w, bool) and v != w:
                flips.append((r["case_id"], k, "now fails" if not w else "now passes"))
    return flips


def show(title, a, b, only_changed=False):
    keys = sorted(set(a) | set(b))
    rows = [(k, a.get(k), b.get(k)) for k in keys if not only_changed or a.get(k) != b.get(k)]
    print(f"\n== {title}")
    if not rows:
        print("  (no differences)")
    for k, x, y in rows:
        mark = "  " if x == y else "* "
        print(f"{mark}{k}: {x}  ->  {y}{pct(x, y) if x != y else ''}")


def main(argv):
    if len(argv) != 2:
        sys.exit(__doc__)
    fa, fb = find(argv[0]), find(argv[1])
    ia, ib = load(fa, "run_info.json"), load(fb, "run_info.json")
    print(f"A: {ia['run_id']}  ({ia.get('started_at')})")
    print(f"B: {ib['run_id']}  ({ib.get('started_at')})")
    show("Setup (* = changed)", setup_view(ia), setup_view(ib))
    show("Time, tokens and cost", flat(ia.get("usage_summary")), flat(ib.get("usage_summary")),
         only_changed=False)
    show("Results", results_view(fa), results_view(fb))
    flips = case_flips(fa, fb)
    if flips:
        print("\n== Checks that changed")
        for cid, check, what in flips:
            print(f"  {cid}\t{check}\t{what}")
    warn = (ia.get("warnings") or []) + (ib.get("warnings") or [])
    if warn:
        print("\n== Warnings recorded in these runs")
        for w in dict.fromkeys(warn):
            print(f"  {w}")


if __name__ == "__main__":
    main(sys.argv[1:])
