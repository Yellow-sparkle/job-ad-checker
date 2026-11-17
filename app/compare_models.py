"""Compare two checker runs on the same golden set, for example Sonnet against Haiku.

    python3 app/compare_models.py eval/run1.jsonl eval/run1-haiku.jsonl
    python3 app/compare_models.py eval/run1.jsonl eval/run1-haiku.jsonl --golden golden-set/golden-set-v0.14.jsonl

Kate's rule (14 Nov): use the cheapest model unless it does measurably worse.
"Worse" here means, on the ads that count at Gate 0: fewer accepted verdicts, a
missed must-fire sign, a must-not-fire sign that fired, or a scam case shown
softer than expected. Judge scores (faithfulness_check.py) are compared
separately with eval/compare_runs.py. Plain code: reads local files only.
"""
import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANK = {"Skip": 2, "Be careful": 1}


def split(s, sep=","):
    return [x.strip() for x in (s or "").split(sep) if x.strip()]


def counts(case):
    v = case.get("counts_at_gate0")
    return str(v).strip().upper() != "N" if v is not None else True


def score(gold, run):
    s = {"cases": 0, "accepted": 0, "must_fire_missed": [], "must_not_fired": [], "softer": [],
         "cost": 0.0, "latency": [], "model": set()}
    for cid, c in gold.items():
        o = run.get(cid)
        if o is None or not counts(c):
            continue
        s["cases"] += 1
        fired = set(o.get("fired", []))
        if o["verdict"] in split(c["acceptable_verdicts"], "|"):
            s["accepted"] += 1
        s["must_fire_missed"] += [f"{cid} {r}" for r in split(c["must_fire"]) if r not in fired]
        s["must_not_fired"] += [f"{cid} {r}" for r in split(c["must_not_fire"]) if r in fired]
        if RANK.get(o["verdict"], 0) < RANK.get(c["expected_verdict"], 0):
            s["softer"].append(f"{cid}: {o['verdict']} (expected {c['expected_verdict']})")
        u = o.get("usage") or {}
        s["cost"] += u.get("cost_usd") or 0
        if u.get("latency_s") is not None:
            s["latency"].append(u["latency_s"])
        if u.get("model"):
            s["model"].add(u["model"])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--golden", default=os.path.join(ROOT, "golden-set", "golden-set-v0.13.jsonl"))
    a = ap.parse_args()
    gold = {c["case_id"]: c for c in map(json.loads, filter(str.strip, open(a.golden, encoding="utf-8")))}
    runs = [{o["case_id"]: o for o in map(json.loads, filter(str.strip, open(p, encoding="utf-8")))}
            for p in (a.run_a, a.run_b)]
    res = [score(gold, r) for r in runs]

    names = [f"{os.path.basename(p)} ({', '.join(sorted(r['model'])) or 'model not recorded'})"
             for p, r in zip((a.run_a, a.run_b), res)]
    print(f"Golden set {os.path.basename(a.golden)}, ads that count at Gate 0 only\n")
    rows = [("Accepted verdicts", lambda r: f"{r['accepted']} of {r['cases']}"),
            ("Must-fire signs missed", lambda r: str(len(r["must_fire_missed"]))),
            ("Must-not-fire signs fired", lambda r: str(len(r["must_not_fired"]))),
            ("Scam cases shown softer", lambda r: str(len(r["softer"]))),
            ("Cost of the run", lambda r: f"${r['cost']:.3f}"),
            ("Median seconds per ad", lambda r: f"{statistics.median(r['latency']):.1f}" if r["latency"] else "-")]
    print(f"{'':28}{'A':>22}{'B':>22}")
    for label, f in rows:
        print(f"{label:28}{f(res[0]):>22}{f(res[1]):>22}")
    print(f"\nA = {names[0]}\nB = {names[1]}")

    for i, tag in ((0, "A"), (1, "B")):
        for key, label in (("must_fire_missed", "missed"), ("must_not_fired", "fired but must not"),
                           ("softer", "softer than expected")):
            if res[i][key]:
                print(f"{tag} {label}: {'; '.join(res[i][key])}")

    diff = [cid for cid in gold if cid in runs[0] and cid in runs[1]
            and runs[0][cid]["verdict"] != runs[1][cid]["verdict"]]
    print(f"\nAds where A and B gave different verdicts ({len(diff)}):")
    for cid in diff:
        print(f"  {cid}: A {runs[0][cid]['verdict']} | B {runs[1][cid]['verdict']} "
              f"| accepted {gold[cid]['acceptable_verdicts']}")

    ra, rb = res
    worse = (rb["accepted"] < ra["accepted"] or len(rb["must_fire_missed"]) > len(ra["must_fire_missed"])
             or len(rb["must_not_fired"]) > len(ra["must_not_fired"]) or len(rb["softer"]) > len(ra["softer"]))
    print("\nVerdict check: " + ("B does worse than A on these ads, so keep A." if worse else
          "B does no worse than A on verdicts and signs. If B is cheaper, compare the judge scores "
          "next (eval/compare_runs.py), then repeat over 3 runs before switching."))


if __name__ == "__main__":
    sys.exit(main())
