""""Same ad, same verdict" check for Job Ad Checker (agreed with Kate 2026-11-12).

Each test ad goes through the checker several times (3 by default). This script
compares the verdicts across those runs. It is plain code: no AI judge, no API
key, nothing leaves the machine.

When to run (all 65 cases, 3 runs each, about 195 checker calls):
  - before Gate 0 (by 15 Nov)
  - after any change to the checker's prompt or model
  - before the 19 Nov code freeze
During the beta, run weekly with --beta (GS-R01 plus the weak-flag boundary
cases, about 15 calls) and --quiet, so it prints nothing unless something blocks.

First set the checker's own temperature to 0 (in the app's code).

What counts:
  BLOCKS GATE 0   a case with a sure or strong flag (must_fire has SURE_ or
                  STRONG_), or GS-R15 (Kate, 12 Nov), falls below "Be careful"
                  in any run.
  KNOWN RISK      any other warning case (No red flags found is not an
                  accepted verdict) gives "No red flags found" in any run, for
                  example GS-R14. Kate must mark it "accept" or "fix" before
                  Gate 0. It goes to the top of the report.
  DEMO CHECK      GS-R01 (Demo Day) must give "Skip" in every run before 19 Nov.
  OUTSIDE ACCEPTED  a verdict outside acceptable_verdicts that is none of the
                  above (for example a genuine ad shown as "Be careful"). The
                  normal single-pass checks already fail it; listed for review.
  LOGGED          a change between two accepted verdicts (for example Skip and
                  "Be careful" on GS-R01), or a change in which rules fired.
                  Information only.
GS-E19 counts from golden set v0.6 (it was a known gap before WEAK_NO_EMPLOYER_NAMED).

Usage:
  # each run file is a tool_outputs.jsonl from one full pass of the checker
  python stability_check.py golden-set-v0.14.jsonl run1.jsonl run2.jsonl run3.jsonl
  python stability_check.py golden-set-v0.14.jsonl run1.jsonl run2.jsonl run3.jsonl --beta --quiet
FREEZE CASE      a case marked counts_at_gate0 = N (GS-E27 to E30, Kate 14 Nov) that
                 would otherwise block. Listed for the 19 Nov freeze; doesn't block Gate 0.
Writes stability_report.csv (known risks first, with an empty "Kate: accept or
fix" column), run_info.json and usage.csv to its own folder, eval/runs/<run_id>/
(see run_record.py). Options: --dry-run, --run-dir <folder>.
Exit code 1 if anything blocks Gate 0, otherwise 0.
"""
import csv, json, sys
from run_record import start_run, parse_args

BETA_CASES = ["GS-R01", "GS-R14", "GS-R15", "GS-E02", "GS-E06"]  # demo + weak-flag boundary
DEMO_CASE = "GS-R01"
MUST_WARN = {"GS-R15"}  # Kate 2026-11-12: "No red flags found" here blocks Gate 0, not a known risk
NOT_COUNTED = {}  # GS-E19 was a known gap until v0.6 (WEAK_NO_EMPLOYER_NAMED); put it back here if that sign misses Gate 0
WARNING_RANK = {"Skip": 2, "Be careful": 1}  # anything else ranks 0
NRF = "No red flags found"
ORDER = {"BLOCKS GATE 0": 0, "KNOWN RISK": 1, "DEMO CHECK FAILED": 2,
         "OUTSIDE ACCEPTED": 3, "FREEZE CASE": 4, "LOGGED": 5, "STABLE": 6, "NOT COUNTED": 7}


def split_list(s, sep=","):
    return [x.strip() for x in (s or "").split(sep) if x.strip()]


def load_run(path):
    return {o["case_id"]: o for o in map(json.loads, filter(str.strip, open(path)))}


def classify(case, outs):
    """outs: one tool output per run (None if the case is missing from a run)."""
    cid = case["case_id"]
    if cid in NOT_COUNTED:
        return ["NOT COUNTED"], NOT_COUNTED[cid]
    verdicts = [o["verdict"] if o else "(missing)" for o in outs]
    accepted = split_list(case["acceptable_verdicts"], "|")
    sure_strong = any(r.startswith(("SURE_", "STRONG_")) for r in split_list(case["must_fire"]))
    labels, notes = [], []

    if (sure_strong or cid in MUST_WARN) and any(WARNING_RANK.get(v, 0) < 1 for v in verdicts):
        labels.append("BLOCKS GATE 0")
        notes.append("must-warn case fell below 'Be careful'" if cid in MUST_WARN and not sure_strong
                     else "sure or strong case fell below 'Be careful'")
    elif NRF not in accepted and case["expected_verdict"] in WARNING_RANK and NRF in verdicts:
        labels.append("KNOWN RISK")
        notes.append(f"warning case shown as '{NRF}' in {verdicts.count(NRF)} of {len(verdicts)} runs")
    if cid == DEMO_CASE and any(v != "Skip" for v in verdicts):
        labels.append("DEMO CHECK FAILED")
        notes.append(f"'Skip' in {verdicts.count('Skip')} of {len(verdicts)} runs, needs all")
    if not labels and any(v not in accepted for v in verdicts):
        labels.append("OUTSIDE ACCEPTED")
        notes.append("verdict outside acceptable_verdicts")

    fired = [tuple(sorted((o or {}).get("fired", []))) for o in outs]
    if not labels and (len(set(verdicts)) > 1 or len(set(fired)) > 1):
        labels.append("LOGGED")
        if len(set(verdicts)) > 1:
            notes.append("moved between accepted verdicts")
        if len(set(fired)) > 1:
            notes.append("rules fired changed")
    if not counts_at_gate0(case) and "BLOCKS GATE 0" in labels:
        labels = ["FREEZE CASE" if l == "BLOCKS GATE 0" else l for l in labels]
        notes.append("would block, but this freeze case doesn't count at Gate 0 (Kate, 14 Nov)")
    return labels or ["STABLE"], "; ".join(notes)


def counts_at_gate0(case):
    """False for freeze cases (Kate, 14 Nov). Golden set field (v0.14+), else its notes."""
    v = case.get("counts_at_gate0")
    if v is not None:
        return str(v).strip().upper() != "N"
    return "not counted at gate 0" not in (case.get("validator_notes") or "").lower()


def main(argv):
    args, opts = parse_args(argv)
    flags = opts["flags"]
    if len(args) < 3:
        sys.exit(__doc__)
    code = {"value": 0}
    with start_run("stability_check", args[0], args[1:], run_dir=opts["run_dir"],
                   dry_run=opts["dry_run"]) as run:
        code["value"] = check(args, flags, run)
    return code["value"]


def check(args, flags, run):
    gold = [json.loads(l) for l in open(args[0]) if l.strip()]
    runs = [load_run(p) for p in args[1:]]
    for r in runs:
        for cid, o in r.items():
            run.add_checker_usage(cid, o)
    if run.dry_run:
        print(f"Dry run: {len(gold)} golden cases, {len(runs)} run files read, no comparison made.")
        return 0
    if "--beta" in flags:
        gold = [c for c in gold if c["case_id"] in BETA_CASES]

    rows = []
    for case in gold:
        outs = [r.get(case["case_id"]) for r in runs]
        labels, note = classify(case, outs)
        verdicts = [o["verdict"] if o else "(missing)" for o in outs]
        rows.append({"status": " + ".join(labels), "case_id": case["case_id"],
                     "accepted": case["acceptable_verdicts"],
                     **{f"run {i + 1}": v for i, v in enumerate(verdicts)},
                     "note": note,
                     "Kate: accept or fix": "" if "KNOWN RISK" in labels else "n/a"})
    rows.sort(key=lambda r: min(ORDER[s] for s in r["status"].split(" + ")))

    with open(run.path("stability_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["status"])
        w.writeheader()
        w.writerows(rows)

    blocking = [r for r in rows if "BLOCKS GATE 0" in r["status"]]
    if "--quiet" not in flags or blocking:
        counts = {}
        for r in rows:
            for s in r["status"].split(" + "):
                counts[s] = counts.get(s, 0) + 1
        print(f"{len(rows)} cases x {len(runs)} runs")
        for s in sorted(counts, key=ORDER.get):
            print(f"  {s}: {counts[s]}")
        for r in rows:
            if r["status"] not in ("STABLE", "NOT COUNTED"):
                runs_txt = " / ".join(r[f"run {i + 1}"] for i in range(len(runs)))
                print(f"{r['status']}\t{r['case_id']}\t{runs_txt}\t{r['note']}")
        print(f"Report: {run.path('stability_report.csv')}")
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
