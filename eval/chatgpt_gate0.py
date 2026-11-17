"""Final Gate 0 score: the checker's 3 runs against the ChatGPT baseline.

    python3 eval/chatgpt_gate0.py

Reads golden-set/chatgpt-baseline.xlsx (ChatGPT's verdict per ad, filled in by hand),
the checker's run files eval/run1-3.jsonl and their judge folders, and prints the seven
Gate 0 bars on the worst of 3 runs plus GO or NO GO. It also writes
golden-set/job-ad-checker-human-eval-kit-v0.12.xlsx: a copy of kit v0.11 with ChatGPT's
verdicts in 'Step 3 Run and score' column Y and the numbers typed into
'Gate 0 over 3 runs'. Kit v0.11 itself isn't changed. Plain code: no API calls.

Options: --judge-dirs runs/gate0-f-judge-run1,runs/gate0-f-judge-run2,runs/gate0-f-judge-run3
         --extra-dir runs/gate0-f-extra
"""
import argparse
import json
import os
import sys

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "golden-set", "golden-set-v0.14.jsonl")
BASELINE = os.path.join(ROOT, "golden-set", "chatgpt-baseline.xlsx")
KIT_IN = os.path.join(ROOT, "golden-set", "job-ad-checker-human-eval-kit-v0.11.xlsx")
KIT_OUT = os.path.join(ROOT, "golden-set", "job-ad-checker-human-eval-kit-v0.12.xlsx")
VERDICTS = ["No red flags found", "Be careful", "Skip"]


def norm_verdict(text):
    """ChatGPT's verdict from a dropdown value or its pasted first line."""
    t = (text or "").strip().lower()
    hits = [(t.find(v.lower()), v) for v in VERDICTS if v.lower() in t]
    return min(hits)[1] if hits else ""


def accepted(case):
    return [v.strip() for v in case["acceptable_verdicts"].split("|")]


def counts(case):
    return str(case.get("counts_at_gate0", "Y")).strip().upper() != "N"


def bars_for(verdicts, gold):
    """The kit's Summary formulas B8 to B10 (or C8 to C10), on ads that count at Gate 0."""
    ids = [c for c in gold if counts(gold[c]) and verdicts.get(c)]
    scams = [c for c in ids if gold[c]["ground_truth"] == "Scam"]
    real = [c for c in ids if gold[c]["ground_truth"] == "Real"]
    judged = [c for c in ids if gold[c]["ground_truth"] != "Unclear"]
    return {
        "scams": sum(verdicts[c] in ("Be careful", "Skip") for c in scams) / len(scams) if scams else None,
        "real_skip": sum(verdicts[c] == "Skip" for c in real) / len(real) if real else None,
        "right": sum(verdicts[c] in accepted(gold[c]) for c in judged) / len(judged) if judged else None,
        "n": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-dirs", default="runs/gate0-f-judge-run1,runs/gate0-f-judge-run2,runs/gate0-f-judge-run3")
    ap.add_argument("--extra-dir", default="runs/gate0-f-extra")
    a = ap.parse_args()
    ev = os.path.join(ROOT, "eval")
    gold = {c["case_id"]: c for c in map(json.loads, filter(str.strip, open(GOLDEN, encoding="utf-8")))}

    # ChatGPT
    ws = openpyxl.load_workbook(BASELINE, data_only=True).active
    model = ws["B2"].value or "(not given)"
    gpt = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        if row[0]:
            gpt[row[0]] = norm_verdict(row[3])
    missing = [c for c in gold if counts(gold[c]) and not gpt.get(c)]
    g = bars_for(gpt, gold)

    # Checker, 3 runs
    runs = []
    for i, d in enumerate(a.judge_dirs.split(","), 1):
        out = {o["case_id"]: o for o in map(json.loads, filter(str.strip, open(os.path.join(ev, f"run{i}.jsonl"))))}
        b = bars_for({c: o["verdict"] for c, o in out.items()}, gold)
        rows = json.load(open(os.path.join(ev, d, "faithfulness_results.json")))
        ids = [c for c in gold if counts(gold[c]) and c in out]
        inv = {r["case_id"] for r in rows if r.get("unfaithful_kind") == "invented" and r["case_id"] in ids}
        prec = [r["precision_score"] for r in rows if "precision_score" in r and r["case_id"] in ids]
        b["invented"] = len(inv) / len(ids)
        b["details"] = sum(prec) / len(prec) if prec else None
        b["must_pass"] = "Yes" if out["GS-R01"]["verdict"] == "Skip" and out["GS-R15"]["verdict"] in ("Be careful", "Skip") else "No"
        runs.append(b)
    extra = json.load(open(os.path.join(ev, a.extra_dir, "extra_check_results.json")))
    problems = {}
    for it in extra:
        bad = any(v is False for k, v in it.items() if k.startswith(("pii", "no_stereotyping", "no_legal")))
        if bad and counts(gold.get(it.get("case_id"), {})):
            problems[it.get("run")] = problems.get(it.get("run"), 0) + 1
    for i, b in enumerate(runs, 1):
        b["privacy"] = problems.get(i, problems.get(str(i), 0))

    worst = {
        "scams": min(b["scams"] for b in runs), "real_skip": max(b["real_skip"] for b in runs),
        "right": min(b["right"] for b in runs), "invented": max(b["invented"] for b in runs),
        "details": min(b["details"] for b in runs), "privacy": max(b["privacy"] for b in runs),
        "must_pass": "Yes" if all(b["must_pass"] == "Yes" for b in runs) else "No",
    }
    have_gpt = not missing
    checks = [
        ("Scams caught", f"{worst['scams']:.0%}", f"{g['scams']:.0%}" if g["scams"] is not None else "-",
         worst["scams"] >= 0.95 and (g["scams"] is None or worst["scams"] >= g["scams"])),
        ("Real jobs wrongly given Skip", f"{worst['real_skip']:.0%}", f"{g['real_skip']:.0%}" if g["real_skip"] is not None else "-",
         worst["real_skip"] <= 0.05 and (g["real_skip"] is None or worst["real_skip"] <= g["real_skip"])),
        ("Right verdict", f"{worst['right']:.0%}", f"{g['right']:.0%}" if g["right"] is not None else "-",
         g["right"] is not None and worst["right"] - g["right"] >= 0.10),
        ("Ads with an invented reason", f"{worst['invented']:.0%}", "", worst["invented"] == 0),
        ("Details picked out right", f"{worst['details']:.0%}", "", worst["details"] >= 0.90),
        ("Must-pass GS-R01 and GS-R15", worst["must_pass"], "", worst["must_pass"] == "Yes"),
        ("Privacy, fairness or legal problems", str(worst["privacy"]), "", worst["privacy"] == 0),
    ]
    print(f"ChatGPT model: {model}. ChatGPT verdicts for {g['n']} of 61 counted ads.")
    if missing:
        print(f"Still missing ChatGPT verdicts for {len(missing)} ads: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    print(f"\n{'Bar':38}{'Checker (worst of 3)':>22}{'ChatGPT':>10}  Result")
    for name, ours, theirs, ok in checks:
        waiting = theirs == "-" and name == "Right verdict"
        print(f"{name:38}{ours:>22}{theirs:>10}  {'waiting for ChatGPT' if waiting else ('PASS' if ok else 'FAIL')}")
    if have_gpt:
        verdict = "GO" if all(c[3] for c in checks) else "NO GO"
    elif any(not c[3] for c in checks[3:]) or not checks[0][3] or not checks[1][3]:
        verdict = "NO GO"
    else:
        verdict = "not final yet: fill in ChatGPT's verdicts first"
    print(f"\nGATE 0 OVERALL (worst of 3 runs): {verdict}")

    # Write kit v0.12
    kit = openpyxl.load_workbook(KIT_IN)
    st = kit["Step 3 Run and score"]
    for r in range(2, st.max_row + 1):
        cid = st[f"A{r}"].value
        if cid in gpt and gpt[cid]:
            st[f"Y{r}"] = gpt[cid]
    kit["Summary"]["E4"] = model
    g0 = kit["Gate 0 over 3 runs"]
    for i, b in enumerate(runs):
        col = "BCD"[i]
        g0[f"{col}5"], g0[f"{col}6"], g0[f"{col}7"] = b["scams"], b["real_skip"], b["right"]
        g0[f"{col}8"], g0[f"{col}9"], g0[f"{col}10"], g0[f"{col}11"] = b["invented"], b["details"], b["must_pass"], b["privacy"]
    if have_gpt:
        g0["E5"], g0["E6"], g0["E7"] = g["scams"], g["real_skip"], g["right"]
    kit.save(KIT_OUT)
    print(f"\nWrote {os.path.relpath(KIT_OUT, ROOT)} (kit v0.11 is unchanged).")


if __name__ == "__main__":
    sys.exit(main())
