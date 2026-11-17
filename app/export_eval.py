"""Run every golden-set ad through the checker and write the eval file.

    python3 app/export_eval.py                        # golden-set-v0.13 -> eval/run1.jsonl
    python3 app/export_eval.py --golden golden-set/golden-set-v0.14.jsonl --out eval/run2.jsonl
    python3 app/export_eval.py --limit 3              # quick try on the first 3 ads
    python3 app/export_eval.py --cases GS-E31,GS-R12 --out eval/smoke1.jsonl   # only some ads
    python3 app/export_eval.py --model claude-haiku-5-5 --out eval/run1-haiku.jsonl

One line per case, in the format eval/faithfulness_check.py and extra_checks.py read:
case_id, verdict, result_text, extraction, reasons[rule_id, note, quote], fired,
raw_llm_output, stored_record, injection_noted, usage, checker_version,
checker_prompt_version. The golden set's taps are used as the answers to the quick
questions. Only made-up golden-set ads go through this script.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import checker  # noqa: E402


def taps_from_case(c):
    age = c.get("tap_posting_age_days")
    return {"who_contacted": c.get("tap_who_contacted"), "money_or_id": c.get("tap_money_or_id"),
            "interview_stage": c.get("tap_interview_stage"),
            "posting_age_days": age if isinstance(age, (int, float)) else None}


def export_line(case, result):
    reasons = []
    for x in result["reasons"]:
        note = x["means"] if x["rule_id"] == "STRONG_LLM_PATTERN" else checker._join(x["title"], x["means"])
        reasons.append({"rule_id": x["rule_id"], "note": note, "quote": x["quote"]})
    for w in result["weak_notes"]:  # a weak sign on its own is shown as a note
        reasons.append({"rule_id": w["rule_id"], "note": checker._join(w["text"], w["means"]), "quote": w["quote"]})
    for s in result["signals"]:
        reasons.append({"rule_id": s["rule_id"], "note": s["text"], "quote": ""})
    return {"case_id": case["case_id"], "verdict": result["verdict"], "result_text": result["result_text"],
            "extraction": result["extraction"], "reasons": reasons, "fired": result["fired"],
            "raw_llm_output": result["raw_llm_output"], "stored_record": "",
            "injection_noted": result["injection_noted"], "usage": result["usage"],
            "checker_version": checker.CHECKER_VERSION, "checker_prompt_version": checker.PROMPT_VERSION}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(ROOT, "golden-set", "golden-set-v0.13.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eval", "run1.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cases", default="", help="only these case ids, comma-separated, e.g. GS-E31,GS-R12")
    ap.add_argument("--model", default=None,
                    help=f"checker model, for example claude-haiku-5-5 (default {checker.DEFAULT_MODEL})")
    args = ap.parse_args()
    if args.model:
        if args.model not in checker.PRICES:
            sys.exit(f"Unknown model {args.model!r}. Use one of: {', '.join(checker.PRICES)}")
        os.environ["CHECKER_MODEL"] = args.model
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY isn't set. Run: export ANTHROPIC_API_KEY=<your key>   then try again.")
    cases = [json.loads(l) for l in open(args.golden, encoding="utf-8") if l.strip()]
    if args.cases:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        unknown = wanted - {c["case_id"] for c in cases}
        if unknown:
            sys.exit(f"Not in the golden set: {', '.join(sorted(unknown))}")
        cases = [c for c in cases if c["case_id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]
    print(f"Checking {len(cases)} ads from {os.path.basename(args.golden)} "
          f"with {os.environ.get('CHECKER_MODEL', checker.DEFAULT_MODEL)}...")
    t0, cost, failed, accepted = time.time(), 0.0, [], 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, c in enumerate(cases, 1):
            result = None
            for attempt in range(3):
                try:
                    result = checker.check(c["input_text"], taps_from_case(c))
                    break
                except checker.CheckerError as e:
                    print(f"  {c['case_id']}: {e} (try {attempt + 1} of 3)")
                    time.sleep(5 * (attempt + 1))
            if result is None:
                failed.append(c["case_id"])
                continue
            out.write(json.dumps(export_line(c, result), ensure_ascii=False) + "\n")
            out.flush()
            cost += (result.get("usage") or {}).get("cost_usd") or 0
            ok = "ok" if result["verdict"] in [v.strip() for v in c["acceptable_verdicts"].split("|")] else "CHECK"
            accepted += ok == "ok"
            print(f"[{i}/{len(cases)}] {c['case_id']}: {result['verdict']}  (expected {c['expected_verdict']}) {ok}")
    print(f"\nDone in {time.time() - t0:.0f}s, about ${cost:.3f}. "
          f"Accepted verdicts: {accepted} of {len(cases) - len(failed)}. Wrote {args.out}")
    if failed:
        print(f"Not written (AI errors): {', '.join(failed)}. Run again to retry.")


if __name__ == "__main__":
    main()
