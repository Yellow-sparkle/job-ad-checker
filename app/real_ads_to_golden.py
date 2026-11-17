"""Turn Kate's real-ads sheet into a file the export can run on.

    python3 app/real_ads_to_golden.py
    python3 app/export_eval.py --golden golden-set/real-ads-v1.jsonl --out eval/real-ads-run1.jsonl

Reads golden-set/real-ads-v1.xlsx (sheet "Ads") and writes golden-set/real-ads-v1.jsonl,
one line per filled-in ad, in the golden-set format. Kate's verdict becomes the expected
verdict. "Unsure" ads are run but not counted. There are no tap answers, so the checker
reads the text alone, like a first-time user who skips the questions.
Plain code: no API calls. Keep both files private (golden-set/ is never shared).
"""
import json
import os
import sys

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEET = os.path.join(ROOT, "golden-set", "real-ads-v1.xlsx")
OUT = os.path.join(ROOT, "golden-set", "real-ads-v1.jsonl")
VERDICT = {"skip": "Skip", "be careful": "Be careful", "no red flags": "No red flags found",
           "no red flags found": "No red flags found", "unsure": "Unsure"}


def main():
    ws = openpyxl.load_workbook(SHEET, data_only=True)["Ads"]
    rows, skipped = [], []
    for r in ws.iter_rows(min_row=2, values_only=True):
        cid, text, source, country, verdict, why, found = (list(r) + [None] * 7)[:7]
        if not cid or not (text or "").strip():
            continue
        v = VERDICT.get(str(verdict or "").strip().lower())
        if not v:
            skipped.append(cid)
            continue
        counted = v != "Unsure"
        ground = {"Skip": "Scam", "Be careful": "Unclear", "No red flags found": "Real"}.get(v, "Unclear")
        rows.append({
            "case_id": cid, "category": "Real ad", "subtype": f"{source or 'unknown source'}, {country or 'unknown country'}",
            "input_type": "", "paste": "Full", "input_text": str(text).strip(),
            "tap_who_contacted": "Not sure", "tap_money_or_id": "Not sure", "tap_interview_stage": "Not sure",
            "tap_posting_age_days": None,
            "ground_truth": ground, "expected_verdict": v if counted else "No red flags found",
            "acceptable_verdicts": v if counted else "No red flags found | Be careful | Skip",
            "must_fire": "", "must_not_fire": "", "expected_ghost_signals": "", "expected_extraction": "",
            "case_pass_criteria": f"Matches Kate's verdict. Why: {why or ''}".strip(),
            "provenance": "Real ad, pasted by Kate", "counts_at_gate0": "Y" if counted else "N",
            "source": source or "", "country": country or "", "date_found": str(found or ""),
        })
    if not rows:
        sys.exit("No filled-in ads with a verdict yet. Fill in the 'Ads' sheet first.")
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} ads to {os.path.relpath(OUT, ROOT)} "
          f"({sum(r['counts_at_gate0'] == 'Y' for r in rows)} counted, "
          f"{sum(r['counts_at_gate0'] == 'N' for r in rows)} 'Unsure').")
    if skipped:
        print(f"Left out (ad text but no verdict yet): {', '.join(skipped)}")
    print("Next: python3 app/export_eval.py --golden golden-set/real-ads-v1.jsonl --out eval/real-ads-run1.jsonl")


if __name__ == "__main__":
    main()
