"""Faithfulness and relevance check for Job Ad Checker results (draft, not yet run).

Context Precision: grades the checker's extracted details (company, contact,
posting age, stage, and any money or ID requested) against the golden set's
expected_extraction, one field at a time (job_ad_extraction.py). Score = share
of expected fields right. Pay is not graded (salary dropped, Kate 2026-11-12).
Feeds the 'Fields right' column.

Answer Relevance: scores the whole result the user sees (verdict + reasons +
"what was checked" text) against Kate's five criteria (job_ad_relevance.py,
approved 2026-11-12) for whether it helps Mia decide on this ad. 0.0 = off-topic, 1.0 = fully relevant. Threshold is a starting
guess (RELEVANCE_MIN) to be calibrated against Kate's labels.

Scores each LLM-written reason (pattern note + quoted words, and extracted
fields) against the ad text with Opik's Hallucination metric.
Score 0 = faithful, 0.5 = distorted, 1 = invented. Any score of 0.5 or more,
or a quote not found in the ad, feeds the 'Unfaithful reason?' column (Q) in
the human eval kit, with a sub-label: Invented (1.0 or made-up quote) or
Distorted (0.5). The two rates are reported separately. The human label stays
the Gate 0 decision.

Only run on golden-set cases (all authored/fictional) until Kate approves
anything else. Real user ads must never go through this script.

Usage:
  pip install opik
  export ANTHROPIC_API_KEY=...   # judge: Claude Sonnet 5.5 (chosen by Kate)
  python faithfulness_check.py golden-set-v0.13.jsonl tool_outputs.jsonl
  python faithfulness_check.py golden-set-v0.13.jsonl tool_outputs.jsonl --dry-run
  python faithfulness_check.py golden-set-v0.13.jsonl tool_outputs.jsonl --run-dir runs/my-run
  python faithfulness_check.py golden-set-v0.14.jsonl tool_outputs.jsonl --batch
    --batch: send all judge calls as one Message Batch (half price, judge prompt
    cached, same scores; usually back within an hour). See judge_batch.py.
    --batch-id <id>: pick up a batch that was already sent. --batch-wait <minutes>
    (default 90). --no-cache: batch without prompt caching.
Each run writes its files to its own folder, eval/runs/<run_id>/ (see run_record.py):
run_info.json, usage.csv, faithfulness_results.json and human_review.csv.
--dry-run checks the files and writes run_info.json without calling the judge.
tool_outputs.jsonl: one line per case:
  {"case_id": "GS-R01", "verdict": "Skip", "result_text": "<full result as shown>",
   "extraction": "company=...; contact=...; posted=...; stage=...",
   "reasons": [{"rule_id": "...", "note": "...", "quote": "..."}]}
  Optional per line: "usage": {"input_tokens": 1234, "output_tokens": 210,
   "latency_s": 3.4, "cost_usd": 0.0021}, "checker_version", "checker_prompt_version".
"""
import csv, json, os, random, re, sys
from run_record import start_run, parse_args  # first: switches off Opik telemetry
import job_ad_faithfulness, job_ad_relevance, job_ad_extraction
from job_ad_faithfulness import JobAdFaithfulness, build_context
from job_ad_relevance import JobAdRelevance
from job_ad_extraction import JobAdExtractionPrecision

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "anthropic/claude-sonnet-5-5")
# Temperature 0 keeps the judges' scores as repeatable as possible, so a change
# between runs comes from the checker, not from the judge (Kate, 2026-11-12).
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.0"))
# track=False: scores are not logged to Opik cloud (Comet) from here.
metric = JobAdFaithfulness(model=JUDGE_MODEL, track=False, temperature=JUDGE_TEMPERATURE)  # project prompt, see job_ad_faithfulness.py
relevance = JobAdRelevance(model=JUDGE_MODEL, track=False, temperature=JUDGE_TEMPERATURE)  # Kate's criteria, see job_ad_relevance.py
RELEVANCE_MIN = float(os.environ.get("RELEVANCE_MIN", "0.7"))
precision = JobAdExtractionPrecision(model=JUDGE_MODEL, track=False, temperature=JUDGE_TEMPERATURE)  # field-by-field, see job_ad_extraction.py
PRECISION_MIN = float(os.environ.get("PRECISION_MIN", "0.8"))
# Judge effort (Kate's cheapest-model rule, 14 Nov). Unset = the model's default, as before.
# JUDGE_EFFORT=low cuts the judge's hidden reasoning, which is most of its output cost.
JUDGE_EFFORT = os.environ.get("JUDGE_EFFORT", "").strip() or None
if JUDGE_EFFORT:
    for _m in (metric, relevance, precision):
        _m._model._reasoning_effort = JUDGE_EFFORT
EXTRACT_INPUT = ("Extract company, contact channel, posting age, interview stage, "
                 "and any money or ID the job seeker is asked for, from this job ad "
                 "or recruiter message. Do not extract salary.")

INPUT = ("Check this job ad or recruiter message for scam and ghost-job "
         "signs before the job seeker applies.")


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def context_for(case):
    return build_context(case)


JUDGES = {"faithfulness": job_ad_faithfulness, "relevance": job_ad_relevance,
          "extraction": job_ad_extraction}


def main(gold_path, out_path, run_dir=None, dry_run=False, batch=False, batch_id=None,
         batch_wait=90, prompt_cache=True):
    with start_run("faithfulness_check", gold_path, [out_path], judges=JUDGES,
                   model=JUDGE_MODEL, temperature=JUDGE_TEMPERATURE,
                   thresholds={"hallucination_unfaithful": 0.5, "invented": 0.75,
                               "relevance_min": RELEVANCE_MIN, "precision_min": PRECISION_MIN},
                   run_dir=run_dir, dry_run=dry_run) as run:
        run.info["judge_effort"] = JUDGE_EFFORT or "model default"
        if batch and not dry_run:
            import judge_batch
            judge_batch.score_all_batch(sys.modules[__name__], gold_path, out_path, run,
                                        batch_id=batch_id, wait_minutes=batch_wait,
                                        use_cache=prompt_cache)
        else:
            run.info["judge_mode"] = {"mode": "live"}
            score_all(gold_path, out_path, run)


def score_all(gold_path, out_path, run):
    gold = {c["case_id"]: c for c in map(json.loads, filter(str.strip, open(gold_path)))}
    rows = []
    for line in filter(str.strip, open(out_path)):
        out = json.loads(line)
        case = gold[out["case_id"]]
        run.add_checker_usage(case["case_id"], out)
        if run.dry_run:
            continue
        unfaithful = False
        for r in out["reasons"]:
            # 1. Free deterministic check: the quote must be in the ad.
            quote_ok = (not r.get("quote")) or norm(r["quote"]) in norm(case["input_text"])
            # 2. LLM judge on the note + quote.
            text = f"{r.get('note', '')} Quoted: \"{r.get('quote', '')}\""
            res = run.timed(case["case_id"], "faithfulness", metric.score,
                            input=INPUT, output=text, context=context_for(case))
            bad = (not quote_ok) or res.value >= 0.5
            kind = unfaithful_kind(quote_ok, res.value)
            unfaithful |= bad
            rows.append({"case_id": case["case_id"], "rule_id": r.get("rule_id"), "output": text,
                         "quote_in_ad": quote_ok, "judge_score": res.value,
                         "judge_reason": res.reason, "unfaithful": bad, "unfaithful_kind": kind})
        rel = run.timed(case["case_id"], "relevance", relevance.score,
                        input=INPUT, output=out.get("result_text", ""), context=context_for(case))
        rows.append({"case_id": case["case_id"], "rule_id": "_RESULT_RELEVANCE", "output": out.get("result_text", ""),
                     "relevance_score": rel.value, "relevance_reason": rel.reason,
                     "relevant": rel.value >= RELEVANCE_MIN})
        if case.get("expected_extraction"):
            pre = run.timed(case["case_id"], "extraction", precision.score,
                            input=EXTRACT_INPUT, output=out.get("extraction", ""),
                            expected_output=case["expected_extraction"],
                            context=context_for(case))
            rows.append({"case_id": case["case_id"], "rule_id": "_EXTRACTION_PRECISION",
                         "output": out.get("extraction", ""),
                         "expected": case["expected_extraction"],
                         "precision_score": pre.value, "precision_reason": pre.reason})
        print(f"{case['case_id']}\tUnfaithful reason? {'Y' if unfaithful else 'N'}"
              f"\tRelevance {rel.value:.2f}{'' if rel.value >= RELEVANCE_MIN else ' LOW'}")
    if run.dry_run:
        print(f"Dry run: {len(gold)} golden cases, files read, no judge called.")
        return
    print_rates(rows)
    json.dump(rows, open(run.path("faithfulness_results.json"), "w"), indent=2)
    write_human_review(rows, gold, run.path("human_review.csv"))


def unfaithful_kind(quote_ok, score):
    """'invented' (made-up quote or score 0.75+), 'distorted' (0.5 to 0.75), or ''."""
    if not quote_ok or score >= 0.75:
        return "invented"
    if score >= 0.5:
        return "distorted"
    return ""


def print_rates(rows):
    """Report invented and distorted reasons separately, so one number doesn't hide two failures."""
    judged = [r for r in rows if "judge_score" in r]
    if not judged:
        return
    n = len(judged)
    inv = sum(r["unfaithful_kind"] == "invented" for r in judged)
    dis = sum(r["unfaithful_kind"] == "distorted" for r in judged)
    print(f"Reasons judged: {n}\tInvented: {inv} ({inv / n:.0%})\tDistorted: {dis} ({dis / n:.0%})")


def write_human_review(rows, gold, path="human_review.csv"):
    """Sheet for Kate: flagged rows, unsure rows near the line, and a spot-check of passes."""
    def kind(r):
        if "judge_score" in r:  # hallucination row
            v = r["judge_score"]
            if r["unfaithful"]:
                label = "invented" if r["unfaithful_kind"] == "invented" else "distorted"
                return (f"judge flagged ({label})",
                        "Is this reason really invented or distorted, compared with the ad or taps?")
            if 0.3 <= v < 0.5:
                return "judge unsure", "Is this reason backed by the ad or taps?"
        elif "precision_score" in r:  # extraction row
            v = r["precision_score"]
            if v < PRECISION_MIN - 0.2:
                return "judge flagged", f"Are the details wrong vs expected ({r['expected']})?"
            if v < PRECISION_MIN:
                return "judge unsure", f"Do the details match expected ({r['expected']}) closely enough?"
        else:  # relevance row
            v = r["relevance_score"]
            if v < RELEVANCE_MIN - 0.1:
                return "judge flagged", "Does this result fail to help Mia decide on this ad?"
            if v < RELEVANCE_MIN + 0.1:
                return "judge unsure", "Does this result help Mia decide on this ad?"
        return None, None

    out, passed = [], []
    for r in rows:
        why, ask = kind(r)
        (out.append((why, ask, r)) if why else passed.append(r))
    random.seed(1)
    for r in random.sample(passed, min(10, len(passed))):
        out.append(("spot-check pass", "Do you agree the judge was right to pass this?", r))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["why_in_sheet", "case_id", "rule_id", "what_to_decide", "checker_output", "ad_text",
                    "judge_score", "judge_reason", "kate_agrees_with_judge (Y/N)", "kate_note"])
        for why, ask, r in out:
            w.writerow([why, r["case_id"], r["rule_id"], ask, r["output"], gold[r["case_id"]]["input_text"],
                        r.get("judge_score", r.get("relevance_score", r.get("precision_score"))),
                        r.get("judge_reason", r.get("relevance_reason", r.get("precision_reason"))), "", ""])


if __name__ == "__main__":
    argv, batch_id, batch_wait = list(sys.argv[1:]), None, 90
    for name in ("--batch-id", "--batch-wait"):
        if name in argv:
            i = argv.index(name)
            if i + 1 >= len(argv):
                sys.exit(f"{name} needs a value")
            val = argv.pop(i + 1)
            argv.pop(i)
            if name == "--batch-id":
                batch_id = val
            else:
                batch_wait = float(val)
    pos, opts = parse_args(argv)
    if len(pos) != 2:
        sys.exit(__doc__)
    flags = opts["flags"]
    main(pos[0], pos[1], run_dir=opts["run_dir"], dry_run=opts["dry_run"],
         batch="--batch" in flags or batch_id is not None, batch_id=batch_id,
         batch_wait=batch_wait, prompt_cache="--no-cache" not in flags)
