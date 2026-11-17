# How to run the evaluations

Each run gets its own folder, so you can always see what was run and compare two runs. The test sets (`golden-set/`) and run folders (`eval/runs/`) are kept private, so the commands below show the shape of a run.

## 1. Run a check

From the `eval` folder, on a test set and the checker's outputs for it (made with `app/export_eval.py`):

```
python extra_checks.py <golden set>.jsonl run1.jsonl run2.jsonl run3.jsonl
python stability_check.py <golden set>.jsonl run1.jsonl run2.jsonl run3.jsonl
python faithfulness_check.py <golden set>.jsonl run1.jsonl
```

Add `--dry-run` to check that the files load and see the setup, without calling the AI judge (costs nothing).
Add `--run-dir runs/my-run` to choose the folder name yourself.

## 2. What you get

A new folder `eval/runs/<date-time>-<script>-<golden version>/` with:

- `run_info.json`: test-set version and fingerprint, judge model and prompt versions, thresholds, checker version, warnings, and totals for time, tokens and cost
- `usage.csv`: one row per judge call or checker result, with seconds, tokens and cost (numbers only, no ad text)
- the script's results (`extra_check_results.json`, `stability_report.csv`, `faithfulness_results.json`, `human_review.csv`)

Costs come from LiteLLM's price list, so treat them as estimates.

## 3. Compare two runs

```
python compare_runs.py <run A folder> <run B folder>
```

Changes are marked with `*`, and checks that flipped between pass and fail are listed by ad.

## Good to know

- **Worst of 3 runs.** AI answers vary between runs, so every result is reported on the worst of 3 runs.
- **Half-price judge runs.** `--batch` sends all judge calls as one Message Batch: half the price, same scores. Results usually come back within an hour; if the script times out, it prints the command to pick the batch up later (`--batch-id <id>`).
- **Quoted ad text:** the word checks for "safe", "verified", "approved" and "great place to work" ignore quotes of 3+ words copied from the ad, so a warning isn't failed for the ad's own words.
- **Temperature:** Claude Sonnet 5.5 doesn't accept a temperature setting, so the judge's scores may vary a little between runs. The stability check shows how much.
- **Changing a judge prompt:** also change `PROMPT_VERSION` at the top of that judge's file. If you forget, the next run warns you.
- **Privacy:** run files hold ids and numbers only. Opik's error reporting and usage stats are switched off.
- **Tests:** `python -m pytest tests -q` (no API key needed).
