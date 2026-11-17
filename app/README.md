# Job Ad Checker: the app

A small local web app. Nothing is saved: no database, no log of what you paste, no cookies, no IDs, no telemetry. The only thing that leaves your computer is the redacted ad text, sent to Anthropic's API to be checked.

## Run it

From the project folder:

```bash
pip3 install -r app/requirements.txt
export ANTHROPIC_API_KEY=your-key-here
python3 app/server.py
```

Open http://localhost:8000. Press Ctrl+C in the terminal to stop it.

To keep the key out of your shell history on a Mac, you can enter it hidden instead:

```bash
read -s "ANTHROPIC_API_KEY?Paste your API key and press Enter: " && export ANTHROPIC_API_KEY
```

## Settings (optional)

- `CHECKER_MODEL`: the AI model. Default `claude-sonnet-5-5`. A cheaper option: `claude-haiku-5-5` (or `python3 app/server.py --model claude-haiku-5-5`).
- `PORT`: default 8000.
- `DAILY_AI_CHECKS` (default 200) and `SPEND_CAP_USD` (default 10): spending caps. When one is reached, or the AI is busy, offline or out of credit, the app gives a rules-only result (fixed rules and your answers only, clearly labelled).

## Sample ads and the demo page

The three "Try a sample ad" buttons use made-up ads from `samples/sample-ads.json`.

- `python3 app/save_samples.py` checks them once for real (about 3 cents) and saves the results in `app/sample_results.json`. The buttons then show the saved results, with no AI call and no internet needed.
- `python3 app/build_demo.py` turns those saved results into the static demo page `docs/index.html` (no key, no cost).

Run both again after changing the checker's instructions.

## Tests (no key, no cost)

```bash
python3 -m unittest discover app/tests
```

## Comparing models

```bash
python3 app/export_eval.py --golden <test set>.jsonl --out eval/run1.jsonl
python3 app/export_eval.py --golden <test set>.jsonl --model claude-haiku-5-5 --out eval/run1-haiku.jsonl
python3 app/compare_models.py eval/run1.jsonl eval/run1-haiku.jsonl
```

The test sets themselves are kept private, so they stay a fair test.

## How it works

- `checker.py`: one AI call reads the text and returns findings, each with a quote. A finding whose quote isn't in the text is dropped. Plain-code checks also read the raw text (chat apps, gift cards and crypto, free email for a named company, disguised email addresses, link shorteners), so a line aimed at AI tools can't hide them. Your answers to the quick questions add evidence (money or ID asked before an interview).
- `rules.py`: the rule list, severities and pre-written reasons. The verdict is always set here, never by the AI: any sure sign gives Skip; a strong sign, talent-pool wording or a line telling AI to hide problems gives at least Be careful; two weak signs give Be careful; one weak sign alone stays No red flags found and shows as a note. Posting age is a note only.
- `textmap.py`: removes hidden characters and redacts emails (the domain is kept, so a Gmail address can still be spotted), personal links and phone numbers before the AI call, and maps quotes back to the original text.
- `server.py` and `static/index.html`: the web page. The on-screen legal wording comes from `legal/ui-copy.json`.
- `export_eval.py`: runs the checker over a test set for the evaluation scripts in `eval/`.
