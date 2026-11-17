"""Check the three made-up sample ads once and save their results.

    python3 app/save_samples.py

The app then shows these saved results when someone clicks a "Try a sample ad"
button and keeps the suggested answers: no AI call, no cost, and it works
without internet (useful for a demo). The samples are made up (samples/sample-ads.json),
so nothing personal is saved. Run it again after changing the checker's instructions;
the app ignores saved results made with older instructions.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import checker  # noqa: E402


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY isn't set. Run step 3 in app/README.md first.")
    with open(os.path.join(ROOT, "samples", "sample-ads.json"), encoding="utf-8") as f:
        samples = json.load(f)["samples"]
    out = {"checker_prompt_version": checker.PROMPT_VERSION, "checker_version": checker.CHECKER_VERSION,
           "model": os.environ.get("CHECKER_MODEL", checker.DEFAULT_MODEL), "samples": []}
    for s in samples:
        taps = dict(s["suggested_taps"])
        r = checker.check(s["text"], taps)
        ok = "ok" if r["verdict"] == s["expected"]["verdict"] else f"CHECK (expected {s['expected']['verdict']})"
        print(f"{s['button']}: {r['verdict']} {ok}")
        out["samples"].append({"id": s["id"], "text": s["text"], "taps": taps, "result": r})
    with open(os.path.join(HERE, "sample_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("Saved app/sample_results.json")


if __name__ == "__main__":
    main()
