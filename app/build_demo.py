"""Build the free, static demo page: docs/index.html.

    python3 app/save_samples.py     (once, with your API key: checks the 3 sample ads for real)
    python3 app/build_demo.py       (no key, no cost)

The demo is the real app page, but it never calls the AI or any server. It replays the
saved results of the three made-up sample ads from app/sample_results.json, exactly as
the checker produced them. Anything else typed into the box gets a polite "demo only"
message. Nothing typed is sent anywhere, and the page loads no outside fonts, so it
needs no privacy page. GitHub Pages can serve docs/ for free.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import checker  # noqa: E402
from server import public_result  # noqa: E402

REPO_URL = "https://github.com/Yellow-sparkle/job-ad-checker"
OUT = os.path.join(ROOT, "docs", "index.html")
DEMO_URL = "https://yellow-sparkle.github.io/job-ad-checker/"

# Link-preview tags, so LinkedIn and other sites can show a title, text and picture for the demo link.
# The picture is docs/og-image.png (1200x627). Image links must be full https addresses.
PREVIEW_TITLE = "Job Ad Checker: is this job real?"
PREVIEW_TEXT = ("Paste a job ad and see the scam warning signs in under a minute. "
                "Free demo with three made-up sample ads.")
PREVIEW_TAGS = """<meta name="description" content="{text}">
<link rel="canonical" href="{url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Job Ad Checker">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{text}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{url}og-image.png">
<meta property="og:image:secure_url" content="{url}og-image.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="627">
<meta property="og:image:alt" content="Job Ad Checker showing a Skip verdict with six scam warning signs">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{text}">
<meta name="twitter:image" content="{url}og-image.png">
""".format(title=PREVIEW_TITLE, text=PREVIEW_TEXT, url=DEMO_URL)

DEMO_NOTE = ("Demo: this page doesn't send anything anywhere. Pick one of the three sample ads to see a saved "
             "result from a real check. To check your own ads, run Job Ad Checker on your computer.")
NOT_A_SAMPLE = ("This demo only shows the three sample ads, with their suggested answers. To check your own "
                "ad, download Job Ad Checker and run it on your computer with your own API key.")


def main():
    path = os.path.join(HERE, "sample_results.json")
    if not os.path.exists(path):
        sys.exit("app/sample_results.json is missing. Run python3 app/save_samples.py first (needs your API key).")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    if saved.get("checker_prompt_version") != checker.PROMPT_VERSION:
        sys.exit("app/sample_results.json was made with older checker instructions. Run python3 app/save_samples.py again.")
    with open(os.path.join(ROOT, "samples", "sample-ads.json"), encoding="utf-8") as f:
        samples = json.load(f)["samples"]

    results = []
    for s in saved["samples"]:
        r = dict(s["result"], saved_sample=True)
        results.append({"text": s["text"], "taps": s["taps"], "result": public_result(r)})

    config = {
        "ui": checker.UI,
        "privacy_note": DEMO_NOTE,
        "samples": [{"label": s["button"], "text": s["text"], "taps": s["suggested_taps"]} for s in samples],
        "key_set": True,
    }
    data = json.dumps({"config": config, "results": results, "notSample": NOT_A_SAMPLE}, ensure_ascii=False)
    data = data.replace("</", "<\\/")  # safe inside a <script> tag

    shim = """<script>
/* Static demo: answers the page's two requests from saved results. No network calls. */
(function () {
  var DEMO = %s;
  function reply(ok, body) {
    return Promise.resolve({ ok: ok, json: function () { return Promise.resolve(body); } });
  }
  window.fetch = function (url, opts) {
    if (url === "/api/config") return reply(true, DEMO.config);
    if (url === "/api/check") {
      var req = JSON.parse((opts && opts.body) || "{}"), taps = req.taps || {};
      for (var i = 0; i < DEMO.results.length; i++) {
        var s = DEMO.results[i];
        var same = ["who_contacted", "money_or_id", "interview_stage"].every(function (k) { return taps[k] === s.taps[k]; });
        if (s.text === req.text && same) return reply(true, s.result);
      }
      return reply(false, { error: DEMO.notSample });
    }
    return reply(false, { error: "Not available in the demo." });
  };
})();
</script>
""" % data

    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        page = f.read()
    # No outside fonts in the demo (the page falls back to the system font), so no visitor data goes anywhere.
    page = "\n".join(l for l in page.split("\n") if "fonts.googleapis.com" not in l and "fonts.gstatic.com" not in l)
    links = '<a href="%s">Get the code</a>' % REPO_URL
    page = page.replace('<a href="/terms">Terms</a><a href="/privacy">Privacy</a>', links)
    page = page.replace("</title>\n", "</title>\n" + PREVIEW_TAGS, 1)
    first_script = page.index("<script")
    page = page[:first_script] + shim + page[first_script:]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print("Wrote docs/index.html with %d saved results. Open it in a browser to try it." % len(results))


if __name__ == "__main__":
    main()
