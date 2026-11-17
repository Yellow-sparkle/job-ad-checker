"""Job Ad Checker: local web app.

Run:   python3 app/server.py      then open http://localhost:8000
Nothing is saved: no database, no log of what you paste, no cookies, no IDs.
Only Python's standard library plus the `anthropic` package.
"""
import copy
import html
import json
import os
import sys
import threading
import time
from collections import defaultdict, deque
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checker  # noqa: E402
from textmap import mask_phones  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = int(os.environ.get("PORT", "8000"))
CHECKS_PER_HOUR = 20
# Spending caps (cost plan, 14 Nov): about 200 checks a day and a dollar cap per run of the app.
# The real limit is the one Kate sets in the Anthropic Console; these stop the app before it.
# When a cap is hit or the AI is unavailable, the app gives the rules-only result.
DAILY_AI_CHECKS = int(os.environ.get("DAILY_AI_CHECKS", "200"))
SPEND_CAP_USD = float(os.environ.get("SPEND_CAP_USD", "10"))
SAVED_SAMPLES = os.path.join(HERE, "sample_results.json")
_budget = {"day": time.strftime("%Y-%m-%d"), "checks": 0, "spent": 0.0}

_hits = defaultdict(deque)  # IP -> times of recent checks, in memory only
_lock = threading.Lock()


def rate_limited(ip):
    now = time.time()
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= CHECKS_PER_HOUR:
            return True
        q.append(now)
        return False


def ai_allowed():
    with _lock:
        today = time.strftime("%Y-%m-%d")
        if _budget["day"] != today:
            _budget.update(day=today, checks=0)
        return _budget["checks"] < DAILY_AI_CHECKS and _budget["spent"] < SPEND_CAP_USD


def record_spend(result):
    with _lock:
        _budget["checks"] += 1
        _budget["spent"] += ((result.get("usage") or {}).get("cost_usd") or 0)


def saved_sample(text, taps):
    """A saved result for a made-up sample ad, if the person kept the suggested answers.
    Saves AI calls during demos and works without internet. Made by app/save_samples.py."""
    try:
        with open(SAVED_SAMPLES, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    if saved.get("checker_prompt_version") != checker.PROMPT_VERSION:
        return None  # made with older instructions
    for s in saved.get("samples", []):
        same_taps = all(taps.get(k) == s["taps"].get(k) for k in ("who_contacted", "money_or_id", "interview_stage"))
        if s["text"] == text and same_taps:
            r = copy.deepcopy(s["result"])
            r["saved_sample"] = True
            return r
    return None


def run_check(text, taps):
    r = saved_sample(text, taps)
    if r:
        return r
    if not ai_allowed():
        return checker.check(text, taps, rules_only=True)
    try:
        r = checker.check(text, taps)
    except checker.CheckerError as e:
        if not e.fallback:
            raise
        print(f"AI unavailable ({e}); gave the rules-only result.", file=sys.stderr)
        return checker.check(text, taps, rules_only=True)
    record_spend(r)
    return r


def public_result(r):
    """What the page needs. Rule ids, raw AI output and costs stay on the server."""
    r = copy.deepcopy(r)
    for x in r["reasons"]:
        x["quote"] = mask_phones(x["quote"])
        x["title"] = mask_phones(x["title"])
        x.pop("rule_id", None)
    for w in r["weak_notes"]:
        w["quote"] = mask_phones(w["quote"])
        w.pop("rule_id", None)
    for n in r["notes"]:
        n["quote"] = mask_phones(n["quote"])
        n["text"] = mask_phones(n["text"])
    for s in r["signals"]:
        s.pop("rule_id", None)
    for k in ("fired", "raw_llm_output", "usage", "dropped_findings", "extraction"):
        r.pop(k, None)
    return r


def legal_page(name):
    path = os.path.join(ROOT, "legal", f"{name}.md")
    with open(path, encoding="utf-8") as f:
        md = f.read()
    body = []
    for line in md.splitlines():
        t = html.escape(line)
        if line.startswith("# "):
            body.append(f"<h1>{t[2:]}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{t[3:]}</h2>")
        elif line.startswith("|"):
            body.append(f"<p class='row'>{t}</p>")
        elif line.startswith("- "):
            body.append(f"<p class='li'>• {t[2:]}</p>")
        elif line.strip():
            body.append(f"<p>{t}</p>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Job Ad Checker: {name.title()}</title><style>"
            "body{font-family:system-ui,sans-serif;max-width:760px;margin:0 auto;padding:24px 16px;"
            "color:#14212B;background:#F4F9FC;line-height:1.55}a{color:#007BA7}.row{font-size:13px;"
            "color:#3C4852;margin:2px 0}.li{margin:4px 0}</style></head><body>"
            "<p><a href='/'>Back to Job Ad Checker</a></p>" + "\n".join(body) + "</body></html>")


class Handler(BaseHTTPRequestHandler):
    server_version = "JobAdChecker"

    def log_message(self, fmt, *args):  # no request log: nothing about users is kept
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/config":
            with open(os.path.join(ROOT, "samples", "sample-ads.json"), encoding="utf-8") as f:
                samples = json.load(f)["samples"]
            return self._send(200, json.dumps({
                "ui": checker.UI,
                "privacy_note": checker.PRIVACY_NOTE,
                "samples": [{"label": s["button"], "text": s["text"], "taps": s["suggested_taps"]} for s in samples],
                "key_set": bool(os.environ.get("ANTHROPIC_API_KEY")),
            }))
        if path in ("/terms", "/privacy"):
            return self._send(200, legal_page(path[1:]), "text/html; charset=utf-8")
        self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if self.path != "/api/check":
            return self._send(404, json.dumps({"error": "Not found"}))
        if rate_limited(self.client_address[0]):
            return self._send(429, json.dumps({"error": f"That's {CHECKS_PER_HOUR} checks this hour. Please try again later."}))
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(min(n, 200_000)) or b"{}")
            result = run_check(req.get("text", ""), req.get("taps") or {})
            self._send(200, json.dumps(public_result(result)))
        except checker.CheckerError as e:
            self._send(502, json.dumps({"error": str(e)}))
        except Exception as e:  # never echo the pasted text
            print(f"Check failed: {type(e).__name__}", file=sys.stderr)
            self._send(500, json.dumps({"error": "Something went wrong with this check. Please try again."}))


def main():
    if "--model" in sys.argv:  # python3 app/server.py --model claude-haiku-5-5
        m = sys.argv[sys.argv.index("--model") + 1]
        if m not in checker.PRICES:
            sys.exit(f"Unknown model {m!r}. Use one of: {', '.join(checker.PRICES)}")
        os.environ["CHECKER_MODEL"] = m
    print(f"Checker model: {os.environ.get('CHECKER_MODEL', checker.DEFAULT_MODEL)}. "
          f"Caps: {DAILY_AI_CHECKS} AI checks a day, ${SPEND_CAP_USD:.2f} while this window runs; "
          f"after that, rules-only results.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Note: ANTHROPIC_API_KEY isn't set in this Terminal window, so checks will fail.\n"
              "Run:  export ANTHROPIC_API_KEY=<your job-ad-checker key>   then start the app again.\n")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Job Ad Checker is running. Open http://localhost:{PORT} in your browser.")
    print("Press Ctrl+C here to stop it.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
