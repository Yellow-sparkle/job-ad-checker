"""Tests for run_record.py and compare_runs.py. No network, no API key needed.

Run from the eval folder:  python -m pytest tests -q
"""
import csv, json, os, sys, types
import pytest

EVAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EVAL)
import run_record  # noqa: E402
from run_record import start_run, golden_version, prompt_hash, parse_args  # noqa: E402

AD_SENTENCE = "Please return the signed offer with your bank details"


@pytest.fixture
def files(tmp_path):
    gold = tmp_path / "golden-set-v0.6.jsonl"
    cases = [{"case_id": "GS-R06", "input_text": AD_SENTENCE + " and a refundable fee.",
              "acceptable_verdicts": "Skip", "expected_verdict": "Skip", "must_fire": "",
              "must_not_fire": "", "ground_truth": "Scam", "expected_extraction": ""},
             {"case_id": "GS-H01", "input_text": "A normal ad.", "acceptable_verdicts": "No red flags found",
              "expected_verdict": "No red flags found", "must_fire": "", "must_not_fire": "",
              "ground_truth": "Real", "expected_extraction": ""}]
    gold.write_text("".join(json.dumps(c) + "\n" for c in cases))
    out = tmp_path / "tool_outputs.jsonl"
    outs = [{"case_id": "GS-R06", "verdict": "Skip", "result_text": AD_SENTENCE, "fired": [],
             "reasons": [], "checker_version": "1.2", "checker_prompt_version": "p3",
             "usage": {"input_tokens": 1000, "output_tokens": 200, "latency_s": 2.0, "cost_usd": 0.002}},
            {"case_id": "GS-H01", "verdict": "No red flags found", "result_text": "ok", "fired": [],
             "reasons": [], "checker_version": "1.2",
             "usage": {"input_tokens": 500, "output_tokens": 100, "latency_s": 4.0, "cost_usd": None}}]
    out.write_text("".join(json.dumps(o) + "\n" for o in outs))
    return gold, out, outs, tmp_path


def fake_module(prompt="Judge prompt v1", version="v1"):
    m = types.ModuleType("fake_judge")
    m.SYSTEM_PROMPT = prompt
    m.USER_TEMPLATE = "INPUT: {input}"
    m.PROMPT_VERSION = version
    return m


def info_of(run):
    return json.load(open(run.path("run_info.json")))


def test_folder_and_required_fields(files):
    gold, out, _, tmp = files
    with start_run("faithfulness_check", str(gold), [str(out)], judges={"faithfulness": fake_module()},
                   model="anthropic/x", temperature=0.0, thresholds={"relevance_min": 0.7},
                   runs_dir=str(tmp / "runs")) as run:
        assert info_of(run)["status"] == "running"
    info = info_of(run)
    for key in ["run_id", "script", "status", "started_at", "finished_at", "golden_set", "outputs",
                "judges", "thresholds", "checker", "warnings", "environment", "usage_summary"]:
        assert key in info
    assert info["status"] == "completed"
    assert info["golden_set"]["version"] == "v0.6" and info["golden_set"]["cases"] == 2
    assert len(info["golden_set"]["sha256"]) == 64
    assert info["outputs"][0]["lines"] == 2
    j = info["judges"]["faithfulness"]
    assert {k: j[k] for k in ("model", "temperature", "prompt_version", "prompt_hash")} == {
        "model": "anthropic/x", "temperature": 0.0, "prompt_version": "v1",
        "prompt_hash": prompt_hash(fake_module())}
    assert "temperature_applied" in j
    assert os.path.dirname(run.folder) == str(tmp / "runs")
    assert os.path.isfile(run.path("usage.csv"))


def test_golden_version():
    assert golden_version("x/golden-set-v0.6.jsonl") == "v0.6"
    assert golden_version("golden-set-v0.13.jsonl") == "v0.13"
    assert golden_version("cases.jsonl") is None


def test_prompt_hash_and_unchanged_version_warning(files):
    gold, out, _, tmp = files
    runs = str(tmp / "runs")
    assert prompt_hash(fake_module("a")) != prompt_hash(fake_module("b"))
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module("a", "v1")},
                   temperature=0, runs_dir=runs) as r1:
        pass
    assert info_of(r1)["warnings"] == []
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module("b", "v1")},
                   temperature=0, runs_dir=runs, run_dir=str(tmp / "runs" / "second")) as r2:
        pass
    assert any("PROMPT_VERSION" in w for w in info_of(r2)["warnings"])
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module("c", "v2")},
                   temperature=0, runs_dir=runs, run_dir=str(tmp / "runs" / "third")) as r3:
        pass
    assert info_of(r3)["warnings"] == []


def test_temperature_not_applied_warning(files):
    pytest.importorskip("opik")
    gold, out, _, tmp = files
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module()},
                   model="anthropic/claude-sonnet-5-5", temperature=0.0,
                   runs_dir=str(tmp / "runs")) as run:
        pass
    info = info_of(run)
    assert info["judges"]["faith"]["temperature_applied"] is False
    assert any("not applied" in w for w in info["warnings"])


def test_temperature_warning(files):
    gold, out, _, tmp = files
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module()},
                   temperature=0.3, runs_dir=str(tmp / "runs")) as run:
        pass
    assert any("temperature is 0.3" in w for w in info_of(run)["warnings"])


class FakeModel:
    def __init__(self, tokens):
        self.tokens = tokens

    def _litellm_completion(self, **kw):
        return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=self.tokens[0],
                                                                 completion_tokens=self.tokens[1]))


class FakeMetric:
    def __init__(self, tokens=(300, 40)):
        self._model = FakeModel(tokens)

    def score(self, **kw):
        self._model._litellm_completion(messages=[])
        return types.SimpleNamespace(value=0.0, reason="fine")


def test_timed_captures_tokens_and_summary_maths(files, monkeypatch):
    gold, out, outs, tmp = files
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion_cost = lambda completion_response: 0.001
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module()}, temperature=0,
                   runs_dir=str(tmp / "runs")) as run:
        m = FakeMetric()
        orig = m._model._litellm_completion
        for o in outs:
            run.add_checker_usage(o["case_id"], o)
            assert run.timed(o["case_id"], "faithfulness", m.score, input="x").value == 0.0
        assert m._model._litellm_completion == orig  # restored after each call
    s = info_of(run)["usage_summary"]
    assert s["judge"]["calls"] == 2 and s["judge"]["input_tokens"] == 600
    assert s["judge"]["output_tokens"] == 80 and s["judge"]["cost_usd"] == pytest.approx(0.002)
    assert s["checker"]["calls"] == 2 and s["checker"]["input_tokens"] == 1500
    assert s["checker"]["cost_usd"] == pytest.approx(0.002) and s["checker"]["calls_without_cost"] == 1
    assert s["checker"]["seconds_total"] == 6.0 and s["checker"]["seconds_mean"] == 3.0
    assert info_of(run)["checker"]["versions"] == {"1.2": 2}
    assert info_of(run)["checker"]["prompt_versions"] == {"p3": 1}
    rows = list(csv.DictReader(open(run.path("usage.csv"))))
    assert [r["source"] for r in rows].count("judge") == 2


def test_seconds_only_fallback(files):
    gold, out, _, tmp = files
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module()}, temperature=0,
                   runs_dir=str(tmp / "runs")) as run:
        run.timed("GS-R06", "faithfulness", lambda **kw: 1, input="x")
    info = info_of(run)
    assert info["usage_summary"]["judge"]["input_tokens"] is None
    assert info["usage_summary"]["judge"]["seconds_total"] is not None
    assert any("seconds only" in w for w in info["warnings"])


def test_no_ad_text_or_api_key_written(files, monkeypatch):
    gold, out, outs, tmp = files
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-very-secret-123")
    with start_run("fc", str(gold), [str(out)], judges={"faith": fake_module()}, temperature=0,
                   runs_dir=str(tmp / "runs")) as run:
        for o in outs:
            run.add_checker_usage(o["case_id"], o)
        run.timed("GS-R06", "faithfulness", FakeMetric().score, input=AD_SENTENCE)
    for name in ("run_info.json", "usage.csv"):
        text = open(run.path(name)).read()
        assert AD_SENTENCE not in text
        assert "sk-test-very-secret-123" not in text


def test_failed_status_and_missing_file(files):
    gold, out, _, tmp = files
    with pytest.raises(KeyError):
        with start_run("fc", str(gold), [str(out)], runs_dir=str(tmp / "runs")) as run:
            raise KeyError("GS-X99")
    assert info_of(run)["status"] == "failed" and "KeyError" in info_of(run)["error"]
    with pytest.raises(SystemExit):
        start_run("fc", str(gold), [str(tmp / "nope.jsonl")], runs_dir=str(tmp / "runs"),
                  run_dir=str(tmp / "runs" / "missing"))
    info = json.load(open(tmp / "runs" / "missing" / "run_info.json"))
    assert info["status"] == "failed" and "nope.jsonl" in info["error"]


def test_dry_run_blocks_judge(files):
    gold, out, _, tmp = files
    with start_run("fc", str(gold), [str(out)], runs_dir=str(tmp / "runs"), dry_run=True) as run:
        with pytest.raises(RuntimeError):
            run.timed("GS-R06", "faithfulness", lambda: 1)
    assert info_of(run)["dry_run"] is True and run.folder.endswith("-dry")


def test_parse_args():
    pos, o = parse_args(["g.jsonl", "o.jsonl", "--run-dir", "runs/x", "--dry-run", "--geval"])
    assert pos == ["g.jsonl", "o.jsonl"] and o["run_dir"] == "runs/x" and o["dry_run"]
    assert o["flags"] == {"--geval"}


def test_faithfulness_check_dry_run_calls_no_judge(files, monkeypatch):
    pytest.importorskip("opik")
    gold, out, _, tmp = files
    monkeypatch.chdir(EVAL)
    import faithfulness_check as fc

    def boom(**kw):
        raise AssertionError("judge called in dry run")
    for m in (fc.metric, fc.relevance, fc.precision):
        monkeypatch.setattr(m, "score", boom)
    fc.main(str(gold), str(out), run_dir=str(tmp / "runs" / "fc-dry"), dry_run=True)
    info = json.load(open(tmp / "runs" / "fc-dry" / "run_info.json"))
    assert info["status"] == "completed" and info["dry_run"]
    assert set(info["judges"]) == {"faithfulness", "relevance", "extraction"}
    assert info["judges"]["faithfulness"]["prompt_version"] == "v0.4"


def test_extra_checks_writes_into_run_folder(files):
    gold, out, _, tmp = files
    import extra_checks
    extra_checks.main([str(gold), str(out), "--run-dir", str(tmp / "runs" / "ec")])
    assert os.path.isfile(tmp / "runs" / "ec" / "extra_check_results.json")
    info = json.load(open(tmp / "runs" / "ec" / "run_info.json"))
    assert info["script"] == "extra_checks" and info["judges"] == {}


def test_stability_check_keeps_exit_code(files):
    gold, out, _, tmp = files
    import stability_check
    code = stability_check.main([str(gold), str(out), str(out), "--run-dir", str(tmp / "runs" / "st")])
    assert code == 0
    assert os.path.isfile(tmp / "runs" / "st" / "stability_report.csv")


def test_compare_runs(files, capsys):
    gold, out, _, tmp = files
    import extra_checks, compare_runs
    extra_checks.main([str(gold), str(out), "--run-dir", str(tmp / "runs" / "a")])
    changed = tmp / "tool_outputs_b.jsonl"
    rows = [json.loads(l) for l in open(out)]
    rows[0]["verdict"] = "Be careful"
    rows[0]["checker_version"] = "1.3"
    changed.write_text("".join(json.dumps(r) + "\n" for r in rows))
    extra_checks.main([str(gold), str(changed), "--run-dir", str(tmp / "runs" / "b")])
    capsys.readouterr()
    compare_runs.main([str(tmp / "runs" / "a"), str(tmp / "runs" / "b")])
    text = capsys.readouterr().out
    assert "checker.versions.1.3" in text
    assert "GS-R06\tverdict_ok\tnow fails" in text


class FakeMessages:
    def create(self, **kw):
        return types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=900, output_tokens=50))


class FakeAnthropicMetric:
    def __init__(self):
        self._model = types.SimpleNamespace(_client=types.SimpleNamespace(messages=FakeMessages()),
                                            model_name="claude-test")

    def score(self, **kw):
        self._model._client.messages.create(messages=[])
        return types.SimpleNamespace(value=1.0, reason="r")


def test_timed_anthropic_sdk_path(files, monkeypatch):
    gold, out, _, tmp = files
    fake_litellm = types.ModuleType("litellm")

    def no_cost(completion_response):
        raise ValueError("unknown")
    fake_litellm.completion_cost = no_cost
    fake_litellm.cost_per_token = lambda model, prompt_tokens, completion_tokens: (0.0027, 0.00075)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    m = FakeAnthropicMetric()
    with start_run("fc", str(gold), [str(out)], runs_dir=str(tmp / "runs")) as run:
        run.timed("GS-R06", "relevance", m.score, output="x")
    assert "create" not in vars(m._model._client.messages)  # patch removed
    s = info_of(run)["usage_summary"]["judge"]
    assert (s["input_tokens"], s["output_tokens"]) == (900, 50)
    assert s["cost_usd"] == pytest.approx(0.00345)
