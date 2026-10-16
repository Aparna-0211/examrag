import pytest

from examrag.config import Settings
from examrag.llm import GroqLLM, OllamaLLM, get_llm


def test_get_llm_selects_groq():
    llm = get_llm(Settings(llm_provider="groq", groq_api_key="k"))
    assert isinstance(llm, GroqLLM)


def test_get_llm_selects_ollama():
    assert isinstance(get_llm(Settings(llm_provider="ollama")), OllamaLLM)


def test_get_llm_unknown_provider():
    with pytest.raises(ValueError):
        get_llm(Settings(llm_provider="nope"))


def test_groq_requires_key():
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqLLM("", "m")


class FakeResp:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._body = body
        self.text = text

    def json(self):
        return self._body


def _patch_post(monkeypatch, resp):
    import requests

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json)
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    return captured


def test_groq_success_sends_expected_request(monkeypatch):
    body = {"choices": [{"message": {"content": "hello"}}]}
    cap = _patch_post(monkeypatch, FakeResp(200, body))
    out = GroqLLM("secret", "model-x").generate("sys", "usr", max_tokens=50)
    assert out == "hello"
    assert cap["headers"]["Authorization"] == "Bearer secret"
    assert cap["json"]["model"] == "model-x"
    assert cap["json"]["max_tokens"] == 50
    assert [m["role"] for m in cap["json"]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    "status,fragment",
    [(401, "rejected"), (429, "rate limit"), (404, "GROQ_MODEL"), (400, "GROQ_MODEL")],
)
def test_groq_friendly_errors(monkeypatch, status, fragment):
    _patch_post(monkeypatch, FakeResp(status))
    with pytest.raises(RuntimeError, match=fragment):
        GroqLLM("k", "m").generate("s", "u")


def test_groq_bad_request_shows_groqs_message(monkeypatch):
    body = {"error": {"message": "model has been decommissioned"}}
    _patch_post(monkeypatch, FakeResp(400, body))
    with pytest.raises(RuntimeError, match="decommissioned"):
        GroqLLM("k", "m").generate("s", "u")


def test_groq_gpt_oss_gets_low_reasoning_and_headroom(monkeypatch):
    body = {"choices": [{"message": {"content": "ok"}}]}
    cap = _patch_post(monkeypatch, FakeResp(200, body))
    GroqLLM("k", "openai/gpt-oss-120b").generate("s", "u", max_tokens=300)
    assert cap["json"]["reasoning_effort"] == "low"
    assert cap["json"]["max_tokens"] == 300 + GroqLLM.REASONING_HEADROOM


def test_groq_other_models_get_no_reasoning_params(monkeypatch):
    body = {"choices": [{"message": {"content": "ok"}}]}
    cap = _patch_post(monkeypatch, FakeResp(200, body))
    GroqLLM("k", "some/other-model").generate("s", "u", max_tokens=300)
    assert "reasoning_effort" not in cap["json"]
    assert cap["json"]["max_tokens"] == 300


def test_groq_strips_think_tags(monkeypatch):
    body = {"choices": [{"message": {"content": "<think>hmm\nmore</think>\nFinal answer."}}]}
    _patch_post(monkeypatch, FakeResp(200, body))
    assert GroqLLM("k", "m").generate("s", "u") == "Final answer."


@pytest.mark.parametrize("content", ["", None, "<think>only thinking</think>"])
def test_groq_empty_answer_is_a_clear_error(monkeypatch, content):
    body = {"choices": [{"message": {"content": content}}]}
    _patch_post(monkeypatch, FakeResp(200, body))
    with pytest.raises(RuntimeError, match="empty answer"):
        GroqLLM("k", "m").generate("s", "u")


def test_groq_other_error_includes_status(monkeypatch):
    _patch_post(monkeypatch, FakeResp(500, text="boom"))
    with pytest.raises(RuntimeError, match="500"):
        GroqLLM("k", "m").generate("s", "u")
