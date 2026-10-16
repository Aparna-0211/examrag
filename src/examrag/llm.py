"""Swappable LLM backends. Switch with LLM_PROVIDER=anthropic|groq|ollama."""
from __future__ import annotations

from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or set "
                "LLM_PROVIDER=ollama to run fully offline."
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OllamaLLM:
    def __init__(self, url: str, model: str):
        self._url = url.rstrip("/")
        self._model = model

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        import requests

        try:
            resp = requests.post(
                f"{self._url}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.2},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=300,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._url}. Start it with `ollama serve` "
                f"and pull the model: `ollama pull {self._model}`."
            ) from exc
        return resp.json()["message"]["content"]


class GroqLLM:
    """Groq's OpenAI-compatible chat endpoint (free tier available; limits change)."""

    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Create a key at console.groq.com, "
                "then add it to your .env file."
            )
        self._key = api_key
        self._model = model

    # Extra tokens for "thinking" models, whose hidden reasoning counts against the limit.
    REASONING_HEADROOM = 1000

    def _payload(self, system: str, user: str, max_tokens: int) -> dict:
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if "gpt-oss" in self._model:
            payload["reasoning_effort"] = "low"
            payload["max_tokens"] = max_tokens + self.REASONING_HEADROOM
        return payload

    @staticmethod
    def _error_detail(resp) -> str:
        """Groq's own error message when available, else the raw text."""
        try:
            return str(resp.json()["error"]["message"])[:300]
        except Exception:
            return (getattr(resp, "text", "") or "")[:300]

    def generate(self, system: str, user: str, max_tokens: int = 1200) -> str:
        import re

        import requests

        try:
            resp = requests.post(
                self.URL,
                headers={"Authorization": f"Bearer {self._key}"},
                json=self._payload(system, user, max_tokens),
                timeout=120,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Groq: {exc}") from exc

        if resp.status_code == 401:
            raise RuntimeError("Groq rejected the API key. Check GROQ_API_KEY in .env.")
        if resp.status_code == 429:
            raise RuntimeError(
                "Groq rate limit reached (free tier). Wait a minute and try again."
            )
        if resp.status_code in (400, 404):
            raise RuntimeError(
                f"Groq rejected the request for model '{self._model}' "
                f"(HTTP {resp.status_code}): {self._error_detail(resp)} "
                "If this is about the model, pick another from your Groq model "
                "list and set GROQ_MODEL in .env."
            )
        if not resp.ok:
            raise RuntimeError(f"Groq error {resp.status_code}: {self._error_detail(resp)}")

        content = resp.json()["choices"][0]["message"].get("content") or ""
        # Some reasoning models inline their thinking in <think> tags; drop it.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        if not content:
            raise RuntimeError(
                f"Groq model '{self._model}' returned an empty answer (it may have "
                "used its whole length limit on thinking). Try again, or set "
                "GROQ_MODEL to a different model."
            )
        return content


def get_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "groq":
        return GroqLLM(settings.groq_api_key, settings.groq_model)
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
