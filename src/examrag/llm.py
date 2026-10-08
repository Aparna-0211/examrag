"""Swappable LLM backends. Switch with LLM_PROVIDER=anthropic|ollama."""
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


def get_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.anthropic_api_key, settings.anthropic_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
