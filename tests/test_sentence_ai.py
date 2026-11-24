"""Tests for the AI sentence writer (no network: provider calls are faked)."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from sign_bridge.language import sentence_ai as sa
from sign_bridge.language.sentence_ai import (
    SentenceAI,
    find_api_key,
    parse_reply,
    plain_sentence,
    save_api_key,
)

SIGNS = ["Mom", "Dad", "No", "Drink"]


def test_plain_sentence() -> None:
    assert plain_sentence(SIGNS) == "Mom dad no drink."
    assert plain_sentence([]) == ""


def test_parse_reply_variants() -> None:
    assert parse_reply('{"sentences": ["A.", "B."]}') == ["A.", "B."]
    assert parse_reply('```json\n{"sentences": ["Mom and Dad don\'t drink."]}\n```') == [
        "Mom and Dad don't drink."]
    assert parse_reply("1. First.\n2. Second.\n3. Third.\n4. Fourth.", 3) == [
        "First.", "Second.", "Third."]


def test_no_key_falls_back() -> None:
    r = SentenceAI(provider="gemini", api_key=None).compose(SIGNS)
    assert r.source == "plain" and r.best == "Mom dad no drink." and r.error


class _Recorder:
    """Stands in for the HTTP POST; returns a canned provider reply."""

    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []

    def __call__(self, url: str, headers: dict[str, str], body: dict[str, Any],
                 timeout: float) -> Any:
        self.calls.append((url, headers, body))
        return self.reply


ANSWER = json.dumps({"sentences": ["Mom and Dad don't drink.", "Mom and Dad, no drinking!"]})
REPLIES = {
    "gemini": {"candidates": [{"content": {"parts": [{"text": ANSWER}]}}]},
    "anthropic": {"content": [{"type": "text", "text": ANSWER}]},
    "openai": {"choices": [{"message": {"content": ANSWER}}]},
}


@pytest.mark.parametrize("provider", ["gemini", "anthropic", "openai"])
def test_each_provider(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = _Recorder(REPLIES[provider])
    monkeypatch.setattr(sa, "_post_json", rec)
    ai = SentenceAI(provider=provider, api_key="k-123")
    r = ai.compose(SIGNS)
    assert r.source == provider and r.error is None
    assert r.sentences == ("Mom and Dad don't drink.", "Mom and Dad, no drinking!")
    url, headers, body = rec.calls[0]
    assert "k-123" in json.dumps(headers)               # key goes in a header, never the URL
    assert "k-123" not in url
    assert "MOM" in json.dumps(body) and "DRINK" in json.dumps(body)
    ai.compose(SIGNS)
    assert len(rec.calls) == 1                          # cached


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b""))  # type: ignore[arg-type]


@pytest.mark.parametrize(("exc", "needle"), [
    (_http_error(401), "API key"),
    (_http_error(404), "ai.model"),
    (_http_error(429), "quota"),
    (urllib.error.URLError("offline"), "no connection"),
])
def test_errors_fall_back(exc: Exception, needle: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> Any:
        raise exc

    monkeypatch.setattr(sa, "_post_json", boom)
    monkeypatch.setattr(sa.time, "sleep", lambda _: None)
    r = SentenceAI(provider="gemini", api_key="k", retries=1).compose(SIGNS)
    assert r.source == "plain" and r.best == "Mom dad no drink."
    assert needle in str(r.error)


def test_key_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    secrets = tmp_path / ".streamlit" / "secrets.toml"
    assert find_api_key("gemini", secrets) is None
    save_api_key("gemini", "  abc  ", secrets)
    save_api_key("anthropic", "xyz", secrets)
    assert find_api_key("gemini", secrets) == "abc"
    assert find_api_key("anthropic", secrets) == "xyz"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert find_api_key("gemini", secrets) == "from-env"


def test_prompt_forbids_adding_meaning() -> None:
    text = sa.SYSTEM_PROMPT.format(n=3)
    assert "Never add new people" in text and '{"sentences"' in text


def test_busy_service_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 (provider busy) is retried; the second attempt succeeds."""
    calls = {"n": 0}

    def flaky(*_: Any, **__: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(503)
        return REPLIES["gemini"]

    monkeypatch.setattr(sa, "_post_json", flaky)
    monkeypatch.setattr(sa.time, "sleep", lambda _: None)
    r = SentenceAI(provider="gemini", api_key="k").compose(SIGNS)
    assert r.source == "gemini" and calls["n"] == 2
    assert r.best == "Mom and Dad don't drink."


def test_bad_key_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def always_401(*_: Any, **__: Any) -> Any:
        calls["n"] += 1
        raise _http_error(401)

    monkeypatch.setattr(sa, "_post_json", always_401)
    monkeypatch.setattr(sa.time, "sleep", lambda _: None)
    r = SentenceAI(provider="gemini", api_key="bad").compose(SIGNS)
    assert calls["n"] == 1 and "API key" in str(r.error)
