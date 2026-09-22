"""네트워크 없이 Claude·ChatGPT 가 실제로 보내는 요청 모양을 본다."""

from types import SimpleNamespace
from typing import Any

import anthropic
import openai
import pytest

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.llm.chatgpt import ChatGPT
from raptor_qdrant.rag.llm.claude import Claude


class Recorder:
    def __init__(self, reply: Any) -> None:
        self.sent: dict[str, Any] = {}
        self.reply = reply

    def create(self, **kwargs: Any) -> Any:
        self.sent = kwargs
        return self.reply


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def claude_with(reply: Any, monkeypatch, **overrides: Any) -> tuple:
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-key")
    llm = Claude(**overrides)
    recorder = Recorder(reply)
    llm.client = SimpleNamespace(  # type: ignore[assignment]
        beta=SimpleNamespace(messages=recorder)
    )
    return llm, recorder


def chatgpt_with(reply: Any, monkeypatch, **overrides: Any) -> tuple:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")
    llm = ChatGPT(**overrides)
    recorder = Recorder(reply)
    llm.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=recorder)
    )
    return llm, recorder


def answered(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="end_turn", stop_details=None, content=[text_block(text)]
    )


def completed(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text),
            )
        ]
    )


def test_claude_omits_sampling_for_models_that_reject_it(monkeypatch):
    llm, recorder = claude_with(
        answered("답"), monkeypatch, model="claude-opus-5"
    )

    llm.answer("맥락", "질문")

    assert recorder.sent["extra_body"] == {}


def test_claude_sends_the_configured_effort(monkeypatch):
    llm, recorder = claude_with(answered("답"), monkeypatch, effort="low")

    llm.answer("맥락", "질문")

    assert recorder.sent["output_config"] == {"effort": "low"}


def test_claude_leaves_effort_out_when_it_is_blank(monkeypatch):
    llm, recorder = claude_with(answered("답"), monkeypatch, effort="")

    llm.answer("맥락", "질문")

    assert recorder.sent["output_config"] is anthropic.omit


def test_claude_asks_for_a_fallback_on_refusal(monkeypatch):
    llm, recorder = claude_with(answered("답"), monkeypatch)

    llm.answer("맥락", "질문")

    assert recorder.sent["fallbacks"] == "default"
    assert recorder.sent["betas"]


def test_claude_joins_the_text_blocks(monkeypatch):
    reply = SimpleNamespace(
        stop_reason="end_turn",
        stop_details=None,
        content=[
            SimpleNamespace(type="thinking", thinking="속으로 생각"),
            text_block("앞부분 "),
            text_block("뒷부분  "),
        ],
    )
    llm, _ = claude_with(reply, monkeypatch)

    assert llm.answer("맥락", "질문") == "앞부분 뒷부분"


def test_claude_refusal_becomes_an_error(monkeypatch):
    reply = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber"),
        content=[],
    )
    llm, _ = claude_with(reply, monkeypatch)

    with pytest.raises(RuntimeError, match="cyber"):
        llm.answer("맥락", "질문")


def test_claude_health_check_uses_a_tiny_budget(monkeypatch):
    llm, recorder = claude_with(answered("pong"), monkeypatch)

    llm.health_check()

    assert recorder.sent["max_tokens"] == 16


def test_chatgpt_sends_temperature_to_a_plain_model(monkeypatch):
    llm, recorder = chatgpt_with(
        completed("답"), monkeypatch, model="gpt-4.1-mini"
    )

    llm.answer("맥락", "질문")

    assert recorder.sent["temperature"] == llm.temperature
    assert recorder.sent["max_completion_tokens"] == llm.max_tokens


def test_chatgpt_leaves_temperature_out_for_reasoning_models(monkeypatch):
    llm, recorder = chatgpt_with(completed("답"), monkeypatch, model="gpt-5")

    llm.answer("맥락", "질문")

    assert recorder.sent["temperature"] is openai.NOT_GIVEN


def test_chatgpt_survives_an_empty_message(monkeypatch):
    llm, _ = chatgpt_with(completed(None), monkeypatch)

    assert llm.answer("맥락", "질문") == ""


def test_oauth_token_is_read_from_its_file(monkeypatch, tmp_path):
    token_file = tmp_path / "oauth-token"
    token_file.write_text("sk-ant-oat-가짜토큰\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "")
    monkeypatch.setattr(
        settings, "ANTHROPIC_OAUTH_TOKEN_FILE", str(token_file)
    )

    llm = Claude()

    assert llm.uses_oauth
    assert llm.client.auth_token == "sk-ant-oat-가짜토큰"
    assert llm.client.api_key is None


def test_oauth_requests_carry_the_bearer_beta_flag(monkeypatch, tmp_path):
    from raptor_qdrant.rag.constants import ANTHROPIC_OAUTH_BETA

    token_file = tmp_path / "oauth-token"
    token_file.write_text("sk-ant-oat-가짜토큰", encoding="utf-8")
    monkeypatch.setattr(
        settings, "ANTHROPIC_OAUTH_TOKEN_FILE", str(token_file)
    )
    llm, recorder = claude_with(answered("답"), monkeypatch)

    llm.answer("맥락", "질문")

    assert ANTHROPIC_OAUTH_BETA in recorder.sent["betas"]


def test_api_key_requests_do_not_carry_the_bearer_flag(monkeypatch):
    from raptor_qdrant.rag.constants import ANTHROPIC_OAUTH_BETA

    monkeypatch.setattr(settings, "ANTHROPIC_OAUTH_TOKEN_FILE", "")
    llm, recorder = claude_with(answered("답"), monkeypatch)

    llm.answer("맥락", "질문")

    assert ANTHROPIC_OAUTH_BETA not in recorder.sent["betas"]


def test_a_missing_token_file_says_which_path(monkeypatch, tmp_path):
    monkeypatch.setattr(
        settings, "ANTHROPIC_OAUTH_TOKEN_FILE", str(tmp_path / "없는파일")
    )

    with pytest.raises(ValueError, match="없는파일"):
        Claude()


def test_an_empty_token_file_is_rejected(monkeypatch, tmp_path):
    token_file = tmp_path / "oauth-token"
    token_file.write_text("   \n", encoding="utf-8")
    monkeypatch.setattr(
        settings, "ANTHROPIC_OAUTH_TOKEN_FILE", str(token_file)
    )

    with pytest.raises(ValueError, match="empty"):
        Claude()


@pytest.mark.parametrize(
    ("model", "sends"),
    [
        ("claude-haiku-4-5", True),
        ("claude-opus-4-6", True),
        ("claude-opus-5", False),
        ("claude-sonnet-5", False),
        ("claude-fable-5-1", False),
    ],
)
def test_temperature_goes_only_to_models_that_take_it(
    monkeypatch, model, sends
):
    llm, recorder = claude_with(answered("답"), monkeypatch, model=model)

    llm.answer("맥락", "질문")

    if sends:
        assert recorder.sent["extra_body"] == {"temperature": llm.temperature}
    else:
        assert recorder.sent["extra_body"] == {}
