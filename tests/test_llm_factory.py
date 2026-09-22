import pytest

from raptor_qdrant.rag.llm import BaseChatbotModel, create_chatbot
from raptor_qdrant.rag.llm.ollama import Ollama


@pytest.mark.parametrize("name", ["ollama", "OLLAMA", "  Ollama  "])
def test_resolves_ollama_regardless_of_casing(name):
    llm = create_chatbot(name)

    assert isinstance(llm, Ollama)
    assert isinstance(llm, BaseChatbotModel)


def test_describe_names_the_model():
    assert create_chatbot("ollama").describe.startswith("Ollama(")


def test_unknown_provider_is_rejected_by_name():
    with pytest.raises(ValueError, match="gpt4all"):
        create_chatbot("gpt4all")


def test_error_lists_the_supported_providers():
    with pytest.raises(ValueError, match="ollama, bedrock"):
        create_chatbot("nope")


def test_falls_back_to_the_configured_provider(monkeypatch):
    from raptor_qdrant.core.config import settings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")

    assert isinstance(create_chatbot(None), Ollama)


def test_model_tag_is_normalised_for_health_check():
    assert (
        Ollama(model="qwen2.5")._fully_qualified_model_tag()
        == "qwen2.5:latest"
    )
    assert (
        Ollama(model="qwen2.5:7b")._fully_qualified_model_tag() == "qwen2.5:7b"
    )


def test_ollama_accepts_a_model_override():
    from raptor_qdrant.core.config import settings

    llm = create_chatbot("ollama", model="qwen2.5:3b")

    assert isinstance(llm, Ollama)
    assert llm.model == "qwen2.5:3b"
    assert create_chatbot("ollama").model == settings.OLLAMA_MODEL


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("claude", "anthropic"),
        ("CHATGPT", "openai"),
        ("  gpt  ", "openai"),
        ("ollama", "ollama"),
    ],
)
def test_aliases_resolve_to_canonical_providers(name, expected):
    from raptor_qdrant.rag.llm import normalize_provider

    assert normalize_provider(name) == expected


@pytest.mark.parametrize(
    "name", ["bedrock", "anthropic", "claude", "openai", "chatgpt"]
)
def test_providers_that_leave_the_machine_are_remote(name):
    from raptor_qdrant.rag.llm import is_remote_provider

    assert is_remote_provider(name)


def test_ollama_stays_local():
    from raptor_qdrant.rag.llm import is_remote_provider

    assert not is_remote_provider("ollama")


def test_error_names_the_new_providers():
    with pytest.raises(ValueError, match=r"anthropic \(claude\)"):
        create_chatbot("nope")


def test_configured_model_follows_the_provider(monkeypatch):
    from raptor_qdrant.core.config import settings
    from raptor_qdrant.rag.llm import configured_model

    monkeypatch.setattr(settings, "ANTHROPIC_MODEL", "claude-x")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-x")

    assert configured_model("claude") == "claude-x"
    assert configured_model("chatgpt") == "gpt-x"


def test_claude_is_built_for_the_anthropic_aliases(monkeypatch):
    from raptor_qdrant.core.config import settings
    from raptor_qdrant.rag.llm.claude import Claude

    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "test-key")

    llm = create_chatbot("claude", model="claude-haiku-4-5")

    assert isinstance(llm, Claude)
    assert llm.describe == "Claude(claude-haiku-4-5)"


def test_chatgpt_is_built_for_the_openai_aliases(monkeypatch):
    from raptor_qdrant.core.config import settings
    from raptor_qdrant.rag.llm.chatgpt import ChatGPT

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    llm = create_chatbot("chatgpt", model="gpt-4.1-mini")

    assert isinstance(llm, ChatGPT)
    assert llm.describe == "ChatGPT(gpt-4.1-mini)"


@pytest.mark.parametrize(
    ("model", "sends_temperature"),
    [("gpt-4.1-mini", True), ("gpt-5", False), ("o3-mini", False)],
)
def test_reasoning_models_get_no_temperature(
    monkeypatch, model, sends_temperature
):
    from raptor_qdrant.core.config import settings
    from raptor_qdrant.rag.llm.chatgpt import ChatGPT

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "test-key")

    assert ChatGPT(model=model).supports_temperature() is sends_temperature


class Recorded(BaseChatbotModel):
    def complete(self, prompt: str) -> str:
        return "답"


def test_lazy_chatbot_waits_for_the_first_call(monkeypatch):
    from raptor_qdrant.rag import llm as llm_module

    built: list[tuple[str | None, str | None]] = []

    def fake_create(provider=None, model=None):
        built.append((provider, model))
        return Recorded()

    monkeypatch.setattr(llm_module, "create_chatbot", fake_create)
    lazy = llm_module.LazyChatbot("ollama", "qwen2.5:3b")

    assert built == []
    assert "not started" in lazy.describe

    lazy.complete("프롬프트")
    lazy.complete("프롬프트")

    assert built == [("ollama", "qwen2.5:3b")]
