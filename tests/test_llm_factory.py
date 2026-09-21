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
