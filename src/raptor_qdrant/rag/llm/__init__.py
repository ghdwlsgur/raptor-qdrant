from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.llm.base import BaseChatbotModel

__all__ = [
    "BaseChatbotModel",
    "LazyChatbot",
    "accepted_providers",
    "configured_model",
    "create_chatbot",
    "is_remote_provider",
    "normalize_provider",
    "supported_providers",
]

CANONICAL_PROVIDERS = ("ollama", "bedrock", "anthropic", "openai")
PROVIDER_ALIASES = {
    "claude": "anthropic",
    "chatgpt": "openai",
    "gpt": "openai",
}
# 볼트 본문이 밖으로 나가는 공급자. 인덱싱 전에 동의를 받는다
REMOTE_PROVIDERS = frozenset({"bedrock", "anthropic", "openai"})
MODEL_SETTING = {
    "ollama": "OLLAMA_MODEL",
    "bedrock": "BEDROCK_MODEL_ID",
    "anthropic": "ANTHROPIC_MODEL",
    "openai": "OPENAI_MODEL",
}


def normalize_provider(provider: str | None) -> str:
    name = (provider or settings.LLM_PROVIDER).strip().lower()
    return PROVIDER_ALIASES.get(name, name)


def is_remote_provider(provider: str | None) -> bool:
    return normalize_provider(provider) in REMOTE_PROVIDERS


def accepted_providers() -> tuple[str, ...]:
    return CANONICAL_PROVIDERS + tuple(PROVIDER_ALIASES)


def supported_providers() -> str:
    """오류 메시지에 쓰는 공급자 목록. 별칭은 괄호에 묶는다."""
    aliases: dict[str, list[str]] = {}
    for alias, target in PROVIDER_ALIASES.items():
        aliases.setdefault(target, []).append(alias)

    return ", ".join(
        name + (f" ({'/'.join(aliases[name])})" if name in aliases else "")
        for name in CANONICAL_PROVIDERS
    )


def configured_model(provider: str | None = None) -> str:
    """그 공급자의 답변 모델 설정값."""
    name = normalize_provider(provider)
    if name not in MODEL_SETTING:
        raise ValueError(
            f"unknown LLM_PROVIDER '{name}'. supported: {supported_providers()}"
        )
    return str(getattr(settings, MODEL_SETTING[name]))


def create_chatbot(
    provider: str | None = None, model: str | None = None
) -> BaseChatbotModel:
    """공급자에 맞는 LLM 을 만든다.

    model 을 주면 그 공급자의 기본 모델 대신 쓴다. 요약에 더 작고 싼 모델을
    따로 태울 때 쓰는 통로다.
    """
    name = normalize_provider(provider)

    if name == "ollama":
        from raptor_qdrant.rag.llm.ollama import Ollama

        return Ollama(model=model)

    if name == "bedrock":
        from raptor_qdrant.rag.llm.bedrock import AmazonBedrock

        return AmazonBedrock(model_id=model)

    if name == "anthropic":
        from raptor_qdrant.rag.llm.claude import Claude

        return Claude(model=model)

    if name == "openai":
        from raptor_qdrant.rag.llm.chatgpt import ChatGPT

        return ChatGPT(model=model)

    raise ValueError(
        f"unknown LLM_PROVIDER '{name}'. supported: {supported_providers()}"
    )


class LazyChatbot(BaseChatbotModel):
    """처음 부를 때까지 공급자 클라이언트를 만들지 않는다.

    status 와 잎만 갱신하는 sync·watch 는 LLM 없이 끝난다. 엔진을 세울 때마다
    클라이언트를 붙이면 쓰지도 않을 자격증명과 서버를 요구하게 된다.
    """

    def __init__(
        self, provider: str | None = None, model: str | None = None
    ) -> None:
        self.provider = provider
        self.model = model
        self._llm: BaseChatbotModel | None = None

    def resolve(self) -> BaseChatbotModel:
        if self._llm is None:
            self._llm = create_chatbot(self.provider, self.model)
        return self._llm

    @property
    def describe(self) -> str:
        if self._llm is None:
            return f"{normalize_provider(self.provider)} (not started)"
        return self._llm.describe

    def complete(self, prompt: str) -> str:
        return self.resolve().complete(prompt)
