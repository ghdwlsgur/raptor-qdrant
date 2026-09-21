"""Qdrant, LLM, 임베딩 모델 없이 빌더를 돌리기 위한 대역들."""

from llama_index.core.schema import TextNode

from raptor_qdrant.rag.builder.models.structure import Node
from raptor_qdrant.rag.chunker.hybrid_chunker import BaseChunker
from raptor_qdrant.rag.embedding import BaseEmbeddingModel
from raptor_qdrant.rag.summarizer import BaseSummarizationModel


class CountingEmbedding(BaseEmbeddingModel):
    """배치 호출과 단건 호출을 따로 센다."""

    def __init__(self) -> None:
        self.batches: list[int] = []
        self.singles = 0

    def create_embedding(self, text: str) -> list[float]:
        self.singles += 1
        return [float(len(text)), 0.0]

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return [[float(len(text)), 0.0] for text in texts]

    @property
    def embedding_dimension(self) -> int:
        return 2

    @property
    def model_name(self) -> str:
        return "fake"


class WholeNoteChunker(BaseChunker):
    def chunk(self, text: str) -> list[TextNode]:
        return [TextNode(text=text)]


class FixedSummary(BaseSummarizationModel):
    def summarize(self, text: str) -> str:
        return "요약이다. 스무 글자는 넘겨야 쓸 만하다고 본다."


class RecordingSummary(BaseSummarizationModel):
    """받은 텍스트를 기록하고, 지정한 텍스트가 처음 올 때 한 번 실패한다."""

    def __init__(self, fail_once_on: str = "", unusable_on: str = "") -> None:
        self.calls: list[str] = []
        self.fail_once_on = fail_once_on
        self.unusable_on = unusable_on
        self._failed = False

    def summarize(self, text: str) -> str:
        self.calls.append(text)
        if (
            self.fail_once_on
            and self.fail_once_on in text
            and not self._failed
        ):
            self._failed = True
            raise RuntimeError("summarizer hiccup")
        if self.unusable_on and self.unusable_on in text:
            return "NO_SUMMARY"
        return (
            f"요약: {text.strip()[:12]} 에 대해 충분히 길게 쓴 요약 문장이다."
        )


def leaf(index: int, text: str = "") -> Node:
    return Node(
        text=text or f"node-{index}",
        index=index,
        children=set(),
        embeddings={"fake": [float(index), 0.0]},
    )


class SinglesOnly(BaseEmbeddingModel):
    """배치 메서드를 덮어쓰지 않은 모델. 기본 구현이 단건으로 돈다."""

    def create_embedding(self, text: str) -> list[float]:
        return [float(len(text))]

    @property
    def embedding_dimension(self) -> int:
        return 1

    @property
    def model_name(self) -> str:
        return "singles"
