"""검색을 재기 위한 질문 모음.

노트에서 질문을 만들고 그 노트를 정답으로 삼는다. 사람이 라벨을 붙이지 않아도
"이 질문에 이 노트가 나와야 한다"는 기준이 생긴다. 완벽한 정답지는 아니다.
같은 주제를 다룬 다른 노트가 나와도 틀렸다고 세므로 점수는 실제보다 박하게
나온다. 대신 같은 자로 재기 때문에 변경 전후를 견줄 수 있다.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from raptor_qdrant.core.config import settings
from raptor_qdrant.rag.llm import BaseChatbotModel
from raptor_qdrant.rag.utils import count_tokens
from raptor_qdrant.vault.loader import VaultNote

logger = logging.getLogger(__name__)

# 이보다 짧은 노트는 물어볼 것이 없다
MIN_NOTE_TOKENS = 300
# 질문을 만들 때 노트에서 읽는 앞부분
NOTE_PREVIEW_CHARS = 3000
SAMPLING_SEED = 224

QUESTION_PROMPT = """다음은 어떤 사람이 자기 지식 저장소에 적어 둔 노트다.

이 사람이 몇 달 뒤 이 노트를 다시 찾으려 할 때 던질 법한 질문을 하나만 써라.

- 노트의 문장을 그대로 베끼지 마라. 기억을 더듬어 묻는 말투로 써라.
- 이 노트가 답이 되는 질문이어야 한다. 너무 일반적이면 안 된다.
- 한국어 한 문장으로, 질문만 출력하라. 다른 말은 붙이지 마라.

[노트: {title}]
{text}
"""


@dataclass(frozen=True)
class EvalCase:
    question: str
    sources: tuple[str, ...]
    # note: 노트 하나가 답. broad: 여러 노트에 걸친 질문
    kind: str = "note"


def dataset_path(collection: str, state_dir: str | None = None) -> Path:
    """평가셋이 놓이는 곳. 저장소가 아니라 상태 디렉토리다.

    질문은 볼트 내용에서 나온 것이라 저장소에 두면 사적인 내용이 함께
    공개된다. 도구는 저장소에, 데이터는 여기에 둔다.
    """
    root = Path(state_dir or settings.STATE_DIR).expanduser()
    return root / "eval" / f"{collection}.jsonl"


def eligible(notes: list[VaultNote]) -> list[VaultNote]:
    return [
        note
        for note in notes
        if count_tokens(note.text) >= MIN_NOTE_TOKENS and not note.is_empty
    ]


def build_cases(
    notes: list[VaultNote], llm: BaseChatbotModel, count: int
) -> list[EvalCase]:
    """노트를 골라 질문을 하나씩 만든다."""
    pool = eligible(notes)
    if not pool:
        raise ValueError(
            f"no notes with at least {MIN_NOTE_TOKENS} tokens to ask about"
        )

    chosen = random.Random(SAMPLING_SEED).sample(pool, min(count, len(pool)))
    cases: list[EvalCase] = []

    for i, note in enumerate(chosen, start=1):
        prompt = QUESTION_PROMPT.format(
            title=note.title, text=note.text[:NOTE_PREVIEW_CHARS]
        )
        try:
            question = llm.complete(prompt).strip().splitlines()[0].strip()
        except Exception as e:
            logger.warning(f"could not build a question for {note.path}: {e}")
            continue

        if not question:
            continue
        cases.append(
            EvalCase(question=question, sources=(note.path,), kind="note")
        )
        logger.info(f"built {i}/{len(chosen)}: {question[:60]}")

    return cases


BROAD_PROMPT = """다음은 어떤 사람의 노트 여러 개를 묶어 만든 요약이다.

이 사람이 이 묶음 전체를 다시 들여다보려 할 때 던질 법한 질문을 하나만 써라.

- 노트 하나만으로는 답이 안 되고 묶음을 봐야 답이 되는 질문이어야 한다.
- 요약의 문장을 그대로 베끼지 마라. 기억을 더듬어 묻는 말투로 써라.
- 한국어 한 문장으로, 질문만 출력하라. 다른 말은 붙이지 마라.

[요약]
{text}
"""

# 노트를 이만큼은 덮는 요약에서만 가로지르는 질문을 만든다
MIN_BROAD_SOURCES = 3


def build_broad_cases(
    summaries: list[tuple[str, list[str]]], llm: BaseChatbotModel, count: int
) -> list[EvalCase]:
    """요약 노드에서 가로지르는 질문을 만든다.

    노트 하나로 만든 질문은 그 노트의 어휘를 그대로 물고 있어 검색이 거의
    틀리지 않는다. 자가 천장에 닿으면 개선을 잴 수 없다. 요약이 덮는 노트
    전부를 정답으로 두면 트리가 실제로 쓰이는지를 잰다.
    """
    pool = [
        (text, sources)
        for text, sources in summaries
        if len(sources) >= MIN_BROAD_SOURCES
    ]
    if not pool:
        raise ValueError(
            f"no summary covers at least {MIN_BROAD_SOURCES} notes"
        )

    chosen = random.Random(SAMPLING_SEED).sample(pool, min(count, len(pool)))
    cases: list[EvalCase] = []

    for i, (text, sources) in enumerate(chosen, start=1):
        try:
            question = (
                llm.complete(BROAD_PROMPT.format(text=text))
                .strip()
                .splitlines()[0]
                .strip()
            )
        except Exception as e:
            logger.warning(f"could not build a broad question: {e}")
            continue

        if not question:
            continue
        cases.append(
            EvalCase(
                question=question,
                sources=tuple(sorted(sources)),
                kind="broad",
            )
        )
        logger.info(f"built broad {i}/{len(chosen)}: {question[:60]}")

    return cases


def save_cases(cases: list[EvalCase], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(asdict(case), ensure_ascii=False) + "\n")
    return path


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise ValueError(
            f"no eval set at {path}. build one with `raptor-qdrant eval --build`"
        )

    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            # 예전 평가셋은 정답을 source 하나로 적었다
            sources = raw.get("sources") or [raw["source"]]
            cases.append(
                EvalCase(
                    question=raw["question"],
                    sources=tuple(sources),
                    kind=raw.get("kind", "note"),
                )
            )
    return cases
