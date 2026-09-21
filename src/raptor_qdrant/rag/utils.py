from functools import cache
from pathlib import Path
from typing import Any

import tiktoken

from raptor_qdrant.rag.constants import DEFAULT_ENCODING

TOKEN_COUNT_KEY = "token_count"


def load_prompt(filename: str) -> str:
    current_dir = Path(__file__).parent

    with open(current_dir / filename, encoding="utf-8") as file:
        return file.read()


@cache
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING):
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(get_tokenizer().encode(text))


def _as_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return None

    if isinstance(value, int) and value >= 0:
        return value

    return None


def resolve_token_count(
    metadata: dict[str, Any] | None, text: str = ""
) -> int:
    stored = _as_non_negative_int((metadata or {}).get(TOKEN_COUNT_KEY))
    return stored if stored is not None else count_tokens(text)
