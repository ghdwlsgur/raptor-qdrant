import pytest

from src.rag.utils import count_tokens, resolve_token_count


def test_uses_stored_count_when_present():
    assert resolve_token_count({"token_count": 42}, "무시된다") == 42


def test_parses_numeric_string():
    assert resolve_token_count({"token_count": "42"}, "무시된다") == 42


def test_zero_is_a_valid_stored_count():
    assert resolve_token_count({"token_count": 0}, "본문") == 0


@pytest.mark.parametrize(
    "stored",
    [None, -3, True, False, "n/a", "", 3.5, [], {}],
    ids=["none", "negative", "true", "false", "words", "empty", "float", "list", "dict"],
)
def test_falls_back_to_counting_the_text(stored):
    text = "신데렐라는 재를 골라냈다."
    assert resolve_token_count({"token_count": stored}, text) == count_tokens(text)


@pytest.mark.parametrize("metadata", [None, {}], ids=["none", "empty"])
def test_missing_metadata_falls_back(metadata):
    text = "신데렐라는 재를 골라냈다."
    assert resolve_token_count(metadata, text) == count_tokens(text)


def test_empty_text_counts_as_zero():
    assert count_tokens("") == 0
    assert resolve_token_count({}, "") == 0
