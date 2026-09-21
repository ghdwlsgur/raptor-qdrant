import json

import pytest

from raptor_qdrant.rag.chunker.models.chunk_metadata import (
    ChunkingMethod,
    ChunkMetadata,
)


@pytest.mark.parametrize("method", list(ChunkingMethod))
def test_serializes_method_as_plain_string(method):
    payload = ChunkMetadata(chunked_by=method, token_count=10).to_dict()

    assert payload["chunked_by"] == method.value
    assert type(payload["chunked_by"]) is str


@pytest.mark.parametrize("method", list(ChunkingMethod))
def test_payload_survives_json_round_trip(method):
    payload = ChunkMetadata(chunked_by=method, token_count=10).to_dict()

    assert json.loads(json.dumps(payload)) == payload


def test_keeps_token_count():
    assert (
        ChunkMetadata(ChunkingMethod.SUMMARY, 137).to_dict()["token_count"]
        == 137
    )
