from raptor_qdrant.eval.report import outcome_of, score


def chunk(layer: int, sources: list[str], tokens: int = 100) -> dict:
    return {
        "layer_number": layer,
        "sources": sources,
        "token_count": tokens,
    }


def test_the_first_chunk_holding_the_answer_sets_the_rank():
    outcome = outcome_of(
        "질문",
        ["정답.md"],
        [chunk(0, ["다른.md"]), chunk(1, ["정답.md"]), chunk(0, ["정답.md"])],
    )

    assert outcome.hit_rank == 2
    assert outcome.leaf_hit_rank == 3


def test_a_miss_leaves_both_ranks_empty():
    outcome = outcome_of("질문", ["정답.md"], [chunk(0, ["다른.md"])])

    assert outcome.hit_rank is None
    assert outcome.leaf_hit_rank is None


def test_summaries_and_tokens_are_counted():
    outcome = outcome_of(
        "질문",
        ["정답.md"],
        [chunk(0, ["정답.md"], 100), chunk(2, ["정답.md"], 250)],
    )

    assert outcome.summaries == 1
    assert outcome.tokens == 350
    assert outcome.chunks == 2


def test_recall_separates_summary_hits_from_leaf_hits():
    only_summary = outcome_of("q1", ["a.md"], [chunk(1, ["a.md"])])
    with_leaf = outcome_of("q2", ["b.md"], [chunk(0, ["b.md"])])
    missed = outcome_of("q3", ["c.md"], [chunk(0, ["x.md"])])

    report = score([only_summary, with_leaf, missed], max_tokens=1000)

    assert report.recall == 2 / 3
    assert report.coverage == 2 / 3
    assert report.leaf_recall == 1 / 3
    assert len(report.misses) == 1


def test_mrr_rewards_an_earlier_hit():
    first = outcome_of("q", ["a.md"], [chunk(0, ["a.md"])])
    fourth = outcome_of(
        "q", ["a.md"], [chunk(0, ["x.md"])] * 3 + [chunk(0, ["a.md"])]
    )

    assert score([first], 1000).mrr == 1.0
    assert score([fourth], 1000).mrr == 0.25


def test_an_empty_run_scores_zero_without_dividing_by_zero():
    report = score([], max_tokens=1000)

    assert report.recall == 0.0
    assert report.mrr == 0.0
    assert report.budget_use == 0.0


def test_a_bare_string_source_is_not_read_as_letters():
    outcome = outcome_of("질문", "정답.md", [chunk(0, ["정"])])

    assert outcome.hit_rank is None


def test_an_older_eval_file_with_one_source_still_loads(tmp_path):
    import json

    from raptor_qdrant.eval.dataset import load_cases

    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps(
            {"question": "질문", "source": "정답.md"}, ensure_ascii=False
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].sources == ("정답.md",)
    assert cases[0].kind == "note"


def test_a_summary_name_drop_does_not_count_as_real_evidence():
    outcome = outcome_of(
        "질문",
        ["a.md", "b.md", "c.md"],
        [chunk(2, ["a.md", "b.md", "c.md"]), chunk(0, ["a.md"])],
    )

    assert outcome.coverage == 1.0
    assert outcome.leaf_coverage == 1 / 3


def test_leaf_coverage_counts_every_leaf_that_carried_a_source():
    outcome = outcome_of(
        "질문",
        ["a.md", "b.md"],
        [chunk(0, ["a.md"]), chunk(0, ["b.md"])],
    )

    assert outcome.leaf_coverage == 1.0
