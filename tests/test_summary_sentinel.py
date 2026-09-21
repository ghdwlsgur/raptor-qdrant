import pytest

from src.rag.summarizer import NO_SUMMARY, is_unusable_summary


@pytest.mark.parametrize(
    "response",
    [
        NO_SUMMARY,
        ":NO_SUMMARY",
        "  NO_SUMMARY  ",
        "**NO_SUMMARY**",
        "- NO_SUMMARY",
        "`NO_SUMMARY`",
        "no_summary",
        "(NO_SUMMARY)",
        "",
        "   ",
        "너무 짧음",
    ],
)
def test_rejects_sentinel_and_too_short(response):
    assert is_unusable_summary(response)


@pytest.mark.parametrize(
    "response",
    [
        "신데렐라는 의붓어머니의 허락을 얻지 못하고 왕궁 축제에 참석했다.",
        "왕자는 계단에 송진을 발라 슬리퍼가 붙게 만들었고 그것으로 신부를 찾았다.",
    ],
)
def test_accepts_real_summaries(response):
    assert not is_unusable_summary(response)
