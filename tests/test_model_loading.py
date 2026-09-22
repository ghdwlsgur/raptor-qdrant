"""가중치를 올리는 자리는 스레드가 여럿이어도 한 번만 돌아야 한다."""

import threading
import time

from raptor_qdrant.rag.embedding import shared_model


def test_one_build_even_when_threads_race():
    store: dict = {}
    builds = []

    def build() -> str:
        builds.append(1)
        time.sleep(0.05)
        return "모델"

    results = []
    threads = [
        threading.Thread(
            target=lambda: results.append(shared_model(store, "kure", build))
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1
    assert results == ["모델"] * 8


def test_different_keys_build_separately():
    store: dict = {}

    first = shared_model(store, "a", lambda: "가")
    second = shared_model(store, "b", lambda: "나")

    assert (first, second) == ("가", "나")
    assert shared_model(store, "a", lambda: "다") == "가"


def test_a_nested_build_does_not_deadlock():
    store: dict = {}

    def outer() -> str:
        return shared_model(store, "inner", lambda: "안쪽") + " 바깥"

    assert shared_model(store, "outer", outer) == "안쪽 바깥"
