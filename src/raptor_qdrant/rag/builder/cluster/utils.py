import logging
import random

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 224
random.seed(RANDOM_SEED)

logger = logging.getLogger(__name__)


def reduce_embedding_dimensions(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """고차원의 복잡한 임베딩 데이터를 UMAP라는 알고리즘을 사용해
    저차원의 단순한 데이터로 '압축' 또는 '요약'하는 역할을 수행
    텍스트 임베딩 데이터는 보통 수백~수천 차원의 아주 높은 고차원 공간에 존재하며
    이런 고차원 데이터는 차원의 저주, 계산 복잡성, 시각화 어려움 등의 문제를 야기

    이러한 문제를 해결하기 위해, 데이터의 '핵심적인 구조'를 최대한 유지하면서
    차원만 낮은 공간으로 옮기는 '차원 축소' 과정이 필요

    https://pair-code.github.io/understanding-umap/
    """
    # 데이터 수가 너무 적으면 차원 축소 없이 원본 반환
    if len(embeddings) <= 2:
        logger.warning(
            f"Too few embeddings ({len(embeddings)}) for UMAP reduction, returning original"
        )
        return embeddings[:, :dim] if embeddings.shape[1] > dim else embeddings

    # n_neighbors는 하나의 점이 주변의 몇 개의 점을'이웃'으로 고려할지를 결정하는 파라미터
    # 이 값이 작을수록 국소적인 구조에 집중하고, 클수록 더 넓은 범위의 구조를 반영
    # "rule of thumb"으로 n_neighbors를 데이터 포인트 수의 제곱근으로 설정
    if n_neighbors is None:
        n_neighbors = max(
            2, min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)
        )

    # UMAP 모델 초기화, UMAP 알고리즘을 사용해 고차원 임베딩을 저차원(dim)으로 축소
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors,  # 시야 설정 (현미경 vs 망원경)
        n_components=dim,  # 목표 차원 설정
        metric=metric,  # 원본 공간에서의 거리 측정 방식 (e.g., 코사인, 유클리드...)
    ).fit_transform(embeddings)  # 학습 및 변환 실행

    # 저차원으로 축소된 임베딩 반환
    return reduced_embeddings


def get_optimal_cluster_count(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    random_state: int = RANDOM_SEED,
) -> int:
    """최적의 클러스터 수를 BIC 기준으로 결정,
    "주어진 데이터를 과연 몇 개의 그룹(클러스터)으로 나누는 것이 가장 적절한가?"를 판단
    그룹 수가 너무 적으면 서로 다른 특성을 가진 데이터가 억지로 한 그룹에 묶이고
    그룹 수가 너무 많으면 사실은 같은 그룹인 데이터가 불필요하게 여러 개로 쪼개짐

    BIC는 무작정 데이터를 잘 맞추기만 하는 복잡한 모델보다 어느 정도 데이터를
    잘 설명하면서도 가장 단순한 모델을 선호
    """
    # 찾으려는 클러스터의 최대 개수가 실제 데이터의 개수보다 많을 수 없음
    max_clusters = min(max_clusters, len(embeddings))
    # 데이터가 너무 적으면 클러스터를 1개로 제한
    if len(embeddings) <= 3:
        return 1

    # 최대 클러스터 수를 데이터 수의 절반으로 제한, 비효율적인 탐색 방지
    max_clusters = min(max_clusters, len(embeddings) // 2)
    # max_clusters가 1보다 작으면 클러스터 수를 1로 설정
    if max_clusters < 1:
        return 1

    # 1부터 max_clusters까지 정수 배열 생성, 각 값을 클러스터 수(k) 후보로 사용
    n_clusters = np.arange(1, max_clusters + 1)
    # 각 클러스터 수 후보에 대해 BIC 점수를 계산하여 저장할 리스트
    bics = []

    # 모든 k 후보에 대해 클러스터링을 시도하고 BIC 점수 계산
    for n in n_clusters:
        try:
            # GMM 모델 초기화
            gm = GaussianMixture(
                n_components=n,  # 현재 클러스터 수(k)
                random_state=random_state,  # 결과의 재현성을 위한 시드
                reg_covar=1e-6,  # 공분산 행렬에 더해지는 값으로, 수치적 안정성 향상
                max_iter=100,  # GMM 알고리즘의 최대 반복 횟수
                tol=1e-3,  # 수렴을 결정하는 허용 오차
            )
            # 주어진 임베딩 데이터로 GMM 모델 학습
            gm.fit(embeddings)

            # 학습된 모델의 BIC 점수를 계산하여 리스트에 추가
            bics.append(gm.bic(embeddings))
        except ValueError:
            # 수치적 불안정성으로 실패한 경우 매우 큰 BIC 값을 할당, 최적이 아님을 표시
            bics.append(np.inf)

    # 유효한 BIC가 없으면 1을 반환
    if all(bic == np.inf for bic in bics):
        return 1

    # bics 리스트에서 가장 낮은 BIC 점수를 가진 값의 인덱스를 찾음.
    # 해당 인덱스는 n_clusters 배열에서 최적의 클러스터 수에 해당
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def gmm_soft_cluster(
    embeddings: np.ndarray, threshold: float, random_state: int = 0
) -> tuple[list[np.ndarray], int]:
    """가우시안 혼합 모델(GMM)을 사용해 소프트 클러스터링 수행, 하나의 데이터가
    경계선에 걸쳐 있을 경우 여러 그룹에 동시에 속할 수 있도록 허용

    :param embeddings: 클러스터링을 수행할 임베딩 데이터
    :param threshold: 클러스터에 속하기 위한 최소 확률 임계값
    :param random_state: 결과 재현성을 위한 랜덤 시드

    :return: 각 데이터 포인트에 할당된 클러스터 라벨 리스트와, 결정된 클러스터의 총 개수를 튜플로 반환.
    (e.g., ([np.array([0]), np.array([1, 2])], 3))
    """
    # BIC를 기준으로 데이터에 가장 적합한 클러스터 수 결정
    n_clusters = get_optimal_cluster_count(embeddings)
    try:
        # 결정된 클러스터 개수로 GMM 모델 초기화
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=random_state,
            reg_covar=1e-6,
            max_iter=100,
            tol=1e-3,
        )
        gm.fit(embeddings)

        # 각 데이터가 모든 클러스터에 속할 '확률' 계산 (소프트 클러스터링)
        # 결과: (데이터 수, 클러스터 수) 형태의 2D 배열
        # e.g., probs[0] = [0.1, 0.7, 0.2] -> 첫 번째 데이터는 두 번째 클러스터에 속할 확률이 가장 높음
        probs = gm.predict_proba(embeddings)

        # 계산된 확률이 주어진 임계값을 넘는 모든 클러스터의 인덱스를 라벨로 할당
        labels = [np.where(prob > threshold)[0] for prob in probs]

        # 계산된 라벨과 클러스터의 개수 반환
        return labels, n_clusters
    except ValueError:
        # GMM이 실패한 경우 모든 포인트를 하나의 클러스터로 할당, 모든 데이터를 클러스터(0)에 포함
        labels = [np.array([0]) for _ in range(len(embeddings))]
        return labels, 1
