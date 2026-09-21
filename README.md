# raptor-qdrant

RAPTOR 방식의 계층형 RAG 파이프라인이다. 한국어 문서를 넣으면 요약 트리를 만들어 Qdrant에 적재하고, 질문을 던지면 트리 전체를 뒤져 답한다. 임베딩은 로컬 KURE-v1이 처리하고, 요약과 답변 생성은 Ollama(로컬)나 AWS Bedrock 중에 고른다.

## 왜 트리인가

보통의 RAG는 문서를 잘라 청크를 그대로 벡터DB에 넣는다. "3장에서 A가 왜 실패했나" 같은 질문은 잘 맞히지만 "전체 과정을 순서대로"처럼 문서를 넓게 봐야 하는 질문은 청크 몇 개로 답이 안 나온다. 검색이 가져온 조각들 사이의 맥락이 통째로 빠져 있기 때문이다.

RAPTOR는 여기에 층을 하나 더 쌓는다. 비슷한 청크끼리 묶어 요약하고 그 요약들을 다시 묶어 요약하기를 반복해 트리를 만든다. 검색할 때는 잎(원문 청크)과 상위 요약 노드를 같은 공간에 펼쳐두고 한꺼번에 뒤진다(collapsed tree). 세부 질문은 잎에서 걸리고 개괄 질문은 요약 노드에서 걸린다.

## 인덱싱 파이프라인

```
원본 텍스트
   │
   ├─ 1. 하이브리드 청킹 ......... src/rag/chunker/hybrid_chunker.py
   │     마크다운 헤더로 1차 분할 → 512토큰 넘는 섹션만 의미 기반 2차 분할
   │
   ├─ 2. 잎 노드 생성 ............ src/rag/builder/tree_builder.py
   │     KURE-v1 임베딩, 스레드풀 병렬 (실패 시 싱글스레드 폴백)
   │
   ├─ 3. 트리 구축 (최대 5층) .... src/rag/builder/cluster/
   │     UMAP 10차원 축소 → GMM 소프트 클러스터링 (전역 → 지역 2단계)
   │     클러스터 수는 BIC로 자동 결정, 한 노드가 여러 클러스터에 동시 소속 가능
   │     클러스터가 3500토큰을 넘으면 재귀적으로 재분할
   │     클러스터별로 LLM 요약 → 부모 노드
   │
   └─ 4. Qdrant 적재 ............. src/rag/retriever/qdrant_retriever.py
         모든 층의 노드를 dense + sparse 하이브리드로 저장
         payload: layer, node_index, document_name, chunked_by, token_count
```

층이 올라갈수록 노드 수가 줄고 남은 노드가 11개 이하가 되면 거기서 멈춘다.

## 질의 파이프라인

하이브리드 검색으로 상위 5개 노드를 가져오고(`alpha=0.8`, 벡터 8 대 키워드 2), 컨텍스트 예산 4096토큰이 찰 때까지 이어붙여 Bedrock에 넘긴다. 레이어 구분 없이 전부 후보에 넣는 것이 기본값이다.

```python
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant import EngineConfig, RaptorEngine

QdrantManager().connect()
engine = RaptorEngine(EngineConfig(collection_name="my-docs"))

engine.add_document(open("report.md").read(), document_name="report")

result = engine.query("핵심 결론이 뭔가요?")
print(result.answer)
for chunk in result.chunks:
    print(chunk["layer_number"], chunk["score"], chunk["token_count"])
```

`query()`는 답변과 근거 청크를 한 번에 돌려준다. 답변 문자열만 필요하면 `answer()`를 쓰면 되는데, 둘을 따로 부르면 같은 질문으로 검색이 두 번 돈다.

## 시작하기

**1. Qdrant 띄우기**

```bash
docker compose up -d qdrant
```

**2. 의존성 설치.** Python 3.12가 필요하다.

```bash
uv sync
```

**3. LLM 준비**

기본 공급자는 로컬 Ollama다. 서버를 띄우고 모델을 한 번 받아두면 된다.

```bash
ollama serve &
ollama pull qwen2.5:7b
```

Bedrock으로 돌리려면 `--llm bedrock`을 주거나 `LLM_PROVIDER=bedrock`을 설정한다. 이때 호출은 boto3의 표준 자격증명 체인을 타므로 `aws configure`로 잡아두었거나 `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`가 환경에 있으면 된다. 쓰려는 리전에서 해당 모델의 액세스를 미리 활성화해야 한다.

어느 쪽이든 문서 인덱싱을 시작하기 전에 모델이 실제로 응답하는지 먼저 확인한다. 임베딩과 청킹을 다 끝낸 뒤에 모델이 없다는 걸 알게 되면 그만큼이 버려진다.

**4. 실행**

```bash
uv run raptor-qdrant                                   # 기본 문서 인덱싱 + 예시 질문
uv run raptor-qdrant --file data/sample_en.txt         # 다른 문서
uv run raptor-qdrant --skip-index --question "질문은?"  # 이미 적재된 컬렉션에 질의만
```

Docker 로 통째로 띄우려면 compose 를 쓴다. Qdrant 가 함께 올라오고, Ollama 는
호스트에서 돌고 있는 것을 쓴다. macOS 에서 컨테이너 안의 Ollama 는 Metal 을
못 쓰기 때문이다.

```bash
docker compose run --rm app --file data/sample_ko.txt
```

| 옵션 | 설명 |
|---|---|
| `--file` | 인덱싱할 문서 경로 (기본 `data/sample_ko.txt`) |
| `--collection` | Qdrant 컬렉션 이름 (기본 `sample`) |
| `--document-name` | 문서 구분 이름 (기본: 파일 이름) |
| `--llm` | `ollama` 또는 `bedrock` (기본: `LLM_PROVIDER` 설정값) |
| `--question` | 질문. 여러 번 줄 수 있다 |
| `--skip-index` | 인덱싱 없이 질의만 |
| `--recreate` | 적재 전에 컬렉션을 통째로 삭제 |

## 설정

`.env` 파일이나 환경변수로 덮어쓴다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant 호스트 |
| `QDRANT_PORT` | `6333` | Qdrant 포트 |
| `EMBEDDING_MODEL` | `nlpai-lab/KURE-v1` | SentenceTransformer 모델 (항상 로컬) |
| `LLM_PROVIDER` | `ollama` | `ollama` 또는 `bedrock` |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama 모델 태그 |
| `BEDROCK_MODEL_ID` | `apac.anthropic.claude-3-7-sonnet-20250219-v1:0` | Bedrock 모델 |
| `AWS_REGION` | `ap-northeast-2` | Bedrock 리전 |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` / `critical` |
| `ENVIRONMENT` | `local` | `production`이면 JSON 구조화 로그로 전환 |

검색·청킹·트리 관련 수치는 `src/rag/constants.py`에 모여 있다. 청크 최대 토큰(512), 컨텍스트 예산(4096), top-k(5), 하이브리드 가중치(0.8) 같은 값들이다.

## 컬렉션과 문서

`collection_name`은 도메인이나 카테고리 단위, `document_name`은 그 안의 개별 문서 단위다. 한 컬렉션에 여러 문서를 담는 것이 기본 사용 방식이라, `add_document()`는 컬렉션을 지우지 않는다.

```python
engine.list_documents()  # 이 컬렉션의 문서명 목록
engine.update_document(text, "report")  # 기존 문서 교체
engine.delete_document("report")  # 문서 단위 삭제
engine.add_document(
    text, "report", recreate_collection=True
)  # 컬렉션 통째로 비우고 적재
```

같은 `document_name`으로 두 번 `add_document()`를 부르면 중복 적재 대신 예외가 난다. 교체할 생각이었다면 `update_document()`를 쓰라는 뜻이다.

## 개발

품질 검사는 CI 가 도는 것과 같다. 테스트는 Qdrant 나 LLM 없이 순수 로직만 본다.

```bash
uv run pytest              # 테스트
uv run ruff check .        # 린트
uv run ruff format .       # 포맷
uv run mypy                # 타입 검사
```

테스트가 지키는 것은 이렇다.

| 파일 | 무엇을 지키는가 |
|---|---|
| `test_chunk_tagging.py` | 청크마다 토큰 수를 따로 센다 (부모 값을 물려받으면 검색이 죽는다) |
| `test_context_window.py` | 큰 노드 하나가 컨텍스트 전체를 비우지 않는다 |
| `test_token_count.py` | `token_count`가 없거나 망가져도 터지지 않는다 |
| `test_summary_sentinel.py` | 모델이 장식을 붙인 `NO_SUMMARY`도 걸러낸다 |
| `test_chunk_metadata.py` | payload가 JSON으로 직렬화된다 |
| `test_tree_structure.py` | 노드와 레이어 매핑 |
| `test_clustering_utils.py` | 데이터가 적을 때의 클러스터 수 경계 |
| `test_llm_factory.py` | 공급자 선택과 오류 메시지 |

## 구조

```
src/raptor_qdrant/
├── cli.py                     콘솔 스크립트 진입점
├── core/
│   ├── config.py              pydantic-settings 기반 환경 설정
│   └── logger.py              표준 logging을 loguru로 넘기고 KST로 출력
├── database/
│   └── qdrant_manager.py      연결(싱글톤), 컬렉션·포인트 운영
└── rag/
    ├── engine.py              RaptorEngine, 이 패키지의 정문
    ├── embedding.py           KURE-v1 임베딩
    ├── chunker/               마크다운 + 의미 기반 하이브리드 청킹
    ├── builder/               RAPTOR 트리 구축
    │   ├── tree_builder.py    잎 노드 생성, 트리 조립
    │   └── cluster/           UMAP 차원 축소 + GMM 소프트 클러스터링
    ├── summarizer/            클러스터 요약
    ├── retriever/             Qdrant 하이브리드 검색, 컨텍스트 예산 조립
    ├── llm/                   LLM 공급자 (base·ollama·bedrock + 팩토리)
    └── prompt/                요약·답변 프롬프트
tests/                         Qdrant·LLM 없이 도는 단위 테스트
```

임베딩 모델, 요약 모델, LLM, 청커, 클러스터링 알고리즘은 모두 추상 기반 클래스를 두고 `EngineConfig`로 주입한다. 다른 구현으로 갈아끼우려면 해당 기반 클래스만 상속하면 된다.

## 알아둘 것

**컬렉션 스키마는 LlamaIndex가 만든다.** 하이브리드 검색에 쓰는 dense·sparse 벡터의 이름 규약은 `QdrantVectorStore`가 정한다. 이 저장소는 같은 스키마를 따로 만들지 않는다. 양쪽이 각자 만들면 이름이 어긋나도 예외가 안 나고 하이브리드 검색이 조용히 반쪽만 동작한다.

**인덱싱 비용은 문서 크기에 비례해 커진다.** 클러스터 하나당 LLM 호출이 한 번씩 들어가고 층이 올라갈 때마다 반복된다. Bedrock이면 요금이, Ollama면 시간이 그만큼 든다. 큰 문서를 넣기 전에 작은 것으로 먼저 확인하는 편이 낫다.

**요약 동시 실행 수는 공급자에 따라 다르다.** 로컬 모델은 요청을 직렬로 처리하므로 Ollama는 2, Bedrock은 10으로 잡아둔다. `src/rag/constants.py`의 `SUMMARIZATION_MAX_WORKERS`에서 바꾼다.

**첫 실행은 느리다.** KURE-v1과 sparse 인코더 모델을 내려받는다.

**로컬·스테이징 로그는 변수 값까지 찍는다.** loguru의 `diagnose=True` 설정이라 예외가 나면 스택 프레임의 지역 변수가 그대로 출력된다. 로그를 공유할 일이 있으면 `src/core/logger.py`에서 끄거나 `ENVIRONMENT=production`으로 돌린다.

## 참고

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1): 한국어 검색 특화 임베딩 모델
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)

## 라이선스

MIT. [LICENSE](LICENSE) 참고.
