# raptor-qdrant

옵시디언 볼트를 위한 RAPTOR 계층형 RAG다. 볼트 전체에 요약 트리 하나를 올려 Qdrant에 적재하고, 질문에는 근거 노트를 함께 답한다. 임베딩은 로컬 KURE-v1이 처리하고, 요약과 답변은 Ollama(로컬), Claude, ChatGPT, AWS Bedrock 중에 고른다.

## 왜 트리인가

보통의 RAG는 문서를 잘라 청크를 그대로 벡터DB에 넣는다. "3장에서 A가 왜 실패했나" 같은 질문은 잘 맞히지만 "전체 과정을 순서대로"처럼 문서를 넓게 봐야 하는 질문은 청크 몇 개로 답이 안 나온다. 검색이 가져온 조각들 사이의 맥락이 통째로 빠져 있기 때문이다.

RAPTOR는 여기에 층을 하나 더 쌓는다. 비슷한 청크끼리 묶어 요약하고 그 요약들을 다시 묶어 요약하기를 반복해 트리를 만든다. 검색할 때는 잎(원문 청크)과 상위 요약 노드를 같은 공간에 펼쳐두고 한꺼번에 뒤진다(collapsed tree). 세부 질문은 잎에서 걸리고 개괄 질문은 요약 노드에서 걸린다.

### 볼트 전체에 트리 하나를 올리는 이유

노트마다 트리를 따로 세우면 RAPTOR가 아무 일도 하지 않는다. 노트 하나는 보통 청크 대여섯 개라 클러스터링 중단 임계(`reduction_dimension + 1`, 기본 11)에 못 미쳐 요약 레이어가 생기지 않는다. 그러면 평범한 청크 검색과 다를 게 없다.

값어치는 노트를 가로지르는 데서 나온다. 여러 노트에 흩어진 내용이 한 클러스터로 묶이고 그 요약이 검색 대상이 되어야 "이 주제로 내가 남긴 것 전반" 같은 질문에 답할 수 있다. 그래서 `build_from_documents`가 노트별로 청킹해 잎마다 출처를 박은 뒤, 그 잎 전체에 트리 하나를 세운다.

## 인덱싱 파이프라인

```
원본 텍스트
   │
   ├─ 1. 하이브리드 청킹 ......... src/raptor_qdrant/rag/chunker/hybrid_chunker.py
   │     마크다운 헤더로 1차 분할 → 512토큰 넘는 섹션만 의미 기반 2차 분할
   │
   ├─ 2. 잎 노드 생성 ............ src/raptor_qdrant/rag/builder/tree_builder.py
   │     KURE-v1 임베딩을 64개 단위 배치로. 끝나는 즉시 Qdrant에 적재
   │
   ├─ 3. 트리 구축 (최대 5층) .... src/raptor_qdrant/rag/builder/cluster/
   │     UMAP 10차원 축소 → GMM 소프트 클러스터링 (전역 → 지역 2단계)
   │     클러스터 수는 BIC로 자동 결정, 한 노드가 여러 클러스터에 동시 소속 가능
   │     클러스터가 3500토큰을 넘으면 재귀적으로 재분할
   │     클러스터별로 LLM 요약 → 배치 임베딩 → 부모 노드. 레이어가 끝날 때마다 적재
   │     실패한 클러스터만 한 번 더 시도하고, 그래도 안 되면 그 클러스터만 뺀다
   │
   └─ 4. 세대 교체 ............... src/raptor_qdrant/rag/retriever/qdrant_retriever.py
         모든 노드가 같은 tree_generation을 달고 dense + sparse 하이브리드로 저장
         트리가 완성되면 그 세대가 아닌 포인트(이전 트리)를 지운다
         payload: layer, node_index, document_name, source_notes, content_hash, tree_generation
```

층이 올라갈수록 노드 수가 줄고 남은 노드가 11개 이하가 되면 거기서 멈춘다.

진행은 로그로 본다. 잎 임베딩은 배치마다, 요약은 레이어별 10% 단위로 남기고, 기본으로 `~/.local/state/raptor-qdrant/raptor-qdrant.log`에도 같은 내용이 쓰인다. 다른 터미널에서 `tail -f`로 보면 된다. 중간에 죽어도 그때까지 적재된 레이어는 남아 있고, 다음 `index`가 끝나지 않은 세대를 치우고 새로 시작한다.

## 질의 파이프라인

검색으로 후보 48개를 받아 그중 12개를 고른다. 고를 때 요약 노드 자리 4개를 남긴다. 포인트의 85%가 잎이라 점수순으로 그냥 자르면 개괄 질문에도 잎만 남기 때문이다. 후보에 요약이 없으면 그 자리는 잎으로 메운다.

`alpha`는 1.0, 곧 dense 단독이다. 컬렉션은 sparse 벡터도 들고 있지만 가중치를 주지 않는다. 라이브러리 기본 sparse 모델이 영어 전용이라 한국어 볼트에서는 점수를 깎는다. 질문 45개로 쓸어보니 dense 단독이 모든 지표에서 가장 좋았고(note recall@1 96% → 100%), sparse 비중을 올릴수록 무너졌다. 한국어를 아는 sparse 모델로 갈아끼우면 그때 다시 잴 값이다.

요약이 걸리면 그 요약이 덮는 노트로 범위를 좁혀 원문을 여섯 개까지 더 가져온다. 요약은 자기가 덮는 노트 이름을 전부 달고 있어서 검색 성적표에는 잘 찍히지만, 답을 쓰려면 이름이 아니라 문장이 필요하다. 이 확장이 없을 때 가로지르는 질문에서 정답 노트의 34%만 원문으로 왔다. 확장과 예산을 함께 올려 63%가 됐다. 둘 중 하나만 해서는 듣지 않는다.

고른 청크는 컨텍스트 예산 8192토큰이 찰 때까지 이어붙여 LLM에 넘긴다. 각 청크에는 `[근거 N | 원문 또는 요약 | 출처]` 머리표를 붙인다. 본문만 넘기면 모델은 무엇이 원문이고 무엇이 요약인지 모른 채 답한다. 요약은 자기가 덮는 노트를 전부 달고 있어서 특정 주장의 직접 근거로 쓰면 안 되는데, 그 구분도 본문만으로는 전할 수 없다. 같은 본문이 두 번 들어오면 두 번째는 버린다.

`start_layer`로 레이어를 지정하면 그 조건이 Qdrant에 필터로 내려간다. 받아온 뒤에 거르면 그 레이어의 좋은 후보는 애초에 후보에 없다.

`RERANKER_MODEL`을 주면 고르기 전에 크로스 인코더가 후보를 다시 줄 세운다. 하이브리드 검색은 질문과 청크를 따로 벡터로 만들어 견주지만 크로스 인코더는 둘을 함께 읽는다. 그만큼 잘 맞히고 그만큼 느리다. 기본값은 꺼짐이다. 볼트 질문 45개로 잰 값이다.

| 설정 | broad recall@1 | broad coverage | 질의당 |
|---|---|---|---|
| 끔 | 85% | 90% | 0.1초 |
| 켬, `RERANK_CANDIDATES=16` | 90% | 91% | 3초 |
| 켬, 후보 48개 전부 | 90% | 94% | 18~28초 |

상위 한 자리 정확도는 열여섯 개만 채점해도 얻는다. coverage는 아래쪽 후보를 끌어올려야 오르는 값이라 전부 채점해야 한다. CPU 기준이고 GPU에서는 셈이 달라진다.

```python
from raptor_qdrant.database.qdrant_manager import QdrantManager
from raptor_qdrant.vault import VaultLoader
from raptor_qdrant import EngineConfig, RaptorEngine

QdrantManager().connect()
engine = RaptorEngine(EngineConfig(collection_name="obsidian"))

notes = VaultLoader("~/Documents/Obsidian Vault").load()
engine.add_corpus(
    {note.path: note.text for note in notes},
    note_hashes={note.path: note.content_hash for note in notes},
)

result = engine.query("이 주제로 내가 정리해둔 게 뭐가 있나?")
print(result.answer)
print(result.sources)
```

`query()`는 답변과 근거를 한 번에 돌려준다. 답변 문자열만 필요하면 `answer()`를 쓰면 되는데, 둘을 따로 부르면 같은 질문으로 검색이 두 번 돈다. `result.sources`는 답변이 참고한 노트를 관련도 순서로 준다. 요약 노드는 자기가 덮는 노트를 전부 물고 있어서 상위 레이어가 걸려도 출처를 잃지 않는다.

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

다른 공급자는 `--llm`으로 고르거나 `LLM_PROVIDER`로 설정한다.

| 공급자 | `--llm` | 자격증명 |
|---|---|---|
| Ollama (로컬) | `ollama` | 없음 |
| Claude | `claude` 또는 `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_OAUTH_TOKEN_FILE`, 또는 `ant auth login` 프로필 |
| ChatGPT | `chatgpt` 또는 `openai` | `OPENAI_API_KEY` |
| Bedrock | `bedrock` | boto3 표준 체인(`aws configure`, `AWS_ACCESS_KEY_ID` 등) |

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uv run raptor-qdrant --llm claude ask "질문"

export OPENAI_API_KEY=sk-...
uv run raptor-qdrant --llm chatgpt ask "질문"
```

Bedrock은 쓰려는 리전에서 해당 모델의 액세스를 미리 활성화해야 한다.

OAuth 토큰은 값을 설정에 넣지 않고 파일 경로만 준다. 값을 `.env`나 환경변수로 옮기면 프로세스 목록과 자식 프로세스에 그대로 흘러간다. 읽기는 런타임에 한 번뿐이고 로그에는 남지 않는다.

볼트 본문이 밖으로 나가는 공급자(`claude`, `chatgpt`, `bedrock`)로 인덱싱하려면 `--allow-remote-llm`을 함께 줘야 한다. 실수로 볼트 전체를 남의 서버에 보내는 일을 막으려는 것이다.

어느 쪽이든 인덱싱을 시작하기 전에 모델이 실제로 응답하는지 먼저 확인한다. 임베딩과 청킹을 다 끝낸 뒤에 모델이 없다는 걸 알게 되면 그만큼이 버려진다. Claude와 ChatGPT는 시작할 때 본 호출과 같은 모양으로 열여섯 토큰짜리 요청을 한 번 보내 본다. 키와 모델 접근뿐 아니라 그 모델이 파라미터를 받아들이는지까지 그때 걸러진다.

**4. 실행**

```bash
uv run raptor-qdrant index                  # 볼트 전체를 새로 인덱싱
uv run raptor-qdrant sync --dry-run         # 무엇이 바뀌었는지만 확인
uv run raptor-qdrant sync                   # 변경분 반영
uv run raptor-qdrant watch                  # 저장될 때마다 잎 갱신
uv run raptor-qdrant ask "질문"              # 질의
uv run raptor-qdrant status                 # 적재 현황
uv run raptor-qdrant eval                   # 검색 성적 측정
```

Docker로 통째로 띄우려면 compose를 쓴다. Qdrant가 함께 올라오고, Ollama는
호스트에서 돌고 있는 것을 쓴다. macOS에서 컨테이너 안의 Ollama는 Metal을
못 쓰기 때문이다.

```bash
docker compose run --rm app index
```

| 명령 | 하는 일 |
|---|---|
| `index` | 볼트 전체로 트리를 새로 세운다. 도는 동안 락을 잡고, 끝나면 그 사이 바뀐 노트를 따라잡는다 |
| `sync` | 볼트와 인덱스를 비교해 바뀐 노트의 잎만 갈아끼운다. `--dry-run`으로 미리 보고, `--rebuild-tree`면 요약 레이어까지 다시 세운다 |
| `watch` | 볼트를 감시하며 저장될 때마다 잎을 갱신한다. `index`가 도는 동안은 모아뒀다가 끝난 뒤 적용한다 |
| `ask` | 적재된 내용에 질문하고 근거 노트를 함께 보여준다 |
| `status` | 적재 현황, 요약 레이어가 낡은 정도, 빌드가 돌고 있는지 보여준다 |
| `eval` | 질문 모음으로 검색 성적을 잰다. `--build`로 질문을 새로 만든다 |

공통 옵션은 `--collection`과 `--llm`이다. `index`, `sync`, `watch`는 `--vault`로 경로를 바꾼다.

LLM이 필요한 명령은 `index`, `ask`, `sync --rebuild-tree`뿐이다. `status`와 잎만 갈아끼우는 `sync`·`watch`는 모델을 한 번도 부르지 않으니 Ollama가 꺼져 있어도, API 키가 없어도 돈다. 임베딩 가중치도 실제로 쓸 때 올린다.

## 쓰는 법

**처음 한 번.** 볼트 전체로 트리를 세운다. 노트 130개 남짓이면 임베딩에 30분, 요약에 10분쯤 걸린다.

```bash
uv run raptor-qdrant index --allow-remote-llm
```

도는 동안 진행은 로그로 본다. 다른 터미널에서 `tail -f ~/.local/state/raptor-qdrant/raptor-qdrant.log`를 걸어두면 된다. 중간에 죽어도 그때까지 적재된 레이어는 남는다. 다음 `index`가 끝나지 않은 세대를 치우고 새로 시작한다.

**평소에는 묻기만 한다.**

```bash
uv run raptor-qdrant ask "인천 IDC 관련해 정리해둔 게 뭐가 있나"
uv run raptor-qdrant ask "Consul은 어떻게 배포했나" --show-chunks 5
```

`--show-chunks`를 주면 어느 레이어에서 몇 점으로 걸렸는지 보인다. 답이 이상할 때 검색이 잘못 물어온 것인지 모델이 잘못 읽은 것인지 가르는 데 쓴다. 답변 아래의 근거 노트는 관련도 순서다.

**노트를 고치면 잎만 갈아끼운다.** 트리를 다시 세우는 데는 LLM 요약이 붙어 오래 걸리니 저장할 때마다 그걸 돌리지 않는다.

```bash
uv run raptor-qdrant sync --dry-run   # 무엇이 바뀌었는지만
uv run raptor-qdrant sync             # 반영
uv run raptor-qdrant watch            # 저장될 때마다 자동으로
```

`watch`는 띄워두면 된다. 잎만 건드리므로 LLM도 API 키도 필요 없다.

**가끔 상태를 본다.**

```bash
uv run raptor-qdrant status
```

"현재 트리 밖의 잎"이 20%를 넘으면 요약 레이어가 그만큼 낡았다는 뜻이다. 그때 `sync --rebuild-tree`로 트리를 다시 세운다. 한 달에 한 번쯤이면 충분하다.

**설정을 바꿨으면 재본다.** 청킹이나 임베딩을 건드렸으면 재인덱싱이 필요하지만, 검색 쪽 설정은 바로 잴 수 있다.

```bash
uv run raptor-qdrant eval
```

### 어떤 명령에 무엇이 필요한가

| 명령 | Qdrant | LLM | 임베딩 모델 |
|---|---|---|---|
| `status` | 필요 | 안 씀 | 안 올림 |
| `sync --dry-run` | 필요 | 안 씀 | 안 올림 |
| `sync`, `watch` | 필요 | 안 씀 | 올림 |
| `eval` | 필요 | `--build` 일 때만 | 올림 |
| `ask` | 필요 | 필요 | 올림 |
| `index`, `sync --rebuild-tree` | 필요 | 필요 | 올림 |

Ollama가 꺼져 있거나 API 키가 없어도 위쪽 네 줄은 돈다.

## 설정

`.env` 파일이나 환경변수로 덮어쓴다. 둘 다 있으면 환경변수가 이긴다.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `QDRANT_HOST` | `localhost` | Qdrant 호스트 |
| `QDRANT_PORT` | `6333` | Qdrant 포트 |
| `VAULT_PATH` | `~/Documents/Obsidian Vault` | 옵시디언 볼트 경로 |
| `COLLECTION_NAME` | `obsidian` | 기본 Qdrant 컬렉션 |
| `STATE_DIR` | `~/.local/state/raptor-qdrant` | 빌드 락 같은 실행 상태를 두는 곳 |
| `EMBEDDING_MODEL` | `nlpai-lab/KURE-v1` | SentenceTransformer 모델 (항상 로컬) |
| `EMBEDDING_DEVICE` | (비움) | 비우면 자동. mac에서는 `cpu`를 고른다 |
| `RERANKER_MODEL` | (비움) | 후보를 다시 줄 세울 크로스 인코더. 비우면 안 한다 |
| `RERANK_CANDIDATES` | `0` | 다시 세울 후보 수. 0이면 전부 |
| `LLM_PROVIDER` | `ollama` | `ollama` · `anthropic`(=`claude`) · `openai`(=`chatgpt`) · `bedrock` |
| `SUMMARY_MODEL` | (비움) | 요약 전용 모델. 비우면 답변 모델을 그대로 쓴다 |
| `SUMMARY_WORKERS` | `0` | 클러스터 요약 동시 실행 수. 0이면 공급자 기본값(ollama 2, bedrock 10, claude·chatgpt 8) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama 모델 태그 |
| `ANTHROPIC_API_KEY` | (비움) | 비우면 SDK가 환경변수와 `ant auth login` 프로필을 차례로 본다 |
| `ANTHROPIC_OAUTH_TOKEN_FILE` | (비움) | OAuth 토큰(`sk-ant-oat...`)이 든 파일 경로. 토큰 값이 아니라 경로만 둔다 |
| `ANTHROPIC_MODEL` | `claude-opus-5` | Claude 모델 |
| `ANTHROPIC_EFFORT` | `medium` | 사고 깊이와 토큰 지출. `low`~`max`. effort를 받지 않는 구형 모델은 비운다 |
| `OPENAI_API_KEY` | (비움) | 비우면 SDK가 `OPENAI_API_KEY` 환경변수를 본다 |
| `OPENAI_MODEL` | `gpt-4.1-mini` | ChatGPT 모델. 추론 모델(`gpt-5`, `o3` 등)이면 temperature를 빼고 보낸다 |
| `BEDROCK_MODEL_ID` | `apac.anthropic.claude-3-7-sonnet-20250219-v1:0` | Bedrock 모델 |
| `AWS_REGION` | `ap-northeast-2` | Bedrock 리전 |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` / `critical` |
| `LOG_FILE` | `~/.local/state/raptor-qdrant/raptor-qdrant.log` | 진행 로그 파일. 10MB 회전, 5개 보관. 비우면 stdout만 |
| `ENVIRONMENT` | `local` | `production`이면 JSON 구조화 로그로 전환 |

검색·청킹·트리 관련 수치는 `src/raptor_qdrant/rag/constants.py`에 모여 있다. 청크 최대 토큰(512), 컨텍스트 예산(8192), top-k(12), 요약 쿼터(4), 원문 확장(6), 하이브리드 가중치(1.0), 임베딩 배치(64) 같은 값들이다.

인덱싱 시간의 대부분은 요약이다. Ollama 서버는 `OLLAMA_NUM_PARALLEL`(기본 1)만큼만 동시에 받으므로 `SUMMARY_WORKERS`를 올리려면 서버도 같이 올려야 하고, 슬롯마다 KV 캐시가 붙어 메모리를 더 쓴다. 메모리가 빠듯하면 `SUMMARY_MODEL`에 더 작은 모델(예: `qwen2.5:3b`, `claude-haiku-4-5`)을 두는 쪽이 낫다. 중간 요약은 검색 앵커 역할이라 답변 모델만큼 클 필요가 없다. 돈이 나가는 공급자라면 더 그렇다.

## 검색 성적 재기

무엇을 바꾸면 검색이 나아지는지는 재봐야 안다. `eval`이 그 자다.

```bash
uv run raptor-qdrant eval --build --allow-remote-llm           # 노트에서 질문 25개
uv run raptor-qdrant eval --build --broad --allow-remote-llm   # 요약에서 가로지르는 질문
uv run raptor-qdrant eval                                      # 재기만 (LLM 안 쓴다)
```

`--build`는 노트를 골라 "몇 달 뒤 이 노트를 찾으려 할 때 던질 법한 질문"을 하나씩 만들고 그 노트를 정답으로 기록한다. 사람이 라벨을 붙이지 않아도 기준이 생긴다. 완벽한 정답지는 아니다. 같은 주제를 다룬 다른 노트가 나와도 틀렸다고 세므로 점수가 실제보다 박하다. 대신 같은 자로 재기 때문에 변경 전후를 견줄 수 있다.

노트 하나로 만든 질문은 그 노트의 어휘를 그대로 물고 있어서 검색이 거의 틀리지 않는다. 실제로 첫 측정에서 recall이 100%, MRR이 0.967로 천장에 닿았다. 그 자로는 개선을 잴 수 없다. `--broad`는 요약 노드에서 질문을 만들고 그 요약이 덮는 노트 전부를 정답으로 삼는다. 트리가 실제로 쓰이는지를 재려면 이쪽이 필요하다.

점수는 여러 개를 같이 본다. recall@1과 @3이 함께 있어야 자가 포화됐는지 보인다. coverage는 묶음 질문에서 정답 노트를 몇 개나 덮었는지고, 괄호 안의 원문 coverage는 그중 실제 원문으로 온 비율이다. 이 둘은 갈라서 봐야 한다. 요약은 덮는 노트를 전부 나열하므로 이름만 스쳐도 coverage가 오른다. 실제로 coverage 90%인 상태에서 원문 coverage는 34%였다. 예산 사용률은 컨텍스트를 놀리는지 알려준다.

**질문은 저장소에 두지 않는다.** 볼트 내용에서 나온 것이라 함께 공개된다. `STATE_DIR/eval/<컬렉션>.jsonl`에 쌓이고 저장소에는 도구만 있다.

## 볼트 파싱

로더가 노트마다 하는 일은 이렇다.

- YAML frontmatter를 본문에서 떼어내 태그로 옮긴다. 남겨두면 모든 청크 앞머리가 YAML이라 임베딩이 오염된다
- `[[위키링크]]`는 표시 텍스트만 본문에 남기고 대상은 따로 기록한다
- `![[이미지]]` 임베드는 버린다
- `#태그`를 본문에서 모은다
- 원본 파일의 SHA-256을 남긴다. `sync`가 이 값으로 변경을 판단한다
- `.obsidian`, `.trash`, `.git`, `node_modules`와 빈 노트는 건너뛴다

## 증분 동기화

`sync`는 볼트의 해시와 적재된 해시를 비교해 추가·변경·삭제를 가른다. 달라진 게 없으면 아무 일도 하지 않는다.

달라졌으면 바뀐 노트의 잎만 갈아끼운다. 트리를 다시 세우는 데는 LLM 요약이 붙어 몇십 분에서 몇 시간이 걸리니 저장할 때마다 그걸 돌릴 수는 없다. 트리 간선은 적재되지 않으므로 잎만 바꿔도 기존 요약 노드는 깨지지 않고, 내용만 그만큼 낡는다. 얼마나 낡았는지는 `tree_generation`으로 센다. 현재 트리 밖의 잎이 20%를 넘으면 `status`와 `sync`가 `--rebuild-tree`를 권한다.

노트를 지우면 그 노트를 덮던 요약도 함께 검색에서 뺀다. 요약은 원문이 사라져도 그 내용을 그대로 물고 있다. 낡은 것과 없는 것은 다르다. 재구축 전까지는 그 요약이 없는 노트를 근거로 내놓지 않게 표시해 둔다. 몇 개가 빠졌는지는 `status`에 나온다.

`watch`는 같은 일을 저장 시점마다 한다. 옵시디언이 타이핑 중에도 자주 저장하므로 파일별로 3초 조용해질 때까지 기다렸다가 한 번만 반영한다.

### 재구축과 동시에 돌 때

`index`는 시작 시점의 스냅샷으로 트리를 세운다. 그 사이 볼트가 바뀌어도 트리에는 없다. 그래서 `index`는 컬렉션마다 락(`STATE_DIR/<컬렉션>.build.json`)을 잡고 돌고, 끝나면 스냅샷과 지금 볼트를 비교해 그 사이 바뀐 노트만 잎으로 넣는다. 락이 잡혀 있으면 `sync`는 거절하고, `watch`는 변경을 모아뒀다가 락이 풀리면 한 번에 적용한다. 락의 pid가 죽어 있으면 끝나지 않은 빌드의 흔적이다. 다음 `index`가 그 세대의 포인트를 치우고 시작한다.

## 컬렉션과 문서

`collection_name`은 도메인이나 카테고리 단위, `document_name`은 그 안의 개별 문서 단위다. 한 컬렉션에 여러 문서를 담는 것이 기본 사용 방식이라, `add_document()`는 컬렉션을 지우지 않는다.

```python
engine.list_documents()  # 이 컬렉션의 문서명 목록
engine.update_document(text, "report")  # 기존 문서 교체
engine.delete_document("report")  # 문서 단위 삭제
engine.indexed_content_hashes()  # 증분 판단용 해시
```

같은 `document_name`으로 두 번 `add_document()`를 부르면 중복 적재 대신 예외가 난다. 교체할 생각이었다면 `update_document()`를 쓰라는 뜻이다.

## 개발

품질 검사는 CI가 도는 것과 같다. 테스트는 Qdrant나 LLM 없이 순수 로직만 본다.

```bash
uv run pytest              # 테스트
uv run ruff check .        # 린트
uv run ruff format .       # 포맷
uv run mypy                # 타입 검사
```

테스트가 지키는 것은 이렇다.

| 파일 | 무엇을 지키는가 |
|---|---|
| `test_vault_loader.py` | frontmatter 제거, 위키링크 평탄화, 해시 기반 증분 판단 |
| `test_cli_helpers.py` | 원격 LLM 가드, 근거 노트 중복 제거, 빌드 중 바뀐 노트 따라잡기 |
| `test_cluster_layers.py` | 클러스터마다 요약은 한 번, 실패한 것만 재시도, 레이어별 배치 임베딩 |
| `test_node_embedding.py` | 잎 임베딩이 배치로 나가고 이미 있는 임베딩은 다시 계산하지 않는다 |
| `test_build_lock.py` | 살아 있는 빌드는 막고 죽은 빌드의 흔적은 넘겨준다 |
| `test_index_health.py` | 낡은 비율과 세대 섞임 판정 |
| `test_log_file.py` | 표준 logging이 파일 sink까지 닿는다 |
| `test_chunk_tagging.py` | 청크마다 토큰 수를 따로 센다 (부모 값을 물려받으면 검색이 죽는다) |
| `test_context_window.py` | 큰 노드 하나가 컨텍스트 전체를 비우지 않는다 |
| `test_token_count.py` | `token_count`가 없거나 망가져도 터지지 않는다 |
| `test_summary_sentinel.py` | 모델이 장식을 붙인 `NO_SUMMARY`도 걸러낸다 |
| `test_chunk_metadata.py` | payload가 JSON으로 직렬화된다 |
| `test_tree_structure.py` | 노드와 레이어 매핑 |
| `test_clustering_utils.py` | 데이터가 적을 때의 클러스터 수 경계 |
| `test_llm_factory.py` | 공급자 선택·별칭·원격 판정, 지연 생성, 오류 메시지 |
| `test_llm_requests.py` | Claude·ChatGPT가 보내는 요청 모양 (effort·temperature·거절 처리·OAuth) |
| `test_layer_mix.py` | 상위 k 안에 요약 자리를 남기되 관련도 순서를 지킨다 |
| `test_reranker.py` | 재순위화가 순서와 점수를 바꾸고 채점 상한을 지킨다 |
| `test_chunk_merging.py` | 작은 조각을 이웃에 붙이고 큰 조각은 상한 안으로 자른다 |
| `test_eval_report.py` | recall·MRR·coverage 집계와 정답 판정 |
| `test_payload_scan.py` | payload 훑기가 필요한 필드만 받아 온다 |

## 구조

```
src/raptor_qdrant/
├── cli.py                     콘솔 스크립트 진입점
├── vault/                     옵시디언 볼트 로더, 증분 비교, 워처, 빌드 락
├── core/
│   ├── config.py              pydantic-settings 기반 환경 설정
│   └── logger.py              표준 logging을 loguru로 넘기고 KST로 stdout·파일에 출력
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
    ├── eval/                  검색 성적 측정 (질문 생성, 집계)
    ├── llm/                   LLM 공급자 (base·ollama·claude·chatgpt·bedrock + 팩토리)
    │                          공급자는 complete만 구현하고, 답변 템플릿은 base가 씌운다
    └── prompt/                요약·답변 프롬프트
tests/                         Qdrant·LLM 없이 도는 단위 테스트
```

임베딩 모델, 요약 모델, LLM, 청커, 클러스터링 알고리즘은 모두 추상 기반 클래스를 두고 `EngineConfig`로 주입한다. 다른 구현으로 갈아끼우려면 해당 기반 클래스만 상속하면 된다. LLM을 하나 더 붙일 때는 `BaseChatbotModel`을 상속해 `rag/llm/`에 두고 팩토리의 표에 한 줄 더한다.

## 참고

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [KURE-v1](https://huggingface.co/nlpai-lab/KURE-v1): 한국어 검색 특화 임베딩 모델
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)
