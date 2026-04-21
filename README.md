# bge_test

`data/` 폴더의 PDF 코퍼스를 대상으로 로컬 임베딩 모델을 비교하는 벤치마크/질의 서버입니다.

## 포함 파일

- `main.py`: 벤치마크 실행 진입점
- `server.py`: `uvicorn + FastAPI` 서버 모드
- `build_chromaDB.py`: 모델/청크별 ChromaDB 사전 빌드
- `runtime_store.py`: on-demand 모델 로드 + LRU 캐시 + vectorstore 캐시
- `config.py`: 모델/청크/쿼리 등 전역 설정
- `providers.py`: 임베딩 모델 로딩/언로딩
- `evaluator.py`: PDF 로드/청킹/유사도 유틸
- `reporter.py`: 마크다운 리포트 생성
- `reports/`: 실행 결과 리포트 보관 폴더(자동 생성)

## 사전 요구사항

- Python 3.10+
- (로컬 GPU 모델 테스트 시) CUDA 사용 가능한 PyTorch 환경

권장 설치 절차(Windows cmd):

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

패키지만 설치할 경우:

```bash
pip install -r requirements.txt
```

## 환경변수 설정

- `EMBED_BATCH_SIZE` (기본 `8`): 임베딩 배치 크기
- `MODEL_CACHE_SIZE` (기본 `1`): 서버 모델 LRU 캐시 크기
- `FORCE_CPU_MODELS` (선택): CPU 강제 모델 ID comma-separated
  - 예: `set FORCE_CPU_MODELS=Alibaba-NLP/gte-multilingual-base`

## 실행 방법

### 1) CLI 벤치마크

```bash
python main.py
```

실행 후 `reports/YYYYMMDD_HHMMSS/embedding_benchmark_report.md` 생성.

### 2) 서버 모드 (`uvicorn + FastAPI`)

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3) ChromaDB 사전 빌드 (권장)

```bash
python build_chromaDB.py
```

옵션:

```bash
python build_chromaDB.py --rebuild
python build_chromaDB.py --models BAAI/bge-m3 intfloat/multilingual-e5-base --chunk-sizes 600 1000
```

저장 위치:
- `data/chroma_db/<model_slug>/chunk_<size>/`
- 컬렉션명: `bench_<model_slug>_<size>` (모델명 포함)

- `GET /health`: 하드웨어/캐시 상태
- `GET /models`: 모델 목록 + chunk size + 현재 캐시 상태
- `POST /query`: 단일 모델/청크 질의
- `POST /query_all`: 전체 모델 순차 질의

`curl` 예시(Windows):

```bat
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"model_id\":\"BAAI/bge-m3\",\"chunk_size\":600,\"question\":\"키 카드를 들고 멀어지면 자동으로 잠기나?\",\"k\":3}"
```

전체 모델 순차 질의:

```bat
curl -X POST "http://127.0.0.1:8000/query_all" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"키 카드를 들고 멀어지면 자동으로 잠기나?\",\"chunk_size\":600,\"k\":3}"
```

터미널 입력형 테스트:

```bash
python interactive_query.py
```

## 동작 특성

- 모델 전부 선로드 대신 **on-demand 로드 + LRU 캐시**(기본 1개)로 VRAM 안정성 확보
- 같은 `(model_id, chunk_size)` 반복 질의는 vectorstore를 재사용하여 PDF 재로딩/재청킹/재인덱싱 최소화
- 실패 케이스도 누락하지 않고 리포트와 API 응답에 `status=failed`, `error_message` 기록
