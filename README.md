# bge_test

`data/manual.txt`를 입력으로 사용해 로컬 임베딩 모델과 OpenAI 임베딩을 비교 벤치마크하고, 결과를 실행 시각 폴더(`reports/YYYYMMDD_HHMMSS/`)에 생성하는 프로젝트입니다.

## 포함 파일

- `main.py`: 벤치마크 실행 진입점
- `config.py`: 모델/청크/쿼리 등 전역 설정
- `providers.py`: 임베딩 Provider 로딩 로직
- `evaluator.py`: 인덱싱/검색/VRAM 측정 로직
- `reporter.py`: 마크다운 리포트/시각화 확장 포인트
- `data/manual.txt`: 벤치마크 입력 문서
- `reports/`: 실행 결과 리포트/차트 보관 폴더(자동 생성)
- `.env.example`: 환경변수 템플릿

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

`.env.example`를 참고해 키를 준비합니다.

```env
OPENAI_API_KEY=
```

이 스크립트는 실행 시 프로젝트 루트의 `.env` 파일을 자동으로 읽습니다.
셸 환경변수로 직접 설정해도 되며, 이미 설정된 값이 있으면 해당 값을 우선 사용합니다.

cmd 예시:

```bat
set OPENAI_API_KEY=sk-...
```

## 실행 방법

프로젝트 루트에서 실행:

```bash
python main.py
```

## 동작 개요

- 입력: `data/manual.txt`
- 비교 모델
  - Local 기준 BGE-M3 (`BAAI/bge-m3`)
  - Local E5-large (`intfloat/multilingual-e5-large`)
  - Local 경량 Ko-SBERT (`jhgan/ko-sroberta-multitask`)
  - API OpenAI small (`text-embedding-3-small`)
- Chunk size: `300`, `600`, `1000` (overlap `50`)
- 지표: 로드 시간, 인덱싱 시간, 평균 검색 시간, VRAM 피크, 검색 품질(Top-3), 추정 비용

## 결과 파일

실행이 완료되면 아래 폴더가 새로 생성됩니다.

- `reports/YYYYMMDD_HHMMSS/embedding_benchmark_report.md`
- `reports/YYYYMMDD_HHMMSS/benchmark_visualization.png`

## 참고 사항

- `OPENAI_API_KEY`가 없으면 OpenAI 케이스는 자동으로 건너뜁니다.
- `.gitignore`에 `.env`, `venv/`, `__pycache__/`, `*.pyc`가 제외되어 있습니다.
- 기존 루트 산출물(`embedding_benchmark_report.md`, `benchmark_visualization.png`)이 있으면 최초 실행 시 `reports/_legacy/`로 자동 이동되어 보존됩니다.
