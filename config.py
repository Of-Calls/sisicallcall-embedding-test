from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DOCS_PDF_DIR = SCRIPT_DIR / "data"
REPORTS_ROOT = SCRIPT_DIR / "reports"
REPORT_FILENAME = "embedding_benchmark_report.md"
CHART_FILENAME = "benchmark_visualization.png"

CHUNK_OVERLAP = 50

# 시멘틱 청킹만 사용 (SemanticChunker, percentile breakpoint).
# 프로필 ID와 percentile — 같은 질문 세트로 모델·민감도 비교 시 사용.
SEMANTIC_CHUNK_PROFILES: list[tuple[str, float]] = [
    ("p85", 85.0),
    ("p92", 92.0),
    ("p97", 97.0),
]

ChunkKey = str


def allowed_chunk_keys() -> list[str]:
    return [pid for pid, _ in SEMANTIC_CHUNK_PROFILES]


def chunk_folder_segment(key: str) -> str:
    """persist 폴더/컬렉션 접미사용 (예: sem_p92)."""
    return f"sem_{key}"


def semantic_percentile(profile_id: str) -> float:
    for pid, pct in SEMANTIC_CHUNK_PROFILES:
        if pid == profile_id:
            return pct
    raise KeyError(f"unknown semantic profile: {profile_id}")


def parse_chunk_key(raw: str | int) -> str:
    s = str(raw).strip()
    if not s:
        raise ValueError("chunk_key가 비었습니다.")
    return s


def normalize_chunk_key(raw: str | int) -> str:
    return str(raw).strip()


def is_valid_chunk_key(key: str | int) -> bool:
    return normalize_chunk_key(key) in allowed_chunk_keys()

# TEST_QUERIES: (질문ID, 질문내용) 형식만 사용합니다.
TEST_QUERIES: list[tuple[str, str]] = [
    ("DJI-01", "앱에 로그인하지 않으면 비행 고도와 거리 제한이 어떻게 되나?"),
    ("DJI-02", "기체를 처음 쓸 때 전원을 켜기 전에 뭘 먼저 해야 하나?"),
    ("DJI-03", "드론 비행 전에 꼭 확인해야 하는 체크리스트는 뭐야?"),
    ("DJI-04", "GNSS 신호가 약하면 비행 거리도 제한되나?"),
    ("DJI-05", "FocusTrack에서 스포츠 모드일 때 장애물 회피가 되나?"),
    ("DJI-06", "ActiveTrack에서 사람을 추적할 때 가능한 수평 거리 범위는?"),
    ("TESLA-01", "터치스크린이 멈추면 어떻게 다시 시작하지?"),
    ("TESLA-02", "음성으로 글로브박스 열 수 있어?"),
    ("TESLA-03", "키 카드를 들고 멀어지면 자동으로 잠기나?"),
    ("TESLA-04", "워크어웨이 도어 잠금이 안 되는 경우는 언제야?"),
    ("TESLA-05", "전면 트렁크는 최대 몇 kg까지 실을 수 있어?"),
    ("TESLA-06", "후면 트렁크 열림 높이를 저장하려면 어떻게 해야 해?"),
    ("TESLA-07", "USB 포트 중에서 감시 모드나 블랙박스 저장에 적합한 건 어디야?"),
    ("TESLA-08", "카메라 보정은 얼마나 달리면 끝나?"),
]


@dataclass
class CaseMetrics:
    model_label: str
    model_id: str
    chunk_key: str
    load_time_sec: float | None
    index_time_sec: float
    retrieval_avg_sec: float
    retrieval_p95_sec: float
    vram_peak_mb: float | None
    index_size_mb: float
    embedding_docs_per_sec: float
    status: str = "ok"
    error_type: str | None = None
    error_message: str | None = None
    query_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    query_latency_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelSpec:
    label: str
    model_id: str
    use_e5_prefix: bool = False


MODELS_TO_TEST: list[ModelSpec] = [
    ModelSpec("E5-large (기준)", "intfloat/multilingual-e5-large", use_e5_prefix=True),
    ModelSpec("BGE-M3 (기준)", "BAAI/bge-m3"),
    ModelSpec("E5-base (경량)", "intfloat/multilingual-e5-base", use_e5_prefix=True),
    ModelSpec("Jina v2-base-en (안정)", "jinaai/jina-embeddings-v2-base-en"),
]

CHROMA_COSINE_CONFIG: dict[str, Any] = {"hnsw": {"space": "cosine"}}
