from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MANUAL_PATH = SCRIPT_DIR / "data" / "manual.txt"
REPORTS_ROOT = SCRIPT_DIR / "reports"
REPORT_FILENAME = "embedding_benchmark_report.md"
CHART_FILENAME = "benchmark_visualization.png"

CHUNK_OVERLAP = 50
CHUNK_SIZES = [300, 600, 1000]

# OpenAI text-embedding-3-small: 공식 요금(2024~ 기준, 변동 가능) — 리포트에 명시
OPENAI_EMBEDDING_PRICE_PER_1M_USD = float(
    os.environ.get("OPENAI_EMBEDDING_PRICE_PER_1M", "0.02")
)

# `data/manual.txt`(비전클라우드 매뉴얼) 기준 검증용 질문·키워드
TEST_QUERIES: list[tuple[str, str, list[str]]] = [
    (
        "Q1",
        "일반 기술 지원은 몇 시부터 가능한가요? 시스템 장애 시 연락처도 알려주세요.",
        ["오전 10시", "오후 5시", "평일", "tech@visioncloud.com", "1588-1111"],
    ),
    (
        "Q2",
        "연간 계약 중도 해지 시 위약금은 어떻게 계산되고, 월정액 환불 조건은 무엇인가요?",
        ["위약금", "환불", "30%", "7일"],
    ),
    (
        "Q3",
        "엔지니어 방문 예약을 취소하면 수수료나 페널티가 붙나요?",
        ["48시간", "10만 원", "페널티", "예약 취소"],
    ),
]


@dataclass
class CaseMetrics:
    model_label: str
    model_id: str
    chunk_size: int
    load_time_sec: float | None  # API는 None
    index_time_sec: float
    retrieval_avg_sec: float
    retrieval_p95_sec: float
    vram_peak_mb: float | None
    index_size_mb: float
    embedding_docs_per_sec: float
    hit_at_3: float
    mrr_at_3: float
    token_count: int | None
    estimated_cost_usd: str
    query_results: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


@dataclass
class ModelSpec:
    label: str
    model_id: str
    kind: str  # "local" | "openai"
    use_e5_prefix: bool = False


MODELS: list[ModelSpec] = [
    ModelSpec("Local 기준 BGE-M3", "BAAI/bge-m3", "local"),
    ModelSpec("Local E5-large", "intfloat/multilingual-e5-large", "local", use_e5_prefix=True),
    ModelSpec("Local 경량 Ko-SBERT", "jhgan/ko-sroberta-multitask", "local"),
    ModelSpec("API OpenAI small", "text-embedding-3-small", "openai"),
]

CHROMA_COSINE_CONFIG: dict[str, Any] = {"hnsw": {"space": "cosine"}}
