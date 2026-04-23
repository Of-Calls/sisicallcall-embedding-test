#!/usr/bin/env python3
"""
sisicallcall — 임베딩 모델 RAG/Semantic Cache 벤치마크
로컬(GPU) 임베딩을 ChromaDB + LangChain 파이프라인으로 비교합니다.

입력은 `data/` 디렉터리의 복수 PDF입니다.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import (
    DEFAULT_CHUNK_KEY,
    DOCS_PDF_DIR,
    MODELS_TO_TEST,
    REPORTS_ROOT,
    TEST_QUERIES,
    CaseMetrics,
)
from evaluator import percentile, print_chunking_summary, print_retrieval_sanity_block
from reporter import create_report_output_dir, format_load_time, format_vram, save_visualization_charts, write_report
from runtime_store import RuntimeStore


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("embedding_benchmark")


def load_dotenv_file(dotenv_path: Path = Path(".env")) -> None:
    """간단한 .env 로더. 이미 설정된 환경변수는 덮어쓰지 않습니다."""
    import os

    if not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def migrate_legacy_report_if_exists() -> None:
    """루트에 있던 기존 산출물을 reports/_legacy로 이동해 보존합니다."""
    legacy_files = [
        Path("embedding_benchmark_report.md"),
        Path("benchmark_visualization.png"),
    ]
    existing = [p for p in legacy_files if p.is_file()]
    if not existing:
        return

    legacy_dir = REPORTS_ROOT / "_legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for src in existing:
        target = legacy_dir / src.name
        if target.exists():
            stem = src.stem
            suffix = src.suffix
            idx = 1
            while (legacy_dir / f"{stem}_{idx}{suffix}").exists():
                idx += 1
            target = legacy_dir / f"{stem}_{idx}{suffix}"
        shutil.move(str(src), str(target))
        log.info("기존 산출물 보관: %s", target)


def main() -> None:
    setup_logging()
    load_dotenv_file()
    print("=" * 60)
    print(" sisicallcall — Embedding + Chroma 벤치마크")
    print("=" * 60)

    migrate_legacy_report_if_exists()

    all_metrics = []
    t_start = time.perf_counter()
    max_cache = int(os.environ.get("MODEL_CACHE_SIZE", "1"))
    try:
        store = RuntimeStore(pdf_dir=DOCS_PDF_DIR, max_models_in_cache=max_cache)
    except Exception as e:
        log.error("런타임 초기화 실패: %s", e)
        sys.exit(1)

    hw = store.hardware_note
    log.info("환경: %s", hw)
    log.info("모델 캐시 정책: LRU 최대 %s개", max_cache)
    log.info("Semantic Chunking E2E / chunk_key=%s", DEFAULT_CHUNK_KEY)

    chunk_key = DEFAULT_CHUNK_KEY
    total_cases = len(MODELS_TO_TEST)
    outer = tqdm(total=total_cases, desc="전체 케이스", unit="case")

    try:
        for spec in MODELS_TO_TEST:
            outer.set_postfix(model=spec.label[:24])
            load_time = None
            index_time = 0.0
            timings: list[float] = []
            query_results: dict[str, list[dict[str, object]]] = {}
            query_latency_ms: dict[str, float] = {}
            status = "ok"
            error_type = None
            error_message = None

            for qid, qtext in TEST_QUERIES:
                res = store.query(spec=spec, chunk_key=chunk_key, question=qtext, k=3)
                if res.get("status") != "ok":
                    status = "failed"
                    error_type = str(res.get("error_type"))
                    error_message = str(res.get("error_message"))
                    break
                if load_time is None:
                    load_time = float(res.get("load_time_sec", 0.0))
                index_time = max(index_time, float(res.get("index_time_sec", 0.0)))
                timings.append(float(res.get("latency_ms", 0.0)) / 1000.0)
                query_latency_ms[qid] = float(res.get("latency_ms", 0.0))
                rows = list(res.get("results", []))
                query_results[qid] = rows  # type: ignore[assignment]
                print_retrieval_sanity_block(
                    model_label=spec.label,
                    chunk_key=chunk_key,
                    question=qtext,
                    rows=rows,  # type: ignore[arg-type]
                )

            if load_time is None:
                load_time = 0.0
            retrieval_avg = sum(timings) / len(timings) if timings else 0.0
            retrieval_p95 = percentile(timings, 95) if timings else 0.0
            docs_count = store.count_docs_cached(spec, chunk_key)
            tc, avg_l, max_l, min_l = store.chunk_length_stats(spec, chunk_key)
            throughput = (docs_count / index_time) if index_time > 0 else 0.0
            index_size_mb = 0.0
            key = (spec.model_id, chunk_key)
            if key in store.vector_cache:
                try:
                    p = Path(store.vector_cache[key].persist_dir)
                    index_size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
                except Exception:
                    index_size_mb = 0.0
            metric = CaseMetrics(
                model_label=spec.label,
                model_id=spec.model_id,
                load_time_sec=load_time,
                index_time_sec=index_time,
                retrieval_avg_sec=retrieval_avg,
                retrieval_p95_sec=retrieval_p95,
                vram_peak_mb=None,
                index_size_mb=index_size_mb,
                embedding_docs_per_sec=throughput,
                total_chunks=tc if status == "ok" else 0,
                avg_chunk_length=avg_l if status == "ok" else 0.0,
                max_chunk_length=max_l if status == "ok" else 0,
                min_chunk_length=min_l if status == "ok" else 0,
                status=status,
                error_type=error_type,
                error_message=error_message,
                query_results=query_results,  # type: ignore[arg-type]
                query_latency_ms=query_latency_ms,
            )
            outer.update(1)
            all_metrics.append(metric)
            if status == "failed":
                log.error("[%s] 실패: %s", spec.label, error_message)
            else:
                print_chunking_summary(metric.total_chunks, metric.avg_chunk_length, metric.max_chunk_length)
                log.info(
                    "[%s] 로드=%s 인덱싱=%.2fs 검색평균=%.4fs VRAM=%s",
                    spec.label,
                    format_load_time(metric),
                    metric.index_time_sec,
                    metric.retrieval_avg_sec,
                    format_vram(metric),
                )
    finally:
        store.close()

    duration = time.perf_counter() - t_start

    output_dir = create_report_output_dir()
    report_path = write_report(
        all_metrics,
        hardware_note=hw,
        duration_sec=duration,
        output_dir=output_dir,
    )
    chart_path = save_visualization_charts(all_metrics, output_dir)
    print(f"\n완료. 리포트: {report_path}")
    print(f"산출물 폴더: {output_dir}")
    if chart_path is not None:
        print(f"차트: {chart_path}")


if __name__ == "__main__":
    main()
