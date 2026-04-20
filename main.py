#!/usr/bin/env python3
"""
sisicallcall — 임베딩 모델 RAG/Semantic Cache 벤치마크
로컬(GPU) 및 OpenAI 임베딩을 ChromaDB + LangChain 파이프라인으로 비교합니다.

입력 텍스트는 `data/manual.txt`를 사용합니다.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import CHUNK_SIZES, MANUAL_PATH, MODELS, REPORTS_ROOT, TEST_QUERIES
from evaluator import load_manual, run_case_safe
from providers import build_local_embeddings, build_openai_embeddings, detect_hardware_note
from reporter import create_report_output_dir, format_load_time, format_vram, write_report


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("embedding_benchmark")


def load_dotenv_file(dotenv_path: Path = Path(".env")) -> None:
    """간단한 .env 로더. 이미 설정된 환경변수는 덮어쓰지 않습니다."""
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

    if not os.environ.get("OPENAI_API_KEY"):
        log.warning(
            "OPENAI_API_KEY 가 없습니다. OpenAI 임베딩 케이스는 스킵됩니다. "
            "(환경변수 설정 후 재실행하세요)"
        )

    hw = detect_hardware_note()
    log.info("환경: %s", hw)
    migrate_legacy_report_if_exists()

    try:
        full_text = load_manual(MANUAL_PATH)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("입력 매뉴얼: %s (%d chars)", MANUAL_PATH.resolve(), len(full_text))

    all_metrics = []
    skipped_api = False
    t_start = time.perf_counter()

    runnable_specs = []
    for spec in MODELS:
        if spec.kind == "openai" and not os.environ.get("OPENAI_API_KEY"):
            skipped_api = True
            continue
        runnable_specs.append(spec)

    total_cases = len(runnable_specs) * len(CHUNK_SIZES)
    outer = tqdm(total=total_cases, desc="전체 케이스", unit="case")

    for spec in runnable_specs:
        shared_embedder = None
        cleanup = lambda: None
        shared_load_time: float | None = None
        first_chunk = True

        try:
            if spec.kind == "local":
                t0 = time.perf_counter()
                shared_embedder, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
                shared_load_time = time.perf_counter() - t0
                log.info("[%s] 모델 로드 완료: %.2fs", spec.label, shared_load_time)
            else:
                shared_embedder = build_openai_embeddings(spec.model_id)
                if shared_embedder is None:
                    skipped_api = True
                    for _ in CHUNK_SIZES:
                        outer.update(1)
                    continue

            for chunk_size in CHUNK_SIZES:
                outer.set_postfix(model=spec.label[:24], chunk=chunk_size)
                metric, err = run_case_safe(
                    full_text,
                    spec,
                    chunk_size,
                    TEST_QUERIES,
                    embedder=shared_embedder,
                    load_time_override=shared_load_time if first_chunk else 0.0,
                    owns_embedder=False,
                )
                outer.update(1)
                first_chunk = False

                if err:
                    log.error("[%s cs=%s] 실패: %s", spec.label, chunk_size, err)
                    continue
                if metric:
                    all_metrics.append(metric)
                    log.info(
                        "[%s chunk=%s] 로드=%s 인덱싱=%.2fs 검색평균=%.4fs VRAM=%s",
                        spec.label,
                        chunk_size,
                        format_load_time(metric),
                        metric.index_time_sec,
                        metric.retrieval_avg_sec,
                        format_vram(metric),
                    )
        finally:
            cleanup()

    duration = time.perf_counter() - t_start

    output_dir = create_report_output_dir()
    report_path = write_report(
        all_metrics,
        skipped_api=skipped_api,
        hardware_note=hw,
        duration_sec=duration,
        output_dir=output_dir,
    )
    print(f"\n완료. 리포트: {report_path}")
    print(f"산출물 폴더: {output_dir}")
    if skipped_api:
        print("참고: OpenAI 케이스는 API 키 없음으로 생략되었습니다.")


if __name__ == "__main__":
    main()
