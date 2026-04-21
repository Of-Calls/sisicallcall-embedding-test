#!/usr/bin/env python3
"""
sisicallcall — 임베딩 모델 RAG/Semantic Cache 벤치마크
로컬(GPU) 임베딩을 ChromaDB + LangChain 파이프라인으로 비교합니다.

입력은 `data/` 디렉터리의 복수 PDF입니다.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path

from tqdm import tqdm

from config import CHUNK_SIZES, DOCS_PDF_DIR, MODELS_TO_TEST, REPORTS_ROOT, TEST_QUERIES
from evaluator import load_pdf_corpus, run_case_safe
from providers import build_local_embeddings, detect_hardware_note
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

    hw = detect_hardware_note()
    log.info("환경: %s", hw)
    migrate_legacy_report_if_exists()

    try:
        full_text = load_pdf_corpus(DOCS_PDF_DIR)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        sys.exit(1)

    log.info("입력 PDF 코퍼스: %s (%d chars)", DOCS_PDF_DIR.resolve(), len(full_text))

    all_metrics = []
    t_start = time.perf_counter()

    # 요청 반영: 모델을 먼저 전부 로드한 뒤 테스트를 수행
    loaded_models: dict[str, tuple[object, float, object, object]] = {}
    # value: (embedder, load_time_sec, cleanup_fn, spec)
    for spec in MODELS_TO_TEST:
        t0 = time.perf_counter()
        try:
            emb, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
            load_sec = time.perf_counter() - t0
            loaded_models[spec.model_id] = (emb, load_sec, cleanup, spec)
            log.info("[선로드 완료] %s: %.2fs", spec.label, load_sec)
        except Exception as e:
            log.error("[선로드 실패] %s: %s", spec.label, e)

    if not loaded_models:
        log.error("로드 가능한 모델이 없습니다. 실행을 종료합니다.")
        sys.exit(1)

    total_cases = len(MODELS_TO_TEST) * len(CHUNK_SIZES)
    outer = tqdm(total=total_cases, desc="전체 케이스", unit="case")

    try:
        for spec in MODELS_TO_TEST:
            loaded = loaded_models.get(spec.model_id)
            if loaded is None:
                for _ in CHUNK_SIZES:
                    outer.update(1)
                continue
            shared_embedder, shared_load_time, _cleanup, _loaded_spec = loaded
            first_chunk = True

            for chunk_size in CHUNK_SIZES:
                outer.set_postfix(model=spec.label[:24], chunk=chunk_size)
                metric, err = run_case_safe(
                    full_text,
                    spec,
                    chunk_size,
                    TEST_QUERIES,
                    embedder=shared_embedder,  # type: ignore[arg-type]
                    load_time_override=shared_load_time if first_chunk else 0.0,  # type: ignore[arg-type]
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
        for model_id, (_emb, _sec, cleanup, spec) in loaded_models.items():
            try:
                cleanup()  # type: ignore[misc]
                log.info("[언로드 완료] %s (%s)", spec.label, model_id)  # type: ignore[attr-defined]
            except Exception as e:
                log.warning("[언로드 실패] %s: %s", model_id, e)

    duration = time.perf_counter() - t_start

    output_dir = create_report_output_dir()
    report_path = write_report(
        all_metrics,
        hardware_note=hw,
        duration_sec=duration,
        output_dir=output_dir,
    )
    print(f"\n완료. 리포트: {report_path}")
    print(f"산출물 폴더: {output_dir}")


if __name__ == "__main__":
    main()
