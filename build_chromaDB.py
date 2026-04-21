from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import CHUNK_SIZES, DOCS_PDF_DIR, MODELS_TO_TEST
from runtime_store import RuntimeStore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모델별/청크별 ChromaDB 사전 빌드")
    p.add_argument("--rebuild", action="store_true", help="기존 chroma_db 폴더를 지우고 재생성")
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="빌드할 model_id 목록 (미지정 시 전체)",
    )
    p.add_argument(
        "--chunk-sizes",
        nargs="*",
        type=int,
        default=[],
        help="빌드할 chunk_size 목록 (미지정 시 config 전체)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    target_chunks = args.chunk_sizes or CHUNK_SIZES
    invalid_chunks = [c for c in target_chunks if c not in CHUNK_SIZES]
    if invalid_chunks:
        raise SystemExit(f"지원하지 않는 chunk_size: {invalid_chunks} / 지원값: {CHUNK_SIZES}")

    selected = [m for m in MODELS_TO_TEST if not args.models or m.model_id in set(args.models)]
    if not selected:
        raise SystemExit("선택된 모델이 없습니다. --models 값을 확인하세요.")

    persist_root = DOCS_PDF_DIR / "chroma_db"
    if args.rebuild and persist_root.exists():
        import shutil

        shutil.rmtree(persist_root, ignore_errors=True)

    store = RuntimeStore(pdf_dir=Path(DOCS_PDF_DIR), max_models_in_cache=1)
    started = time.perf_counter()
    rows: list[dict] = []
    try:
        for spec in selected:
            for cs in target_chunks:
                t0 = time.perf_counter()
                ok, err = store.ensure_built(spec, cs)
                elapsed = time.perf_counter() - t0
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "chunk_size": cs,
                        "status": "ok" if ok else "failed",
                        "elapsed_sec": round(elapsed, 2),
                        "collection_name": store.collection_name(spec.model_id, cs),
                        "db_path": str(store.db_path(spec.model_id, cs)),
                        "error_message": err,
                    }
                )
                print(
                    f"[{'OK' if ok else 'FAIL'}] {spec.model_id} chunk={cs} "
                    f"({elapsed:.2f}s) -> {store.db_path(spec.model_id, cs)}"
                )
                if err:
                    print(f"  error: {err}")
    finally:
        store.close()

    total = time.perf_counter() - started
    report_dir = Path("reports") / time.strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    md = report_dir / "chroma_build_report.md"
    js = report_dir / "chroma_build_report.json"

    lines = [
        "# ChromaDB 빌드 리포트\n\n",
        f"- elapsed_sec: {total:.1f}\n",
        f"- persist_root: `{persist_root}`\n",
        f"- models: {len(selected)}\n",
        f"- chunk_sizes: {target_chunks}\n\n",
        "| model_id | chunk_size | status | elapsed_sec | collection_name | db_path | error |\n",
        "|---|---:|---|---:|---|---|---|\n",
    ]
    for r in rows:
        err = (r["error_message"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['model_id']} | {r['chunk_size']} | {r['status']} | {r['elapsed_sec']} | "
            f"{r['collection_name']} | {r['db_path']} | {err} |\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    js.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {md}")


if __name__ == "__main__":
    main()
