from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import DOCS_PDF_DIR, MODELS_TO_TEST, allowed_chunk_keys, is_valid_chunk_key
from runtime_store import RuntimeStore

_PROJECT_ROOT = Path(__file__).resolve().parent


def load_dotenv_file(dotenv_path: Path | None = None) -> None:
    """main.py와 동일 규칙: 이미 설정된 환경변수는 덮어쓰지 않음."""
    import os

    path = dotenv_path or (_PROJECT_ROOT / ".env")
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_chunk_key_arg(s: str) -> int | str:
    s = str(s).strip()
    if s.isdigit():
        return int(s)
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="모델별/청크별 ChromaDB 사전 빌드. "
        "기본: meta+chroma.sqlite3가 유효하면 해당 조합은 건너뜀(미완료·실패만 재시도)."
    )
    p.add_argument("--rebuild", action="store_true", help="기존 chroma_db 폴더 전체를 지우고 재생성")
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="유효한 인덱스가 있어도 선택한 조합은 모두 다시 빌드(시간·VRAM 많이 듦)",
    )
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="(호환용, 무시됨) 예전부터 쓰이던 이름. 기본이 이미 '완료 건 스킵'입니다.",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="빌드할 model_id 목록 (미지정 시 전체)",
    )
    p.add_argument(
        "--chunks",
        nargs="*",
        default=[],
        help="빌드할 chunk_key (기본: config.DEFAULT_CHUNK_KEY). 미지정 시 허용 전체",
    )
    return p.parse_args()


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    if args.chunks:
        target_keys = [parse_chunk_key_arg(x) for x in args.chunks]
    else:
        target_keys = allowed_chunk_keys()

    invalid = [c for c in target_keys if not is_valid_chunk_key(c)]
    if invalid:
        raise SystemExit(f"지원하지 않는 chunk_key: {invalid} / 허용값: {allowed_chunk_keys()}")

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
    skip_if_ready = not args.no_skip
    try:
        for spec in selected:
            for ck in target_keys:
                if skip_if_ready and store.is_persist_index_ready(spec.model_id, ck):
                    rows.append(
                        {
                            "model_id": spec.model_id,
                            "chunk_key": ck,
                            "status": "skipped",
                            "elapsed_sec": 0.0,
                            "collection_name": store.collection_name(spec.model_id, ck),
                            "db_path": str(store.db_path(spec.model_id, ck)),
                            "error_message": None,
                        }
                    )
                    print(f"[SKIP] {spec.model_id} chunk_key={ck} (이미 유효한 인덱스)")
                    continue

                t0 = time.perf_counter()
                ok, err = store.ensure_built(spec, ck)
                elapsed = time.perf_counter() - t0
                rows.append(
                    {
                        "model_id": spec.model_id,
                        "chunk_key": ck,
                        "status": "ok" if ok else "failed",
                        "elapsed_sec": round(elapsed, 2),
                        "collection_name": store.collection_name(spec.model_id, ck),
                        "db_path": str(store.db_path(spec.model_id, ck)),
                        "error_message": err,
                    }
                )
                print(
                    f"[{'OK' if ok else 'FAIL'}] {spec.model_id} chunk_key={ck} "
                    f"({elapsed:.2f}s) -> {store.db_path(spec.model_id, ck)}"
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
        "- chunk_mode: semantic (고정)\n",
        f"- skip_if_ready: {skip_if_ready} (--no-skip 으로 끔)\n",
        f"- persist_root: `{persist_root}`\n",
        f"- models: {len(selected)}\n",
        f"- chunk_keys: {target_keys}\n\n",
        "| model_id | chunk_key | status | elapsed_sec | collection_name | db_path | error |\n",
        "|---|---:|---|---:|---|---|---|\n",
    ]
    for r in rows:
        err = (r["error_message"] or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r['model_id']} | {r['chunk_key']} | {r['status']} | {r['elapsed_sec']} | "
            f"{r['collection_name']} | {r['db_path']} | {err} |\n"
        )
    md.write_text("".join(lines), encoding="utf-8")
    js.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {md}")


if __name__ == "__main__":
    main()
