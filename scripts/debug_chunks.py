from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKER_PATH = PROJECT_ROOT / "app" / "services" / "rag" / "chunker.py"


def _load_chunker() -> Any:
    if not CHUNKER_PATH.is_file():
        raise FileNotFoundError(f"chunker.py를 찾을 수 없습니다: {CHUNKER_PATH}")
    spec = importlib.util.spec_from_file_location("project_chunker", CHUNKER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"chunker 모듈 로딩 실패: {CHUNKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["project_chunker"] = module
    spec.loader.exec_module(module)
    return module


chunk_pdf_by_headers = _load_chunker().chunk_pdf_by_headers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="헤더 기반 청킹 결과 디버그 출력")
    parser.add_argument("--pdf", required=True, help="대상 PDF 파일 경로")
    parser.add_argument("--tenant", required=True, help="tenant id")
    return parser.parse_args()


def _preview(text: str, limit: int = 100) -> str:
    one_line = " ".join(str(text).split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def print_table(chunks: list[dict[str, Any]]) -> None:
    print("\n청킹 결과 테이블")
    print("-" * 110)
    print(f"{'번호':<6}{'섹션 제목':<40}{'글자 수':<10}{'미리보기(앞 100자)'}")
    print("-" * 110)
    for idx, chunk in enumerate(chunks, start=1):
        section_title = str(chunk["metadata"].get("section_title", "제목없음"))
        text = str(chunk.get("text", ""))
        length = len(text)
        warn = ""
        if length < 300:
            warn = " [경고:짧음]"
        elif length > 1200:
            warn = " [경고:김]"
        print(
            f"{idx:<6}{section_title[:38]:<40}{length:<10}{_preview(text)}{warn}"
        )
    print("-" * 110)


def print_stats(chunks: list[dict[str, Any]]) -> None:
    lengths = [len(str(chunk.get("text", ""))) for chunk in chunks]
    if not lengths:
        print("청크가 없습니다.")
        return

    print("\n요약 통계")
    print(f"- 전체 청크 수: {len(lengths)}")
    print(f"- 평균 글자 수: {mean(lengths):.2f}")
    print(f"- 최소 글자 수: {min(lengths)}")
    print(f"- 최대 글자 수: {max(lengths)}")


def save_outputs(chunks: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"chunks_{ts}.json"
    txt_path = output_dir / f"chunks_{ts}.txt"

    json_rows = []
    txt_blocks: list[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        section_title = str(chunk["metadata"].get("section_title", "제목없음"))
        text = str(chunk.get("text", ""))
        char_count = len(text)
        metadata = chunk.get("metadata", {})
        json_rows.append(
            {
                "chunk_number": idx,
                "section_title": section_title,
                "text": text,
                "metadata": metadata,
            }
        )

        txt_blocks.append(
            "\n".join(
                [
                    "=" * 40,
                    f"청크 #{idx} | 섹션: {section_title} | 글자 수: {char_count}",
                    "=" * 40,
                    text,
                    "",
                ]
            )
        )

    json_path.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    txt_path.write_text("\n".join(txt_blocks), encoding="utf-8")
    return json_path, txt_path


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

    chunks = chunk_pdf_by_headers(str(pdf_path), tenant_id=args.tenant)
    print_table(chunks)
    print_stats(chunks)

    json_path, txt_path = save_outputs(chunks, output_dir=Path("debug_output"))
    print("\n저장 완료")
    print(f"- JSON: {json_path}")
    print(f"- TXT: {txt_path}")


if __name__ == "__main__":
    main()
