#!/usr/bin/env python3
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

INPUT_PDF_PATH = Path(r"C:\ahy\final\bge_test\data\Store.pdf")
OUTPUT_DIR = Path(r"c:\ahy\final\bge_test\test_outputs")

A_REGEX = re.compile(r"(?=\n## )")
A_CHUNK_SIZE = 1000
A_CHUNK_OVERLAP = 100
B_MIN_MERGE_LEN = 350


def _read_pdf_markdown(pdf_path: Path) -> str:
    markdown = str(pymupdf4llm.to_markdown(str(pdf_path)) or "")
    return markdown.replace("\r\n", "\n").replace("\r", "\n").strip()


def _method_a_chunks(markdown: str, source_pdf: str) -> list[dict[str, Any]]:
    raw_parts = [p.strip() for p in A_REGEX.split(markdown) if p.strip()]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=A_CHUNK_SIZE,
        chunk_overlap=A_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict[str, Any]] = []
    idx = 0
    for part_idx, part in enumerate(raw_parts):
        if len(part) <= A_CHUNK_SIZE:
            chunks.append(
                {
                    "text": part,
                    "meta": {"method": "A", "source_pdf": source_pdf, "part_index": part_idx, "chunk_index": idx},
                }
            )
            idx += 1
            continue
        for sub_idx, piece in enumerate(splitter.split_text(part)):
            text = piece.strip()
            if not text:
                continue
            chunks.append(
                {
                    "text": text,
                    "meta": {
                        "method": "A",
                        "source_pdf": source_pdf,
                        "part_index": part_idx,
                        "sub_part_index": sub_idx,
                        "chunk_index": idx,
                    },
                }
            )
            idx += 1
    return chunks


def _method_b_chunks(markdown: str, source_pdf: str) -> list[dict[str, Any]]:
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")], strip_headers=False)
    docs = splitter.split_text(markdown)
    base_chunks: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        text = doc.page_content.strip()
        if not text:
            continue
        meta = {"method": "B", "source_pdf": source_pdf, **dict(doc.metadata), "chunk_index": idx}
        base_chunks.append({"text": text, "meta": meta})

    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(base_chunks):
        current = base_chunks[i]
        if len(current["text"]) <= B_MIN_MERGE_LEN and i + 1 < len(base_chunks):
            nxt = base_chunks[i + 1]
            merged_text = f"{current['text'].rstrip()}\n\n{nxt['text'].lstrip()}".strip()
            merged_meta = dict(nxt["meta"])
            merged_meta["merged_prev_short"] = True
            merged_meta["merged_prev_len"] = len(current["text"])
            merged_meta["merged_prev_meta"] = current["meta"]
            merged.append({"text": merged_text, "meta": merged_meta})
            i += 2
            continue
        merged.append(current)
        i += 1

    for idx, chunk in enumerate(merged):
        chunk["meta"]["chunk_index"] = idx
    return merged


def _extract_unique_headers(chunks: list[dict[str, Any]]) -> list[str]:
    unique_headers: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        meta = chunk.get("meta", {})
        if not isinstance(meta, dict):
            continue
        for key in ("h1", "h2"):
            value = str(meta.get(key, "")).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            unique_headers.append(value)
    return unique_headers


def _mock_local_llm_call(headers_list: list[str]) -> list[str]:
    _ = headers_list
    started_at = time.perf_counter()
    time.sleep(3)
    elapsed = time.perf_counter() - started_at
    categories = ["환불 규정", "요금제", "고객지원"]
    print(f"로컬 LLM 단 1회 호출 완료 ({elapsed:.2f}초)")
    return categories


def _write_chunks(path: Path, chunks: list[dict[str, Any]], doc_categories: list[str] | None = None) -> None:
    lines: list[str] = []
    if doc_categories:
        lines.append(f"[문서 대표 카테고리: {', '.join(doc_categories)}]\n\n")
    for i, item in enumerate(chunks, start=1):
        text = str(item["text"])
        meta = item["meta"]
        lines.append(f"=== Chunk {i} (Length: {len(text)}, Meta: {meta}) ===\n")
        lines.append(text)
        lines.append("\n\n")
    path.write_text("".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_stats(name: str, chunks: list[dict[str, Any]]) -> None:
    lengths = [len(str(c["text"])) for c in chunks]
    if not lengths:
        print(f"[{name}] chunks=0")
        return
    avg_len = sum(lengths) / len(lengths)
    print(f"[{name}] chunks={len(lengths)} | avg_len={avg_len:.1f} | min={min(lengths)} | max={max(lengths)}")


def _sanitize_name(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]+', "_", name).strip()
    return sanitized or "unnamed_pdf"


def main() -> None:
    if not INPUT_PDF_PATH.exists():
        raise SystemExit(f"지정한 PDF를 찾을 수 없습니다: {INPUT_PDF_PATH.resolve()}")
    pdfs = [INPUT_PDF_PATH]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    for pdf_path in pdfs:
        print(f"- 처리 중: {pdf_path.name}")
        markdown = _read_pdf_markdown(pdf_path)
        a_chunks = _method_a_chunks(markdown, pdf_path.name)
        b_chunks = _method_b_chunks(markdown, pdf_path.name)
        unique_headers = _extract_unique_headers(b_chunks)
        representative_categories = _mock_local_llm_call(unique_headers)

        pdf_dir = run_dir / _sanitize_name(pdf_path.stem)
        pdf_dir.mkdir(parents=True, exist_ok=True)
        a_path = pdf_dir / "A_chunks.txt"
        b_path = pdf_dir / "B_chunks.txt"
        _write_chunks(a_path, a_chunks)
        _write_chunks(b_path, b_chunks, doc_categories=representative_categories)

        print(f"\n[{pdf_path.name}]")
        _print_stats("A", a_chunks)
        _print_stats("B", b_chunks)
        print(f"A 결과: {a_path.resolve()}")
        print(f"B 결과: {b_path.resolve()}")

    print("\n=== Chunking Test 완료 ===")
    print(f"실행 폴더: {run_dir.resolve()}")


if __name__ == "__main__":
    main()