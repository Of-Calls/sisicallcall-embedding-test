from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_chroma import Chroma

MARKDOWN_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
FAQ_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:#{1,6}\s*)?Q\s*(\d+)\s*[.)]\s*(.+?)\s*$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^\s*■\s*(.+?)\s*$")

MIN_SECTION_LENGTH = 300
MAX_SECTION_LENGTH = 1200


def _normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # OCR/추출 과정에서 단어 중간 줄바꿈이 발생하면 단어를 복원한다.
    normalized = re.sub(r"(?<=[가-힣A-Za-z0-9])\n(?=[가-힣A-Za-z0-9])", "", normalized)
    # "- " 형태의 빈 리스트 노이즈를 제거한다.
    normalized = re.sub(r"(?m)^\s*-\s*$", "", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"[ \t]*\n[ \t]*", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _sanitize_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name))
    return cleaned.strip("_") or "tenant_default"


def _extract_markdown_from_pdf(pdf_path: str) -> str:
    markdown = pymupdf4llm.to_markdown(pdf_path)
    return _normalize_text(markdown)


def _clean_header_text(header_text: str) -> str:
    title = " ".join(header_text.split()).strip()
    # 헤더 라인에 본문이 붙은 경우(예: 문장부호 뒤 본문) 제목 부분만 남긴다.
    title = re.split(r"(?<=[.!?])\s+", title, maxsplit=1)[0]
    return title.strip(" -:\t") or "문서 전체"


def _compose_section_title(section: dict[str, Any]) -> str:
    subsection = str(section.get("subsection", "")).strip()
    if subsection:
        return subsection
    sec = str(section.get("section", "")).strip()
    if sec:
        return sec
    parent = str(section.get("parent_section", "")).strip()
    if parent:
        return parent
    return "문서 전체"


def _build_sections_from_markdown(markdown_text: str) -> list[dict[str, Any]]:
    lines = markdown_text.split("\n")
    sections: list[dict[str, Any]] = []

    current_parent = ""
    current_section = ""
    current_subsection = ""
    active: dict[str, Any] | None = None

    def start_section() -> dict[str, Any]:
        return {
            "parent_section": current_parent,
            "section": current_section,
            "subsection": current_subsection,
            "title": "",
            "content_lines": [],
        }

    for raw in lines:
        line = raw.rstrip()
        m = MARKDOWN_HEADER_RE.match(line.strip())
        if m:
            if active is not None:
                active["content"] = _normalize_text("\n".join(active.pop("content_lines")))
                active["title"] = _compose_section_title(active)
                sections.append(active)
                active = None

            level = len(m.group(1))
            header_text = _clean_header_text(m.group(2))
            if level == 1:
                current_parent = header_text
                current_section = ""
                current_subsection = ""
            elif level == 2:
                current_section = header_text
                current_subsection = ""
            elif level == 3:
                current_subsection = header_text

            active = start_section()
            continue

        if active is None:
            active = start_section()
        active["content_lines"].append(line)

    if active is not None:
        active["content"] = _normalize_text("\n".join(active.pop("content_lines")))
        active["title"] = _compose_section_title(active)
        sections.append(active)

    cleaned: list[dict[str, Any]] = []
    for item in sections:
        content = str(item.get("content", "")).strip()
        if not content and not item.get("title"):
            continue
        cleaned.append(
            {
                "parent_section": str(item.get("parent_section", "")).strip(),
                "section": str(item.get("section", "")).strip(),
                "subsection": str(item.get("subsection", "")).strip(),
                "title": str(item.get("title", "문서 전체")).strip() or "문서 전체",
                "content": content,
            }
        )

    if cleaned:
        return cleaned
    return [{"parent_section": "", "section": "", "subsection": "", "title": "문서 전체", "content": markdown_text}]


def _merge_small_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(sections):
        current = sections[i]
        content = current["content"].strip()
        should_merge = (not content) or (len(content) < MIN_SECTION_LENGTH)

        if should_merge and i + 1 < len(sections):
            nxt = sections[i + 1]
            merged_title = f"{current['title']} / {nxt['title']}"
            merged_content = "\n".join(part for part in [content, nxt["content"].strip()] if part).strip()
            sections[i + 1] = {
                "parent_section": nxt.get("parent_section", "") or current.get("parent_section", ""),
                "section": nxt.get("section", "") or current.get("section", ""),
                "subsection": nxt.get("subsection", "") or current.get("subsection", ""),
                "title": merged_title,
                "content": merged_content,
            }
        elif should_merge and merged:
            prev = merged[-1]
            prev["title"] = f"{prev['title']} / {current['title']}"
            if content:
                prev["content"] = f"{prev['content']}\n{content}".strip()
        else:
            merged.append(
                {
                    "parent_section": current.get("parent_section", ""),
                    "section": current.get("section", ""),
                    "subsection": current.get("subsection", ""),
                    "title": current["title"],
                    "content": content,
                }
            )
        i += 1
    return merged


def _split_faq_chunks(section: dict[str, Any]) -> list[dict[str, Any]] | None:
    section_title = section["title"]
    content = section["content"]
    lines = content.split("\n")
    faq_indexes = [i for i, line in enumerate(lines) if FAQ_RE.match(line.strip())]
    if not faq_indexes:
        return None

    chunks: list[dict[str, Any]] = []
    for idx, start in enumerate(faq_indexes):
        end = faq_indexes[idx + 1] if idx + 1 < len(faq_indexes) else len(lines)
        qa_text = "\n".join(lines[start:end]).strip()
        if not qa_text:
            continue
        q_match = FAQ_RE.match(lines[start].strip())
        q_number = q_match.group(1) if q_match else str(idx + 1)
        chunks.append(
            {
                "parent_section": section.get("parent_section", ""),
                "section": section.get("section", ""),
                "subsection": section.get("subsection", ""),
                "title": f"{section_title} - Q{q_number}",
                "content": qa_text,
            }
        )
    return chunks


def _split_large_section(section: dict[str, Any]) -> list[dict[str, Any]]:
    section_title = section["title"]
    content = section["content"]
    if len(content) <= MAX_SECTION_LENGTH:
        return [section]

    lines = content.split("\n")
    sub_indexes = [i for i, line in enumerate(lines) if SUBSECTION_RE.match(line.strip())]
    if not sub_indexes:
        return [section]

    chunks: list[dict[str, Any]] = []
    for idx, start in enumerate(sub_indexes):
        end = sub_indexes[idx + 1] if idx + 1 < len(sub_indexes) else len(lines)
        subtitle_line = lines[start].strip()
        subtitle = subtitle_line.lstrip("■").strip() or f"소제목{idx + 1}"
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append(
                {
                    "parent_section": section.get("parent_section", ""),
                    "section": section.get("section", ""),
                    "subsection": section.get("subsection", ""),
                    "title": f"{section_title} - {subtitle}",
                    "content": body,
                }
            )

    if chunks:
        return chunks
    return [section]


def _finalize_chunks(
    section_units: list[dict[str, Any]], tenant_id: str, source_file: str
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, section in enumerate(section_units):
        section_title = section["title"].strip() or "제목없음"
        content = _normalize_text(section["content"])
        chunk_text = f"[{section_title}]: {content}"
        results.append(
            {
                "text": chunk_text,
                "metadata": {
                    "section_title": section_title,
                    "chunk_index": idx,
                    "source_file": source_file,
                    "tenant_id": tenant_id,
                    "parent_section": str(section.get("parent_section", "")).strip(),
                    "section": str(section.get("section", "")).strip(),
                    "subsection": str(section.get("subsection", "")).strip(),
                },
            }
        )
    return results


def chunk_pdf_by_headers(pdf_path: str, tenant_id: str) -> list[dict[str, Any]]:
    """
    PDF를 헤더 기반으로 청킹하고 Chroma 저장용 데이터 형태를 반환한다.

    Returns:
        [{"text": "...", "metadata": {...}}, ...]
    """
    source_file = Path(pdf_path).name
    full_text = _extract_markdown_from_pdf(pdf_path)
    if not full_text:
        return []

    sections = _build_sections_from_markdown(full_text)
    merged_sections = _merge_small_sections(sections)

    units: list[dict[str, Any]] = []
    for section in merged_sections:
        if not section["content"].strip():
            continue

        faq_chunks = _split_faq_chunks(section)
        if faq_chunks:
            units.extend(faq_chunks)
            continue

        units.extend(_split_large_section(section))

    return _finalize_chunks(units, tenant_id=tenant_id, source_file=source_file)


def save_chunks_to_chroma(
    chunks: list[dict[str, Any]],
    tenant_id: str,
    embedding_function: Any,
    persist_directory: str = "chroma_db",
) -> Chroma:
    """
    청크를 tenant별 Chroma 컬렉션에 저장한다.
    """
    if not chunks:
        raise ValueError("저장할 청크가 없습니다.")

    texts = [item["text"] for item in chunks]
    metadatas = [item["metadata"] for item in chunks]
    source_name = Path(str(metadatas[0].get("source_file", "source"))).stem
    ids = [f"{tenant_id}:{source_name}:{i}" for i in range(len(chunks))]
    collection_name = _sanitize_collection_name(f"tenant_{tenant_id}")

    persist_path = Path(persist_directory)
    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma.from_texts(
        texts=texts,
        embedding=embedding_function,
        metadatas=metadatas,
        ids=ids,
        collection_name=collection_name,
        persist_directory=str(persist_path),
    )
