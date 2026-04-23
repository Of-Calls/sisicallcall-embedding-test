from __future__ import annotations

import re
import shutil
import time
import textwrap
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from config import CHROMA_COSINE_CONFIG, MODELS_TO_TEST, ModelSpec
from evaluator import chroma_similarity_from_distance
from providers import build_local_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIR = PROJECT_ROOT / "data"
ISOLATED_DB_ROOT = PROJECT_ROOT / "chroma_db_격리"
TOP_K = 3
CHUNK_PREVIEW_COUNT = 40
MIN_CHUNK_LENGTH = 60


def compact_text(text: str, limit: int = 150) -> str:
    one_line = " ".join(str(text).split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def slugify_label(label: str) -> str:
    slug = re.sub(r"[\\/:*?\"<>|]+", "-", label).strip()
    slug = re.sub(r"\s+", "_", slug)
    return slug or "model"


def preprocess_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rule A: If a line break follows a non-sentence-ending character,
    # force-join with a space (applies to both \n and \n\n).
    text = re.sub(r"(?<![다요까.?!])\s*\n+\s*", " ", text)
    # Rule B: Compress noisy spaces/newlines into a cleaner form.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def merge_small_chunks(docs: list[object], min_length: int = MIN_CHUNK_LENGTH) -> list[object]:
    if not docs:
        return []

    merged: list[object] = []
    current_doc = None
    current_text = ""

    for doc in docs:
        text = str(getattr(doc, "page_content", "")).strip()
        if not text:
            continue

        if current_doc is None:
            current_doc = doc
            current_text = text
            continue

        if len(current_text) < min_length:
            # 누적 텍스트가 짧으면 다음 청크를 강제로 이어 붙인다.
            current_text = f"{current_text} {text}".strip()
            setattr(current_doc, "page_content", current_text)
            continue

        setattr(current_doc, "page_content", current_text)
        merged.append(current_doc)
        current_doc = doc
        current_text = text

    if current_doc is not None:
        if len(current_text) < min_length and merged:
            prev = merged[-1]
            prev_text = str(getattr(prev, "page_content", "")).strip()
            setattr(prev, "page_content", f"{prev_text} {current_text}".strip())
        else:
            setattr(current_doc, "page_content", current_text)
            merged.append(current_doc)

    for i, doc in enumerate(merged):
        getattr(doc, "metadata", {})["chunk_index"] = i
    return merged


def print_chunk_preview(docs: list[object], count: int = CHUNK_PREVIEW_COUNT, full_text: bool = False) -> None:
    end = min(count, len(docs))
    print(f"\n[청크 미리보기] 초반 {end}개")
    print("-" * 90)
    for i in range(end):
        doc = docs[i]
        raw_text = str(getattr(doc, "page_content", ""))
        compact = " ".join(raw_text.split())
        chunk_idx = getattr(doc, "metadata", {}).get("chunk_index", i)
        print(f"[{i + 1:02d}/{end:02d}] chunk_index={chunk_idx} | chars={len(compact)}")

        if full_text:
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            if not paragraphs:
                print("(빈 청크)")
            else:
                for para_idx, para in enumerate(paragraphs, start=1):
                    print(textwrap.fill(para, width=100))
                    if para_idx != len(paragraphs):
                        print()
        else:
            preview = compact_text(compact, limit=420)
            print(textwrap.fill(preview, width=100))

        print("-" * 90)
    print("[청크 미리보기 종료]\n")


def choose_model() -> ModelSpec:
    print("\n테스트할 모델을 선택하세요:")
    for idx, spec in enumerate(MODELS_TO_TEST, start=1):
        print(f"{idx}) {spec.label} ({spec.model_id})")

    while True:
        raw = input("모델 번호 입력: ").strip()
        if not raw.isdigit():
            print("숫자 번호를 입력해 주세요.")
            continue
        num = int(raw)
        if 1 <= num <= len(MODELS_TO_TEST):
            return MODELS_TO_TEST[num - 1]
        print(f"1부터 {len(MODELS_TO_TEST)} 사이 번호를 입력해 주세요.")


def choose_single_pdf() -> Path:
    if not PDF_DIR.is_dir():
        raise FileNotFoundError(f"PDF 디렉터리가 없습니다: {PDF_DIR}")

    pdf_files = sorted(PDF_DIR.glob("**/*.pdf"))
    if not pdf_files:
        raise ValueError(f"data/ 폴더에서 PDF를 찾지 못했습니다: {PDF_DIR}")

    print("\n테스트할 PDF를 선택하세요:")
    for idx, pdf in enumerate(pdf_files, start=1):
        rel = pdf.relative_to(PDF_DIR).as_posix()
        print(f"{idx}) {rel}")

    while True:
        raw = input("PDF 번호 입력: ").strip()
        if not raw.isdigit():
            print("숫자 번호를 입력해 주세요.")
            continue
        num = int(raw)
        if 1 <= num <= len(pdf_files):
            return pdf_files[num - 1]
        print(f"1부터 {len(pdf_files)} 사이 번호를 입력해 주세요.")


def build_or_load_vectorstore(spec: ModelSpec, embedder: object, pdf_path: Path) -> Chroma:
    model_dir = ISOLATED_DB_ROOT / slugify_label(spec.label) / slugify_label(pdf_path.stem)
    rebuild = input('선택한 PDF로 DB를 새로 빌드(Rebuild) 하시겠습니까? (y/n): ').strip().lower()

    if rebuild == "y":
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        if not docs:
            raise ValueError(f"선택한 PDF에서 텍스트를 찾지 못했습니다: {pdf_path}")

        source_page_count = len(docs)
        for doc in docs:
            doc.page_content = preprocess_pdf_text(doc.page_content)

        text_splitter = SemanticChunker(
            embedder,
            buffer_size=3,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=70.0,
        )
        # 1) 분할
        chunks = text_splitter.split_documents(docs)
        # 2) 강제 병합 (반드시 덮어쓰기)
        chunks = merge_small_chunks(chunks, min_length=MIN_CHUNK_LENGTH)

        show_full = input("청크 원문 전체를 잘리지 않게 출력할까요? (y/n): ").strip().lower() == "y"
        # 3) 병합 완료된 최종 chunks로 미리보기 출력
        print_chunk_preview(chunks, full_text=show_full)
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[빌드] PDF: {pdf_path.name} | 페이지 {source_page_count}개, 청크 {len(chunks)}개 생성 중...")
        # 4) 병합된 최종 chunks를 DB에 저장
        return Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_name=f"isolated_{slugify_label(spec.model_id)}_{slugify_label(pdf_path.stem)}",
            persist_directory=str(model_dir),
            collection_configuration=CHROMA_COSINE_CONFIG,
        )

    if rebuild != "n":
        raise ValueError("입력은 y 또는 n 만 가능합니다.")

    if not model_dir.exists() or not (model_dir / "chroma.sqlite3").is_file():
        raise FileNotFoundError(
            f"기존 DB를 찾을 수 없습니다: {model_dir}\n먼저 y를 선택해 DB를 빌드해 주세요."
        )

    print(f"\n[로드] 기존 DB 사용: {model_dir}")
    return Chroma(
        collection_name=f"isolated_{slugify_label(spec.model_id)}_{slugify_label(pdf_path.stem)}",
        embedding_function=embedder,
        persist_directory=str(model_dir),
        collection_configuration=CHROMA_COSINE_CONFIG,
    )


def print_results(spec: ModelSpec, latency_sec: float, scored: list[tuple[object, float]]) -> None:
    latency_ms = latency_sec * 1000.0
    print(f"\n========== [모델명: {spec.label}] ==========")
    print(f"⏱️ 검색 레이턴시: {latency_sec:.3f} 초 ({latency_ms:.0f} ms)\n")
    print("Top-3 검색 결과:")
    for i, (doc, distance) in enumerate(scored, start=1):
        sim = chroma_similarity_from_distance(float(distance))
        raw_text = str(getattr(doc, "page_content", "")).strip()
        chunk_idx = getattr(doc, "metadata", {}).get("chunk_index", "?")
        print(f"{i}) [chunk_index={chunk_idx}] [sim={sim:.3f}]")
        if raw_text:
            paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
            for para_idx, para in enumerate(paragraphs, start=1):
                print(textwrap.fill(para, width=100))
                if para_idx != len(paragraphs):
                    print()
        else:
            print("(빈 청크)")
        print("-" * 90)
    print("===============================================")


def interactive_loop(spec: ModelSpec, vectorstore: Chroma) -> None:
    while True:
        question = input("\n질문> ").strip()
        if question.lower() in {"exit", "quit"}:
            print("종료합니다.")
            return
        if not question:
            continue

        t0 = time.perf_counter()
        scored = vectorstore.similarity_search_with_score(question, k=TOP_K)
        latency = time.perf_counter() - t0
        print_results(spec, latency, scored)


def main() -> None:
    spec = choose_model()
    pdf_path = choose_single_pdf()
    print(f"\n[모델 로드] {spec.label} ({spec.model_id})")
    print(f"[PDF 선택] {pdf_path.relative_to(PDF_DIR).as_posix()}")
    embedder, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
    try:
        vectorstore = build_or_load_vectorstore(spec, embedder, pdf_path)
        interactive_loop(spec, vectorstore)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
