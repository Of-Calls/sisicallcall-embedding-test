from __future__ import annotations

import re
import shutil
import time
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


def compact_text(text: str, limit: int = 150) -> str:
    one_line = " ".join(str(text).split())
    return one_line[:limit] + ("..." if len(one_line) > limit else "")


def slugify_label(label: str) -> str:
    slug = re.sub(r"[\\/:*?\"<>|]+", "-", label).strip()
    slug = re.sub(r"\s+", "_", slug)
    return slug or "model"


def print_chunk_preview(docs: list[object], count: int = CHUNK_PREVIEW_COUNT, full_text: bool = False) -> None:
    end = min(count, len(docs))
    print(f"\n[청크 미리보기] 초반 {end}개")
    for i in range(end):
        doc = docs[i]
        raw_text = " ".join(str(getattr(doc, "page_content", "")).split())
        text = raw_text if full_text else compact_text(raw_text, limit=220)
        print(f"{i + 1:02d}) {text}")
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
        raw_docs = loader.load()
        if not raw_docs:
            raise ValueError(f"선택한 PDF에서 텍스트를 찾지 못했습니다: {pdf_path}")

        chunker = SemanticChunker(
            embedder,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95.0,
        )
        docs = chunker.split_documents(raw_docs)
        for i, doc in enumerate(docs):
            doc.metadata["chunk_index"] = i

        show_full = input("청크 원문 전체를 잘리지 않게 출력할까요? (y/n): ").strip().lower() == "y"
        print_chunk_preview(docs, full_text=show_full)
        shutil.rmtree(model_dir, ignore_errors=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[빌드] PDF: {pdf_path.name} | 페이지 {len(raw_docs)}개, 청크 {len(docs)}개 생성 중...")
        return Chroma.from_documents(
            documents=docs,
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
        preview = compact_text(getattr(doc, "page_content", ""))
        print(f'{i}) [sim={sim:.3f}] "{preview}"')
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
