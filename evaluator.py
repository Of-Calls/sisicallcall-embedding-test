from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

try:
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "필수 패키지가 없습니다. 아래를 설치한 뒤 다시 실행하세요.\n"
        "  pip install -r requirements.txt"
    ) from e

from config import CHROMA_COSINE_CONFIG, CHUNK_OVERLAP, CaseMetrics, ModelSpec
from providers import build_local_embeddings, cuda_available, max_cuda_vram_usage_mb, reset_cuda_stats

SANITY_PREVIEW_MAX_CHARS = 150


def _compact_preview(text: str, limit: int = SANITY_PREVIEW_MAX_CHARS) -> str:
    one_line = " ".join(str(text).split())
    return (one_line[:limit] if len(one_line) > limit else one_line) + "..."


def _chunk_label(row: dict[str, float | int | str]) -> str:
    idx = int(row.get("chunk_index", -1))
    return f"chunk_{idx:03d}" if idx >= 0 else "chunk_---"


def print_retrieval_sanity_block(
    *,
    model_label: str,
    chunk_size: int,
    question: str,
    rows: list[dict[str, float | int | str]],
) -> None:
    if not rows:
        return
    header = f"========== [모델명: {model_label} | Chunk: {chunk_size}] =========="
    lines = [f"\n{header}", f"질문: {question}", "", "Top-3 검색 결과"]
    for i, row in enumerate(rows[:3], start=1):
        sim = float(row["cosine_similarity"])
        chunk = _chunk_label(row)
        preview = _compact_preview(str(row["text"]))
        lines.append(f"{i}) {chunk} | sim={sim:.3f}")
        lines.append(f" \"{preview}\"")
        if i < min(3, len(rows)):
            lines.append("")
    lines.append("=" * len(header))
    tqdm.write("\n".join(lines), file=sys.stderr)


def load_pdf_corpus(directory: Path) -> str:
    """`docs/` 등 디렉터리 내 모든 PDF를 읽어 페이지 텍스트를 하나로 병합합니다."""
    if not directory.is_dir():
        raise FileNotFoundError(
            f"PDF 디렉터리가 없습니다: {directory}\n"
            f"프로젝트 루트에 `{directory.name}` 폴더를 만들고 PDF를 넣은 뒤 다시 실행하세요."
        )
    loader = PyPDFDirectoryLoader(str(directory), glob="**/*.pdf")
    raw_docs = loader.load()
    if not raw_docs:
        raise ValueError(
            f"PDF가 없거나 로드 결과가 비었습니다: {directory}\n"
            f"`**/*.pdf` 패턴에 맞는 파일을 추가하세요."
        )
    parts = [d.page_content.strip() for d in raw_docs if d.page_content and str(d.page_content).strip()]
    if not parts:
        raise ValueError(f"PDF에서 추출된 텍스트가 없습니다(스캔 PDF 등): {directory}")
    return "\n\n".join(parts)


def split_documents(text: str, chunk_size: int, overlap: int = CHUNK_OVERLAP) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs: list[Document] = []
    for i, c in enumerate(chunks):
        docs.append(Document(page_content=c, metadata={"chunk_index": i, "chunk_size": chunk_size}))
    return docs


def chroma_similarity_from_distance(distance: float) -> float:
    """Chroma cosine space: distance = 1 - cos_sim → cos_sim = 1 - distance."""
    return max(0.0, min(1.0, 1.0 - distance))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def directory_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total / (1024 * 1024)


def run_single_case(
    full_text: str,
    spec: ModelSpec,
    chunk_size: int,
    queries: list[tuple[str, str]],
    retrieval_warmup: int = 1,
    retrieval_repeats: int = 5,
    embedder: Embeddings | None = None,
    load_time_override: float | None = None,
    owns_embedder: bool = True,
) -> CaseMetrics:
    documents = split_documents(full_text, chunk_size)
    collection_name = f"bench_{uuid.uuid4().hex}"

    reset_cuda_stats()
    load_time: float | None = load_time_override
    vram_max = 0.0
    cleanup: Callable[[], None] = lambda: None

    if embedder is None:
        t0 = time.perf_counter()
        embedder, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
        load_time = time.perf_counter() - t0
        if cuda_available():
            vram_max = max(vram_max, max_cuda_vram_usage_mb())
    elif cuda_available():
        vram_max = max(vram_max, max_cuda_vram_usage_mb())

    persist_dir = tempfile.mkdtemp(prefix="chroma_bench_")
    index_t0 = time.perf_counter()
    try:
        vs = Chroma.from_documents(
            documents=documents,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=persist_dir,
            collection_configuration=CHROMA_COSINE_CONFIG,
        )
        index_time = time.perf_counter() - index_t0
        if cuda_available():
            vram_max = max(vram_max, max_cuda_vram_usage_mb())

        timings: list[float] = []
        query_results_raw: dict[str, list[dict[str, float | int | str]]] = {}

        total_inner = len(queries) * (retrieval_warmup + retrieval_repeats)
        with tqdm(total=total_inner, desc=f"{spec.label} cs={chunk_size} retrieval", leave=False) as pbar:
            for _warm_idx in range(retrieval_warmup):
                for _qkey, qtext in queries:
                    vs.similarity_search_with_score(qtext, k=3)
                    pbar.update(1)
            for repeat_idx in range(retrieval_repeats):
                for qkey, qtext in queries:
                    tq = time.perf_counter()
                    scored = vs.similarity_search_with_score(qtext, k=3)
                    timings.append(time.perf_counter() - tq)
                    if repeat_idx == retrieval_repeats - 1:
                        rows = []
                        for rank, (doc, dist) in enumerate(scored, start=1):
                            rows.append(
                                {
                                    "rank": rank,
                                    "chroma_distance": dist,
                                    "cosine_similarity": chroma_similarity_from_distance(dist),
                                    "text": doc.page_content,
                                    "chunk_index": doc.metadata.get("chunk_index", -1),
                                }
                            )
                        query_results_raw[qkey] = rows
                        if rows and repeat_idx == retrieval_repeats - 1:
                            print_retrieval_sanity_block(
                                model_label=spec.label,
                                chunk_size=chunk_size,
                                question=qtext,
                                rows=rows,
                            )
                    pbar.update(1)

        retrieval_avg = sum(timings) / len(timings) if timings else 0.0
        retrieval_p95 = percentile(timings, 95)

        if cuda_available():
            vram_max = max(vram_max, max_cuda_vram_usage_mb())

        index_size_mb = directory_size_mb(persist_dir)
        embedding_docs_per_sec = (len(documents) / index_time) if index_time > 0 else 0.0

        return CaseMetrics(
            model_label=spec.label,
            model_id=spec.model_id,
            chunk_size=chunk_size,
            load_time_sec=load_time,
            index_time_sec=index_time,
            retrieval_avg_sec=retrieval_avg,
            retrieval_p95_sec=retrieval_p95,
            vram_peak_mb=vram_max if cuda_available() else None,
            index_size_mb=index_size_mb,
            embedding_docs_per_sec=embedding_docs_per_sec,
            query_results=query_results_raw,  # type: ignore[arg-type]
        )
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)
        if owns_embedder:
            cleanup()


def run_case_safe(
    full_text: str,
    spec: ModelSpec,
    chunk_size: int,
    queries: list[tuple[str, str]],
    embedder: Embeddings | None = None,
    load_time_override: float | None = None,
    owns_embedder: bool = True,
) -> tuple[CaseMetrics | None, str | None]:
    try:
        m = run_single_case(
            full_text,
            spec,
            chunk_size,
            queries,
            embedder=embedder,
            load_time_override=load_time_override,
            owns_embedder=owns_embedder,
        )
        return m, None
    except Exception as e:
        return None, str(e)
