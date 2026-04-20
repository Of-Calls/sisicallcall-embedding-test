from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
import uuid
from typing import Callable

try:
    import tiktoken
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "필수 패키지가 없습니다. 아래를 설치한 뒤 다시 실행하세요.\n"
        "  pip install -r requirements.txt"
    ) from e

from config import (
    CHROMA_COSINE_CONFIG,
    CHUNK_OVERLAP,
    OPENAI_EMBEDDING_PRICE_PER_1M_USD,
    CaseMetrics,
    ModelSpec,
)
from providers import (
    build_local_embeddings,
    build_openai_embeddings,
    cuda_available,
    max_cuda_vram_usage_mb,
    reset_cuda_stats,
)


def load_manual(path) -> str:
    """벤치마크 입력은 프로젝트의 `data/manual.txt`를 사용합니다."""
    if not path.is_file():
        raise FileNotFoundError(
            f"매뉴얼 파일이 없습니다: {path}\n"
            f"`data/manual.txt`(비전클라우드 고객센터 매뉴얼 등)를 두고 다시 실행하세요."
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"매뉴얼 파일이 비어 있습니다: {path}")
    return text


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


def doc_matches_keywords(doc_text: str, keywords: list[str]) -> bool:
    return any(k in doc_text for k in keywords)


def count_tokens_openai(text: str) -> int:
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


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
    queries: list[tuple[str, str, list[str]]],
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
        if spec.kind == "local":
            t0 = time.perf_counter()
            embedder, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
            load_time = time.perf_counter() - t0
            if cuda_available():
                vram_max = max(vram_max, max_cuda_vram_usage_mb())
        else:
            embedder = build_openai_embeddings(spec.model_id)
            if embedder is None:
                raise RuntimeError("OPENAI_API_KEY 가 설정되지 않았습니다.")
    elif spec.kind == "local" and cuda_available():
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
        if cuda_available() and spec.kind == "local":
            vram_max = max(vram_max, max_cuda_vram_usage_mb())

        timings: list[float] = []
        query_results_raw: dict[str, list[dict[str, float | int | str]]] = {}

        total_inner = len(queries) * (retrieval_warmup + retrieval_repeats)
        with tqdm(total=total_inner, desc=f"{spec.label} cs={chunk_size} retrieval", leave=False) as pbar:
            for _warm_idx in range(retrieval_warmup):
                for _qkey, qtext, _kws in queries:
                    vs.similarity_search_with_score(qtext, k=3)
                    pbar.update(1)
            for repeat_idx in range(retrieval_repeats):
                for qkey, qtext, _kws in queries:
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
                                }
                            )
                        query_results_raw[qkey] = rows
                    pbar.update(1)

        retrieval_avg = sum(timings) / len(timings) if timings else 0.0
        retrieval_p95 = percentile(timings, 95)

        if cuda_available() and spec.kind == "local":
            vram_max = max(vram_max, max_cuda_vram_usage_mb())

        # 품질 지표 (최종 반복 Top-3 기준)
        hit_count = 0
        reciprocal_ranks: list[float] = []
        for qkey, _, kws in queries:
            rows = query_results_raw.get(qkey, [])
            rr = 0.0
            for row in rows:
                if doc_matches_keywords(str(row["text"]), kws):
                    rr = 1.0 / int(row["rank"])
                    break
            if rr > 0:
                hit_count += 1
            reciprocal_ranks.append(rr)
        denom = len(queries) if queries else 1
        hit_at_3 = hit_count / denom
        mrr_at_3 = sum(reciprocal_ranks) / denom

        index_size_mb = directory_size_mb(persist_dir)
        embedding_docs_per_sec = (len(documents) / index_time) if index_time > 0 else 0.0

        token_count: int | None = None
        cost_str: str
        if spec.kind == "openai":
            token_count = count_tokens_openai(full_text)
            cost = (token_count / 1_000_000.0) * OPENAI_EMBEDDING_PRICE_PER_1M_USD
            cost_str = f"${cost:.6f} (cl100k_base 추정)"
        else:
            cost_str = "$0 (단, 하드웨어 고정비 발생)"

        return CaseMetrics(
            model_label=spec.label,
            model_id=spec.model_id,
            chunk_size=chunk_size,
            load_time_sec=load_time,
            index_time_sec=index_time,
            retrieval_avg_sec=retrieval_avg,
            retrieval_p95_sec=retrieval_p95,
            vram_peak_mb=vram_max if cuda_available() and spec.kind == "local" else None,
            index_size_mb=index_size_mb,
            embedding_docs_per_sec=embedding_docs_per_sec,
            hit_at_3=hit_at_3,
            mrr_at_3=mrr_at_3,
            token_count=token_count,
            estimated_cost_usd=cost_str,
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
    queries: list[tuple[str, str, list[str]]],
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
