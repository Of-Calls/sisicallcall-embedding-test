from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from config import CHROMA_COSINE_CONFIG, CHUNK_SIZES, DOCS_PDF_DIR, MODELS_TO_TEST
from evaluator import chroma_similarity_from_distance, load_pdf_corpus, split_documents
from providers import build_local_embeddings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chunk_size: int | None = None
    top_k: int = Field(default=3, ge=1, le=10)


class ModelRuntime:
    def __init__(self, label: str, model_id: str, use_e5_prefix: bool) -> None:
        self.label = label
        self.model_id = model_id
        self.use_e5_prefix = use_e5_prefix
        self.embedder = None
        self.cleanup = lambda: None
        self.vectorstores: dict[int, Chroma] = {}
        self.persist_dirs: list[str] = []
        self.load_time_sec: float = 0.0


app = FastAPI(title="Embedding Sanity Server", version="1.0.0")
RUNTIMES: dict[str, ModelRuntime] = {}


def _build_runtime(full_text: str) -> None:
    for spec in MODELS_TO_TEST:
        rt = ModelRuntime(spec.label, spec.model_id, spec.use_e5_prefix)
        t0 = time.perf_counter()
        emb, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
        rt.embedder = emb
        rt.cleanup = cleanup
        rt.load_time_sec = time.perf_counter() - t0

        for chunk_size in CHUNK_SIZES:
            docs = split_documents(full_text, chunk_size)
            persist_dir = tempfile.mkdtemp(prefix=f"serve_{spec.model_id.replace('/', '_')}_")
            vs = Chroma.from_documents(
                documents=docs,
                embedding=rt.embedder,
                collection_name=f"serve_{spec.model_id.replace('/', '_')}_{chunk_size}",
                persist_directory=persist_dir,
                collection_configuration=CHROMA_COSINE_CONFIG,
            )
            rt.vectorstores[chunk_size] = vs
            rt.persist_dirs.append(persist_dir)
        RUNTIMES[spec.model_id] = rt


@app.on_event("startup")
def startup_event() -> None:
    full_text = load_pdf_corpus(Path(DOCS_PDF_DIR))
    _build_runtime(full_text)


@app.on_event("shutdown")
def shutdown_event() -> None:
    import shutil

    for rt in RUNTIMES.values():
        try:
            rt.cleanup()
        except Exception:
            pass
        for d in rt.persist_dirs:
            shutil.rmtree(d, ignore_errors=True)
    RUNTIMES.clear()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": len(RUNTIMES),
        "model_labels": [rt.label for rt in RUNTIMES.values()],
        "chunk_sizes": CHUNK_SIZES,
    }


@app.post("/query")
def query_all(req: QueryRequest) -> dict[str, Any]:
    if not RUNTIMES:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    chunk_sizes = [req.chunk_size] if req.chunk_size else CHUNK_SIZES
    invalid = [cs for cs in chunk_sizes if cs not in CHUNK_SIZES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 chunk_size: {invalid}")

    results: list[dict[str, Any]] = []
    for rt in RUNTIMES.values():
        for cs in chunk_sizes:
            vs = rt.vectorstores[cs]
            scored = vs.similarity_search_with_score(req.question, k=req.top_k)
            rows: list[dict[str, Any]] = []
            for rank, (doc, dist) in enumerate(scored, start=1):
                rows.append(
                    {
                        "rank": rank,
                        "chunk_index": int(doc.metadata.get("chunk_index", -1)),
                        "cosine_similarity": round(chroma_similarity_from_distance(float(dist)), 4),
                        "text_preview": (" ".join(doc.page_content.split())[:150] + "..."),
                    }
                )
            results.append(
                {
                    "model_label": rt.label,
                    "model_id": rt.model_id,
                    "chunk_size": cs,
                    "rows": rows,
                }
            )
    return {"question": req.question, "results": results}
