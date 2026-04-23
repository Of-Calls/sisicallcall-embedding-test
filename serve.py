from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from pydantic import BaseModel, Field

from config import (
    CHROMA_COSINE_CONFIG,
    DOCS_PDF_DIR,
    MODELS_TO_TEST,
    allowed_chunk_keys,
    chunk_folder_segment,
    is_valid_chunk_key,
    semantic_percentile,
)
from evaluator import chroma_similarity_from_distance, load_pdf_corpus, split_documents_semantic
from providers import build_local_embeddings


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chunk_key: str | None = None
    top_k: int = Field(default=3, ge=1, le=10)


class ModelRuntime:
    def __init__(self, label: str, model_id: str, use_e5_prefix: bool) -> None:
        self.label = label
        self.model_id = model_id
        self.use_e5_prefix = use_e5_prefix
        self.embedder = None
        self.cleanup = lambda: None
        self.vectorstores: dict[str, Chroma] = {}
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

        for ck in allowed_chunk_keys():
            docs = split_documents_semantic(full_text, rt.embedder, ck, semantic_percentile(ck))
            persist_dir = tempfile.mkdtemp(prefix=f"serve_{spec.model_id.replace('/', '_')}_")
            seg = chunk_folder_segment(ck)

            vs = Chroma.from_documents(
                documents=docs,
                embedding=rt.embedder,
                collection_name=f"serve_{spec.model_id.replace('/', '_')}_{seg}",
                persist_directory=persist_dir,
                collection_configuration=CHROMA_COSINE_CONFIG,
            )
            rt.vectorstores[ck] = vs
            rt.persist_dirs.append(persist_dir)
        RUNTIMES[spec.model_id] = rt


@app.on_event("startup")
def startup_event() -> None:
    full_text = load_pdf_corpus(Path(DOCS_PDF_DIR))
    _build_runtime(full_text)


@app.on_event("shutdown")
def shutdown_event() -> None:
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
        "chunk_mode": "semantic",
        "chunk_keys": allowed_chunk_keys(),
        "models_loaded": len(RUNTIMES),
        "model_labels": [rt.label for rt in RUNTIMES.values()],
    }


@app.post("/query")
def query_all(req: QueryRequest) -> dict[str, Any]:
    if not RUNTIMES:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    keys = [req.chunk_key] if req.chunk_key is not None else allowed_chunk_keys()

    invalid = [x for x in keys if not is_valid_chunk_key(x)]
    if invalid:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 chunk_key: {invalid}")

    results: list[dict[str, Any]] = []
    for rt in RUNTIMES.values():
        for ck in keys:
            vs = rt.vectorstores[ck]
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
                    "chunk_key": ck,
                    "rows": rows,
                }
            )
    return {"question": req.question, "results": results}
