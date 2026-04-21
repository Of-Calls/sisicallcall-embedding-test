from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import CHUNK_SIZES, DOCS_PDF_DIR, MODELS_TO_TEST
from runtime_store import RuntimeStore

log = logging.getLogger("embedding_server")
app = FastAPI(title="Embedding Benchmark Server")
store: RuntimeStore | None = None


class QueryRequest(BaseModel):
    model_id: str
    chunk_size: int
    question: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class QueryAllRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chunk_size: int
    k: int = Field(default=3, ge=1, le=10)


@app.on_event("startup")
def startup() -> None:
    global store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    max_models = int(os.environ.get("MODEL_CACHE_SIZE", "1"))
    store = RuntimeStore(pdf_dir=DOCS_PDF_DIR, max_models_in_cache=max_models)
    log.info("하드웨어: %s", store.hardware_note)
    log.info("캐시 정책: model LRU size=%s", max_models)
    log.info("chunk sizes: %s", CHUNK_SIZES)


@app.on_event("shutdown")
def shutdown() -> None:
    global store
    if store is not None:
        store.close()
        store = None


@app.get("/health")
def health() -> dict[str, Any]:
    if store is None:
        return {"status": "starting"}
    return {
        "status": "ok",
        "hardware": store.hardware_note,
        "cache": store.cache_snapshot(),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    if store is None:
        return {"status": "starting", "models": []}
    return {
        "status": "ok",
        "models": [
            {"label": m.label, "model_id": m.model_id, "use_e5_prefix": m.use_e5_prefix}
            for m in MODELS_TO_TEST
        ],
        "chunk_sizes": CHUNK_SIZES,
        "cache": store.cache_snapshot(),
    }


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    if store is None:
        return {"status": "failed", "error_type": "server_not_ready", "error_message": "store not ready"}
    spec = next((m for m in MODELS_TO_TEST if m.model_id == req.model_id), None)
    if spec is None:
        return {
            "status": "failed",
            "model_id": req.model_id,
            "chunk_size": req.chunk_size,
            "error_type": "unknown_model",
            "error_message": "config.MODELS_TO_TEST에 없는 모델입니다.",
        }
    return store.query(spec=spec, chunk_size=req.chunk_size, question=req.question, k=req.k)


@app.post("/query_all")
def query_all(req: QueryAllRequest) -> dict[str, Any]:
    if store is None:
        return {"status": "failed", "error_type": "server_not_ready", "error_message": "store not ready"}
    outputs: list[dict[str, Any]] = []
    for spec in MODELS_TO_TEST:
        outputs.append(store.query(spec=spec, chunk_size=req.chunk_size, question=req.question, k=req.k))
    return {
        "status": "ok",
        "question": req.question,
        "chunk_size": req.chunk_size,
        "k": req.k,
        "results": outputs,
    }
