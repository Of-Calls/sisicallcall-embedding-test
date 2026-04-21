from __future__ import annotations

import shutil
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import CHROMA_COSINE_CONFIG, CHUNK_SIZES, ModelSpec
from evaluator import chroma_similarity_from_distance, load_pdf_corpus, split_documents
from providers import build_local_embeddings, detect_hardware_note


@dataclass
class LoadedModel:
    spec: ModelSpec
    embedder: Embeddings
    cleanup: Any
    loaded_at: float


@dataclass
class VectorStoreEntry:
    model_id: str
    chunk_size: int
    vectorstore: Chroma
    persist_dir: str
    build_time_sec: float
    docs_count: int


class RuntimeStore:
    def __init__(self, pdf_dir: Path, max_models_in_cache: int = 1) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.max_models_in_cache = max(1, max_models_in_cache)
        self.full_text = load_pdf_corpus(self.pdf_dir)
        self.docs_cache: dict[int, list[Any]] = {}
        self.model_cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self.vector_cache: dict[tuple[str, int], VectorStoreEntry] = {}
        self.hardware_note = detect_hardware_note()

    def _touch_model(self, model_id: str) -> None:
        if model_id in self.model_cache:
            self.model_cache.move_to_end(model_id)

    def _evict_if_needed(self) -> None:
        while len(self.model_cache) > self.max_models_in_cache:
            old_model_id, old = self.model_cache.popitem(last=False)
            try:
                old.cleanup()
            except Exception:
                pass
            keys = [k for k in self.vector_cache if k[0] == old_model_id]
            for k in keys:
                entry = self.vector_cache.pop(k)
                shutil.rmtree(entry.persist_dir, ignore_errors=True)

    def get_or_load_model(self, spec: ModelSpec) -> tuple[LoadedModel | None, float, str | None]:
        if spec.model_id in self.model_cache:
            self._touch_model(spec.model_id)
            return self.model_cache[spec.model_id], 0.0, None
        try:
            t0 = time.perf_counter()
            emb, cleanup = build_local_embeddings(spec.model_id, spec.use_e5_prefix)
            load_time = time.perf_counter() - t0
            loaded = LoadedModel(spec=spec, embedder=emb, cleanup=cleanup, loaded_at=time.time())
            self.model_cache[spec.model_id] = loaded
            self._touch_model(spec.model_id)
            self._evict_if_needed()
            return loaded, load_time, None
        except Exception as e:
            return None, 0.0, str(e)

    def get_or_build_vectorstore(
        self, spec: ModelSpec, chunk_size: int
    ) -> tuple[VectorStoreEntry | None, float, str | None]:
        key = (spec.model_id, chunk_size)
        if key in self.vector_cache:
            self._touch_model(spec.model_id)
            return self.vector_cache[key], 0.0, None

        loaded, _load_time, err = self.get_or_load_model(spec)
        if loaded is None:
            return None, 0.0, err

        try:
            docs = self.docs_cache.get(chunk_size)
            if docs is None:
                docs = split_documents(self.full_text, chunk_size)
                self.docs_cache[chunk_size] = docs

            t0 = time.perf_counter()
            persist_dir = tempfile.mkdtemp(prefix=f"vs_{spec.model_id.replace('/', '_')}_{chunk_size}_")
            vs = Chroma.from_documents(
                documents=docs,
                embedding=loaded.embedder,
                collection_name=f"bench_{spec.model_id.replace('/', '_')}_{chunk_size}",
                persist_directory=persist_dir,
                collection_configuration=CHROMA_COSINE_CONFIG,
            )
            sec = time.perf_counter() - t0
            entry = VectorStoreEntry(
                model_id=spec.model_id,
                chunk_size=chunk_size,
                vectorstore=vs,
                persist_dir=persist_dir,
                build_time_sec=sec,
                docs_count=len(docs),
            )
            self.vector_cache[key] = entry
            self._touch_model(spec.model_id)
            return entry, sec, None
        except Exception as e:
            return None, 0.0, str(e)

    def query(self, spec: ModelSpec, chunk_size: int, question: str, k: int = 3) -> dict[str, Any]:
        if chunk_size not in CHUNK_SIZES:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                "chunk_size": chunk_size,
                "error_type": "invalid_chunk_size",
                "error_message": f"지원 chunk_size: {CHUNK_SIZES}",
            }

        loaded, load_time, load_err = self.get_or_load_model(spec)
        if loaded is None:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                "chunk_size": chunk_size,
                "error_type": "model_load_error",
                "error_message": load_err,
            }

        entry, index_time, idx_err = self.get_or_build_vectorstore(spec, chunk_size)
        if entry is None:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                "chunk_size": chunk_size,
                "error_type": "index_build_error",
                "error_message": idx_err,
            }

        try:
            t0 = time.perf_counter()
            scored = entry.vectorstore.similarity_search_with_score(question, k=k)
            qsec = time.perf_counter() - t0
            rows = []
            for rank, (doc, dist) in enumerate(scored, start=1):
                rows.append(
                    {
                        "rank": rank,
                        "chunk_id": f"chunk_{int(doc.metadata.get('chunk_index', -1)):03d}",
                        "chunk_index": int(doc.metadata.get("chunk_index", -1)),
                        "cosine_similarity": round(chroma_similarity_from_distance(float(dist)), 3),
                        "chroma_distance": float(dist),
                        "text": doc.page_content,
                    }
                )
            return {
                "status": "ok",
                "model_id": spec.model_id,
                "chunk_size": chunk_size,
                "question": question,
                "load_time_sec": load_time,
                "index_time_sec": index_time,
                "latency_ms": round(qsec * 1000, 2),
                "results": rows,
            }
        except Exception as e:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                "chunk_size": chunk_size,
                "error_type": "query_error",
                "error_message": str(e),
            }

    def cache_snapshot(self) -> dict[str, Any]:
        return {
            "max_models_in_cache": self.max_models_in_cache,
            "loaded_models": list(self.model_cache.keys()),
            "vectorstores": [
                {"model_id": k[0], "chunk_size": k[1]} for k in sorted(self.vector_cache.keys())
            ],
        }

    def close(self) -> None:
        for loaded in self.model_cache.values():
            try:
                loaded.cleanup()
            except Exception:
                pass
        for entry in self.vector_cache.values():
            shutil.rmtree(entry.persist_dir, ignore_errors=True)
        self.model_cache.clear()
        self.vector_cache.clear()
