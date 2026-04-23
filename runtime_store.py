from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from config import (
    CHROMA_COSINE_CONFIG,
    ModelSpec,
    SEMANTIC_BREAKPOINT_PERCENTILE,
    SEMANTIC_CHUNK_MAX_LENGTH,
    SEMANTIC_CHUNK_MIN_LENGTH,
    allowed_chunk_keys,
    chunk_folder_segment,
    normalize_chunk_key,
)
from evaluator import chroma_similarity_from_distance, load_pdf_corpus, split_documents_semantic
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
    chunk_key: str
    vectorstore: Chroma
    persist_dir: str
    build_time_sec: float
    docs_count: int


class RuntimeStore:
    def __init__(self, pdf_dir: Path, max_models_in_cache: int = 1) -> None:
        self.pdf_dir = Path(pdf_dir)
        self.max_models_in_cache = max(1, max_models_in_cache)
        self.full_text = load_pdf_corpus(self.pdf_dir)
        self.persist_root = self.pdf_dir / "chroma_db"
        self.persist_root.mkdir(parents=True, exist_ok=True)
        self.pdf_signature = self._compute_pdf_signature()
        self.docs_cache: dict[tuple[Any, ...], list[Any]] = {}
        self.chunk_stats_cache: dict[tuple[Any, ...], tuple[int, float, int, int]] = {}
        self.model_cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self.vector_cache: dict[tuple[str, str], VectorStoreEntry] = {}
        self.hardware_note = detect_hardware_note()

    def _compute_pdf_signature(self) -> str:
        h = hashlib.sha256()
        pdfs = sorted(self.pdf_dir.glob("**/*.pdf"))
        for p in pdfs:
            st = p.stat()
            rel = p.relative_to(self.pdf_dir).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
        return h.hexdigest()

    def _model_slug(self, model_id: str) -> str:
        return model_id.replace("/", "__")

    def _db_dir(self, model_id: str, chunk_key: str) -> Path:
        seg = chunk_folder_segment(chunk_key)
        return self.persist_root / self._model_slug(model_id) / f"chunk_{seg}"

    def _meta_path(self, model_id: str, chunk_key: str) -> Path:
        return self._db_dir(model_id, chunk_key) / "meta.json"

    def _collection_name(self, model_id: str, chunk_key: str) -> str:
        seg = chunk_folder_segment(chunk_key)
        return f"bench_{self._model_slug(model_id)}_{seg}"

    def collection_name(self, model_id: str, chunk_key: str) -> str:
        return self._collection_name(model_id, chunk_key)

    def db_path(self, model_id: str, chunk_key: str) -> Path:
        return self._db_dir(model_id, chunk_key)

    def _meta_matches(self, model_id: str, chunk_key: str) -> bool:
        mp = self._meta_path(model_id, chunk_key)
        if not mp.is_file():
            return False
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return False
        if meta.get("model_id") != model_id:
            return False
        if meta.get("pdf_signature") != self.pdf_signature:
            return False
        if meta.get("chunk_mode") != "semantic":
            return False
        if str(meta.get("chunk_key")) != str(chunk_key):
            return False
        if float(meta.get("semantic_breakpoint_percentile", -1)) != float(SEMANTIC_BREAKPOINT_PERCENTILE):
            return False
        if int(meta.get("semantic_chunk_max_length", -1)) != int(SEMANTIC_CHUNK_MAX_LENGTH):
            return False
        if int(meta.get("semantic_chunk_min_length", -1)) != int(SEMANTIC_CHUNK_MIN_LENGTH):
            return False
        return True

    def is_persist_index_ready(self, model_id: str, chunk_key: str) -> bool:
        """디스크에 유효한 meta + Chroma DB 파일이 있으면 True (재빌드 불필요)."""
        if not self._meta_matches(model_id, chunk_key):
            return False
        db_dir = self._db_dir(model_id, chunk_key)
        if not (db_dir / "chroma.sqlite3").is_file():
            return False
        return True

    def _write_meta(self, model_id: str, chunk_key: str) -> None:
        mp = self._meta_path(model_id, chunk_key)
        mp.parent.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "model_id": model_id,
            "chunk_mode": "semantic",
            "chunk_key": chunk_key,
            "semantic_breakpoint_percentile": SEMANTIC_BREAKPOINT_PERCENTILE,
            "semantic_chunk_max_length": SEMANTIC_CHUNK_MAX_LENGTH,
            "semantic_chunk_min_length": SEMANTIC_CHUNK_MIN_LENGTH,
            "pdf_signature": self.pdf_signature,
            "updated_at": int(time.time()),
        }
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _docs_cache_key(self, spec: ModelSpec, chunk_key: str) -> tuple[Any, ...]:
        return (spec.model_id, chunk_key)

    def count_docs_cached(self, spec: ModelSpec, chunk_key: str) -> int:
        k = self._docs_cache_key(spec, chunk_key)
        d = self.docs_cache.get(k)
        return len(d) if d else 0

    def chunk_length_stats(self, spec: ModelSpec, chunk_key: str) -> tuple[int, float, int, int]:
        k = self._docs_cache_key(spec, chunk_key)
        return self.chunk_stats_cache.get(k, (0, 0.0, 0, 0))

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
            keys = [kk for kk in self.vector_cache if kk[0] == old_model_id]
            for kk in keys:
                self.vector_cache.pop(kk)
            doc_keys = [kk for kk in self.docs_cache if kk[0] == old_model_id]
            for kk in doc_keys:
                self.docs_cache.pop(kk)
                self.chunk_stats_cache.pop(kk, None)

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
        self, spec: ModelSpec, chunk_key: str
    ) -> tuple[VectorStoreEntry | None, float, str | None]:
        cache_k = (spec.model_id, chunk_key)
        if cache_k in self.vector_cache:
            self._touch_model(spec.model_id)
            return self.vector_cache[cache_k], 0.0, None

        loaded, _load_time, err = self.get_or_load_model(spec)
        if loaded is None:
            return None, 0.0, err

        try:
            dk = self._docs_cache_key(spec, chunk_key)
            docs = self.docs_cache.get(dk)
            stats: tuple[int, float, int, int]
            if docs is None:
                docs, stats = split_documents_semantic(self.full_text, loaded.embedder)
                self.docs_cache[dk] = docs
                self.chunk_stats_cache[dk] = stats
            else:
                cached_stats = self.chunk_stats_cache.get(dk)
                if cached_stats is None:
                    stats = compute_stats_from_docs(docs)
                    self.chunk_stats_cache[dk] = stats
                else:
                    stats = cached_stats

            t0 = time.perf_counter()
            db_dir = self._db_dir(spec.model_id, chunk_key)
            db_dir.mkdir(parents=True, exist_ok=True)

            if self._meta_matches(spec.model_id, chunk_key):
                vs = Chroma(
                    collection_name=self._collection_name(spec.model_id, chunk_key),
                    embedding_function=loaded.embedder,
                    persist_directory=str(db_dir),
                    collection_configuration=CHROMA_COSINE_CONFIG,
                )
            else:
                shutil.rmtree(db_dir, ignore_errors=True)
                db_dir.mkdir(parents=True, exist_ok=True)
                vs = Chroma.from_documents(
                    documents=docs,
                    embedding=loaded.embedder,
                    collection_name=self._collection_name(spec.model_id, chunk_key),
                    persist_directory=str(db_dir),
                    collection_configuration=CHROMA_COSINE_CONFIG,
                )
                self._write_meta(spec.model_id, chunk_key)
            sec = time.perf_counter() - t0
            entry = VectorStoreEntry(
                model_id=spec.model_id,
                chunk_key=chunk_key,
                vectorstore=vs,
                persist_dir=str(db_dir),
                build_time_sec=sec,
                docs_count=len(docs),
            )
            self.vector_cache[cache_k] = entry
            self._touch_model(spec.model_id)
            return entry, sec, None
        except Exception as e:
            return None, 0.0, str(e)

    @staticmethod
    def _chunk_response_fields(chunk_key: str) -> dict[str, Any]:
        return {"chunk_key": chunk_key}

    def query(self, spec: ModelSpec, chunk_key: str | int, question: str, k: int = 3) -> dict[str, Any]:
        chunk_key = normalize_chunk_key(chunk_key)
        if chunk_key not in allowed_chunk_keys():
            return {
                "status": "failed",
                "model_id": spec.model_id,
                **self._chunk_response_fields(chunk_key),
                "error_type": "invalid_chunk_key",
                "error_message": f"지원 chunk_key: {allowed_chunk_keys()}",
            }

        loaded, load_time, load_err = self.get_or_load_model(spec)
        if loaded is None:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                **self._chunk_response_fields(chunk_key),
                "error_type": "model_load_error",
                "error_message": load_err,
            }

        entry, index_time, idx_err = self.get_or_build_vectorstore(spec, chunk_key)
        if entry is None:
            return {
                "status": "failed",
                "model_id": spec.model_id,
                **self._chunk_response_fields(chunk_key),
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
                **self._chunk_response_fields(chunk_key),
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
                **self._chunk_response_fields(chunk_key),
                "error_type": "query_error",
                "error_message": str(e),
            }

    def cache_snapshot(self) -> dict[str, Any]:
        return {
            "chunk_mode": "semantic",
            "allowed_chunk_keys": allowed_chunk_keys(),
            "max_models_in_cache": self.max_models_in_cache,
            "loaded_models": list(self.model_cache.keys()),
            "vectorstores": [
                {"model_id": k[0], "chunk_key": k[1]}
                for k in sorted(self.vector_cache.keys(), key=lambda x: (x[0], str(x[1])))
            ],
            "persist_root": str(self.persist_root),
        }

    def ensure_built(self, spec: ModelSpec, chunk_key: str) -> tuple[bool, str | None]:
        entry, _sec, err = self.get_or_build_vectorstore(spec, chunk_key)
        return (entry is not None), err

    def close(self) -> None:
        for loaded in self.model_cache.values():
            try:
                loaded.cleanup()
            except Exception:
                pass
        self.model_cache.clear()
        self.vector_cache.clear()


def compute_stats_from_docs(docs: list[Any]) -> tuple[int, float, int, int]:
    from evaluator import compute_chunk_length_stats

    return compute_chunk_length_stats(docs)
