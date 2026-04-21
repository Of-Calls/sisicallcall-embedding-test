from __future__ import annotations

import os
from typing import Any, Callable

try:
    import torch
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.embeddings import Embeddings
except ImportError as e:
    raise SystemExit(
        "필수 패키지가 없습니다. 아래를 설치한 뒤 다시 실행하세요.\n"
        "  pip install -r requirements.txt"
    ) from e


class E5PrefixEmbeddings(Embeddings):
    """intfloat E5 계열 권장: 문서 passage:, 쿼리 query:"""

    def __init__(self, inner: HuggingFaceEmbeddings) -> None:
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"query: {text}")


def cuda_available() -> bool:
    return torch.cuda.is_available()


def mb_from_bytes(b: float) -> float:
    return b / (1024 * 1024)


def reset_cuda_stats() -> None:
    if cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def snapshot_cuda_allocated_mb() -> float:
    if not cuda_available():
        return 0.0
    return mb_from_bytes(torch.cuda.memory_allocated())


def snapshot_cuda_peak_mb() -> float:
    if not cuda_available():
        return 0.0
    return mb_from_bytes(torch.cuda.max_memory_allocated())


def max_cuda_vram_usage_mb() -> float:
    """요구사항 반영: `memory_allocated()` 스냅샷과 `max_memory_allocated()` 피크 중 큰 값."""
    if not cuda_available():
        return 0.0
    return max(snapshot_cuda_allocated_mb(), snapshot_cuda_peak_mb())


def _needs_trust_remote_code(model_id: str) -> bool:
    mid = model_id.lower()
    return any(
        token in mid
        for token in ("bge-m3", "jina-embeddings", "gte-multilingual", "alibaba-nlp")
    )


def build_local_embeddings(model_id: str, use_e5_prefix: bool) -> tuple[Embeddings, Callable[[], None]]:
    # VRAM·처리량: CUDA 사용 시 GPU 고정, 배치 16 기본(OOM 시 EMBED_BATCH_SIZE로 하향)
    model_kwargs: dict[str, Any] = {"device": "cuda" if cuda_available() else "cpu"}
    if _needs_trust_remote_code(model_id):
        model_kwargs["trust_remote_code"] = True

    encode_kwargs: dict[str, Any] = {
        "normalize_embeddings": True,
        "batch_size": int(os.environ.get("EMBED_BATCH_SIZE", "16")),
    }

    hf = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    emb: Embeddings = E5PrefixEmbeddings(hf) if use_e5_prefix else hf

    def cleanup() -> None:
        try:
            del emb
        except Exception:
            pass
        if cuda_available():
            torch.cuda.empty_cache()

    return emb, cleanup


def detect_hardware_note() -> str:
    parts = []
    if cuda_available():
        parts.append(f"CUDA: {torch.cuda.get_device_name(0)}")
    else:
        parts.append("CUDA 사용 불가 (CPU 모드)")
    parts.append(f"PyTorch {torch.__version__}")
    return " / ".join(parts)
