from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import REPORT_FILENAME, REPORTS_ROOT, CaseMetrics, TEST_QUERIES


def create_report_output_dir(base_dir: Path = REPORTS_ROOT) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = base_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir


def format_load_time(m: CaseMetrics) -> str:
    if m.load_time_sec is None:
        return "—"
    if m.load_time_sec == 0:
        return "0.00s (cached)"
    return f"{m.load_time_sec:.2f}s"


def format_vram(m: CaseMetrics) -> str:
    if m.vram_peak_mb is None:
        return "—"
    return f"{m.vram_peak_mb:.1f} MB"


def write_report(
    metrics: list[CaseMetrics],
    hardware_note: str,
    duration_sec: float,
    output_dir: Path,
) -> Path:
    report_path = output_dir / REPORT_FILENAME
    lines: list[str] = []
    lines.append("# 임베딩 벤치마크 리포트 (Sanity Check)\n\n")
    lines.append(f"- 실행 환경: {hardware_note}\n")
    lines.append(f"- 총 실행 시간: {duration_sec:.1f}s\n")
    lines.append(f"- 케이스 수: {len(metrics)}\n\n")
    lines.append("정량 자동 채점(Hit@3, MRR)은 제외하고, 검색 문맥 확인 중심으로 기록했습니다.\n\n")

    lines.append("## 1) 모델/청크 실행 요약\n\n")
    lines.append("| 모델 | Chunk | 로드 | 인덱싱 | 검색 평균 | 검색 P95 | VRAM | 인덱스 크기 | 처리량 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for m in metrics:
        lines.append(
            f"| {m.model_label} | {m.chunk_size} | {format_load_time(m)} | "
            f"{m.index_time_sec:.2f}s | {m.retrieval_avg_sec:.4f}s | {m.retrieval_p95_sec:.4f}s | "
            f"{format_vram(m)} | {m.index_size_mb:.2f} MB | {m.embedding_docs_per_sec:.2f} docs/s |\n"
        )

    lines.append("\n## 2) 질문별 Top-3 결과\n")
    for qid, qtext in TEST_QUERIES:
        lines.append(f"\n### {qid}\n")
        lines.append(f"- 질문: {qtext}\n\n")
        for m in metrics:
            rows = m.query_results.get(qid, [])
            lines.append(f"#### {m.model_label} | Chunk {m.chunk_size}\n")
            if not rows:
                lines.append("- 결과 없음\n\n")
                continue
            for row in rows[:3]:
                chunk_idx = int(row.get("chunk_index", -1))
                chunk_label = f"chunk_{chunk_idx:03d}" if chunk_idx >= 0 else "chunk_---"
                preview = " ".join(str(row["text"]).split())
                preview = preview[:200] + ("..." if len(preview) > 200 else "")
                lines.append(
                    f"- Rank {int(row['rank'])} | {chunk_label} | sim={float(row['cosine_similarity']):.3f} | "
                    f"dist={float(row['chroma_distance']):.3f}\n"
                )
                lines.append(f"  - {preview}\n")
            lines.append("\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path