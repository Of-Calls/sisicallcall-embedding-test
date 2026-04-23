from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config import REPORT_FILENAME, REPORTS_ROOT, CaseMetrics, TEST_QUERIES, allowed_chunk_keys


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


def _top1_top2_gap(m: CaseMetrics) -> float | None:
    gaps: list[float] = []
    for rows in m.query_results.values():
        if len(rows) < 2:
            continue
        s1 = float(rows[0].get("cosine_similarity", 0.0))
        s2 = float(rows[1].get("cosine_similarity", 0.0))
        gaps.append(s1 - s2)
    if not gaps:
        return None
    return sum(gaps) / len(gaps)


def _domain_hit_rate_at_3(m: CaseMetrics) -> float | None:
    # 질문 ID prefix(DJI/TESLA)와 Top-3 텍스트의 도메인 키워드 일치율
    dji_kw = ("dji", "드론", "비행", "기체", "조종기")
    tesla_kw = ("tesla", "모델", "차량", "트렁크", "충전", "주행")
    hits = 0
    total = 0
    for qid, rows in m.query_results.items():
        if not rows:
            continue
        top3_text = " ".join(str(r.get("text", "")) for r in rows[:3]).lower()
        if qid.startswith("DJI-"):
            hit = any(k in top3_text for k in dji_kw)
        elif qid.startswith("TESLA-"):
            hit = any(k in top3_text for k in tesla_kw)
        else:
            continue
        total += 1
        hits += 1 if hit else 0
    if total == 0:
        return None
    return hits / total


def _domain_hit_for_question(qid: str, rows: list[dict[str, object]]) -> bool | None:
    dji_kw = ("dji", "드론", "비행", "기체", "조종기")
    tesla_kw = ("tesla", "모델", "차량", "트렁크", "충전", "주행")
    if not rows:
        return None
    top3_text = " ".join(str(r.get("text", "")) for r in rows[:3]).lower()
    if qid.startswith("DJI-"):
        return any(k in top3_text for k in dji_kw)
    if qid.startswith("TESLA-"):
        return any(k in top3_text for k in tesla_kw)
    return None


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
    lines.append(f"- 케이스 수: {len(metrics)}\n")
    lines.append(f"- 청킹: 시멘틱 고정 / 활성 chunk_key: `{allowed_chunk_keys()}`\n\n")
    lines.append("정량 자동 채점(Hit@3, MRR)은 제외하고, 검색 문맥 확인 중심으로 기록했습니다.\n\n")

    lines.append("## 1) 모델/청크 실행 요약\n\n")
    lines.append(
        "| 모델 | Chunk | 상태 | 에러 | 로드 | 인덱싱 | Avg Latency | 검색 P95 | Top1-Top2 Gap | Domain Hit@3 | VRAM | 인덱스 크기 | 처리량 |\n"
    )
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for m in metrics:
        err = (m.error_message or "").replace("\n", " ").replace("|", "\\|")
        gap = _top1_top2_gap(m)
        gap_str = f"{gap:.3f}" if gap is not None else "—"
        hit = _domain_hit_rate_at_3(m)
        hit_str = f"{hit*100:.1f}%" if hit is not None else "—"
        lines.append(
            f"| {m.model_label} | {m.chunk_key} | {m.status} | {err} | {format_load_time(m)} | "
            f"{m.index_time_sec:.2f}s | {m.retrieval_avg_sec*1000:.1f}ms | {m.retrieval_p95_sec:.4f}s | "
            f"{gap_str} | {hit_str} | {format_vram(m)} | {m.index_size_mb:.2f} MB | {m.embedding_docs_per_sec:.2f} docs/s |\n"
        )

    lines.append("\n## 2) 질문별 성능 비교\n")
    for qid, qtext in TEST_QUERIES:
        lines.append(f"\n### {qid}\n")
        lines.append(f"- 질문: {qtext}\n\n")
        lines.append("| 모델 | Chunk | 상태 | Top1 sim | Top1-Top2 Gap | Latency | Domain Hit@3 |\n")
        lines.append("|---|---:|---|---:|---:|---:|---|\n")
        for m in metrics:
            rows = m.query_results.get(qid, [])
            status = m.status if rows or m.status != "ok" else "no_result"
            top1 = float(rows[0].get("cosine_similarity", 0.0)) if rows else None
            gap = None
            if len(rows) >= 2:
                gap = float(rows[0].get("cosine_similarity", 0.0)) - float(
                    rows[1].get("cosine_similarity", 0.0)
                )
            lat = m.query_latency_ms.get(qid)
            hit = _domain_hit_for_question(qid, rows)
            lines.append(
                f"| {m.model_label} | {m.chunk_key} | {status} | "
                f"{(f'{top1:.3f}' if top1 is not None else '—')} | "
                f"{(f'{gap:.3f}' if gap is not None else '—')} | "
                f"{(f'{lat:.1f}ms' if lat is not None else '—')} | "
                f"{('Y' if hit else ('N' if hit is False else '—'))} |\n"
            )

    lines.append("\n## 3) 질문별 Top-3 결과\n")
    for qid, qtext in TEST_QUERIES:
        lines.append(f"\n### {qid}\n")
        lines.append(f"- 질문: {qtext}\n\n")
        for m in metrics:
            rows = m.query_results.get(qid, [])
            lines.append(f"#### {m.model_label} | Chunk {m.chunk_key}\n")
            if m.status != "ok":
                lines.append(f"- 실패: `{m.error_type}` - {m.error_message}\n\n")
                continue
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