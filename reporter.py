from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from config import (
    CHART_FILENAME,
    MANUAL_PATH,
    REPORT_FILENAME,
    REPORTS_ROOT,
    CaseMetrics,
    TEST_QUERIES,
)
from evaluator import doc_matches_keywords

log = logging.getLogger("embedding_benchmark")


def create_report_output_dir(base_dir: Path = REPORTS_ROOT) -> Path:
    """실행 시각 기반 보고서 폴더를 생성합니다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = base_dir / timestamp
    report_dir.mkdir(parents=True, exist_ok=False)
    return report_dir

def format_load_time(m: CaseMetrics) -> str:
    if m.load_time_sec is None:
        return "— (API)"
    if m.load_time_sec == 0:
        return "0.00s (cached)"
    return f"{m.load_time_sec:.2f}s"

def format_vram(m: CaseMetrics) -> str:
    if m.vram_peak_mb is None:
        return "—"
    return f"{m.vram_peak_mb:.1f} MB"


def format_rate(v: float) -> str:
    return f"{v:.2f} docs/s"


def recommend_combo(metrics: list[CaseMetrics]) -> tuple[str, str]:
    """5초 이내 레이턴시 + 로컬 VRAM 관점에서 휴리스틱 추천."""
    locals_only = [m for m in metrics if m.load_time_sec is not None]
    if not locals_only:
        locals_only = metrics
    scored: list[tuple[float, CaseMetrics]] = []
    for m in locals_only:
        latency_score = m.retrieval_avg_sec + 0.05 * m.index_time_sec + 0.2 * m.retrieval_p95_sec
        vram_penalty = (m.vram_peak_mb or 0) / 8192.0
        quality_bonus = 0.5 * m.hit_at_3 + 0.5 * m.mrr_at_3
        combined = latency_score + vram_penalty - 0.1 * quality_bonus
        scored.append((combined, m))
    scored.sort(key=lambda x: x[0])
    best = scored[0][1]
    reason = (
        f"로컬 기준 가중 점수(평균/95p 검색 지연 + 인덱싱×0.05 + VRAM/8GB - 품질 보너스)가 가장 유리했습니다. "
        f"(측정값: 검색 평균 {best.retrieval_avg_sec*1000:.1f} ms, P95 {best.retrieval_p95_sec*1000:.1f} ms, 인덱싱 {best.index_time_sec:.2f}s"
        + (f", 피크 VRAM 약 {best.vram_peak_mb:.0f} MB)" if best.vram_peak_mb else ")")
    )
    return f"{best.model_label} + Chunk {best.chunk_size}", reason

def save_visualization_charts(metrics: list[CaseMetrics], output_path: Path) -> None:
    """Matplotlib을 사용해 모델/청크별 성능 지표를 그룹 막대 그래프로 시각화 및 저장"""
    if not metrics:
        log.warning("시각화할 데이터가 없습니다.")
        return

    # 1. 윈도우 한글 폰트 및 마이너스 기호 깨짐 방지 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 데이터 그룹화 (모델명, 청크 사이즈 추출)
    models = list(dict.fromkeys(m.model_label for m in metrics))
    chunk_sizes = list(dict.fromkeys(m.chunk_size for m in metrics))

    # 차트용 데이터 컨테이너
    latency_data = {cs: [] for cs in chunk_sizes}
    latency_p95_data = {cs: [] for cs in chunk_sizes}
    vram_data = {cs: [] for cs in chunk_sizes}
    index_data = {cs: [] for cs in chunk_sizes}
    hit3_data = {cs: [] for cs in chunk_sizes}
    mrr3_data = {cs: [] for cs in chunk_sizes}

    for cs in chunk_sizes:
        for mod in models:
            # 해당 모델 + 청크 조합 찾기
            match = next((m for m in metrics if m.model_label == mod and m.chunk_size == cs), None)
            if match:
                latency_data[cs].append(match.retrieval_avg_sec)
                latency_p95_data[cs].append(match.retrieval_p95_sec)
                vram_data[cs].append(match.vram_peak_mb or 0.0)
                index_data[cs].append(match.index_time_sec)
                hit3_data[cs].append(match.hit_at_3)
                mrr3_data[cs].append(match.mrr_at_3)
            else:
                # 데이터가 누락된 경우 0으로 처리
                latency_data[cs].append(0.0)
                latency_p95_data[cs].append(0.0)
                vram_data[cs].append(0.0)
                index_data[cs].append(0.0)
                hit3_data[cs].append(0.0)
                mrr3_data[cs].append(0.0)

    # 3. 레이아웃 설정 (2행 3열)
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle('sisicallcall 임베딩 모델 성능 벤치마크 (로컬 최적화)', fontsize=16, fontweight='bold', y=1.05)
    axes_flat = axes.flatten()

    x = np.arange(len(models))
    width = 0.8 / len(chunk_sizes)

    # 4. 차트 그리기
    for i, cs in enumerate(chunk_sizes):
        offset = (i - len(chunk_sizes) / 2 + 0.5) * width
        
        # 차트 1: 평균 검색 레이턴시
        axes_flat[0].bar(x + offset, latency_data[cs], width, label=f'Chunk {cs}')
        # 차트 2: P95 검색 레이턴시
        axes_flat[1].bar(x + offset, latency_p95_data[cs], width, label=f'Chunk {cs}')
        # 차트 2: VRAM 사용량
        axes_flat[2].bar(x + offset, vram_data[cs], width, label=f'Chunk {cs}')
        # 차트 3: 인덱싱 소요 시간
        axes_flat[3].bar(x + offset, index_data[cs], width, label=f'Chunk {cs}')
        # 차트 5: Hit@3
        axes_flat[4].bar(x + offset, hit3_data[cs], width, label=f'Chunk {cs}')
        # 차트 6: MRR@3
        axes_flat[5].bar(x + offset, mrr3_data[cs], width, label=f'Chunk {cs}')

    # 차트 1 (레이턴시) 설정
    axes_flat[0].set_title('검색 평균 레이턴시 (짧을수록 좋음)')
    axes_flat[0].set_ylabel('평균 소요 시간 (초)')
    axes_flat[0].set_xticks(x)
    axes_flat[0].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[0].legend()

    # 차트 2 (P95 레이턴시) 설정
    axes_flat[1].set_title('검색 P95 레이턴시 (짧을수록 좋음)')
    axes_flat[1].set_ylabel('P95 소요 시간 (초)')
    axes_flat[1].set_xticks(x)
    axes_flat[1].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[1].legend()

    # 차트 3 (VRAM) 설정
    axes_flat[2].set_title('피크 VRAM 사용량 (적을수록 좋음)')
    axes_flat[2].set_ylabel('메모리 할당량 (MB)')
    axes_flat[2].set_xticks(x)
    axes_flat[2].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[2].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[2].legend()

    # 차트 4 (인덱싱) 설정
    axes_flat[3].set_title('텍스트 DB 인덱싱 시간 (짧을수록 좋음)')
    axes_flat[3].set_ylabel('소요 시간 (초)')
    axes_flat[3].set_xticks(x)
    axes_flat[3].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[3].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[3].legend()

    # 차트 5 (Hit@3) 설정
    axes_flat[4].set_title('Hit@3 (높을수록 좋음)')
    axes_flat[4].set_ylabel('Hit 비율')
    axes_flat[4].set_ylim(0, 1.05)
    axes_flat[4].set_xticks(x)
    axes_flat[4].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[4].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[4].legend()

    # 차트 6 (MRR@3) 설정
    axes_flat[5].set_title('MRR@3 (높을수록 좋음)')
    axes_flat[5].set_ylabel('MRR')
    axes_flat[5].set_ylim(0, 1.05)
    axes_flat[5].set_xticks(x)
    axes_flat[5].set_xticklabels(models, rotation=30, ha='right')
    axes_flat[5].grid(axis='y', linestyle='--', alpha=0.7)
    axes_flat[5].legend()

    # 5. 여백 정리 및 고화질 저장
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log.info(f"시각화 차트 저장 완료: {output_path}")


def write_report(
    metrics: list[CaseMetrics],
    skipped_api: bool,
    hardware_note: str,
    duration_sec: float,
    output_dir: Path,
) -> Path:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report_path = output_dir / REPORT_FILENAME
    visualization_path = output_dir / CHART_FILENAME

    # 마크다운 파일 생성 전 시각화 차트 먼저 생성
    save_visualization_charts(metrics, output_path=visualization_path)

    lines: list[str] = []
    lines.append("# sisicallcall 임베딩 벤치마크 리포트\n")
    lines.append("> 자동 생성 문서 — RAG 및 Semantic Cache 설계를 위한 로컬/API 임베딩 비교\n")
    lines.append(f"\n**생성 시각:** {now}  \n")
    lines.append(f"**전체 실행 시간:** {duration_sec:.1f}s\n")

    lines.append("\n## 1. 테스트 목적 및 환경 요약\n")
    lines.append(
        "실시간 전화 상담 에이전트(sisicallcall)는 **약 5초 이내 응답**을 목표로 합니다. "
        "본 실험은 동일 매뉴얼 텍스트에 대해 임베딩 모델·청크 크기 조합별로 "
        "**GPU 메모리(피크 할당량), 로드·인덱싱·검색 지연, 검색 결과 품질**을 비교합니다.\n"
    )
    lines.append(
        f"\n- **입력:** `{MANUAL_PATH.as_posix().split('/')[-2]}/{MANUAL_PATH.name}` (필수)\n"
        "- **청크 오버랩:** 고정 50자\n"
        "- **벡터 DB:** ChromaDB (거리 metric: cosine)\n"
        "- **VRAM 측정:** 각 단계 후 `torch.cuda.memory_allocated()` 스냅샷과 "
        "`torch.cuda.max_memory_allocated()` 피크 중 **큰 값**을 비교하여 기록했습니다.\n"
        "- **유사도 표기:** Chroma cosine distance \\(d\\) 에 대해 **cosine similarity ≈ 1 − d** 로 환산\n"
    )
    lines.append(f"- **실행 환경 메모:** {hardware_note}\n")
    if skipped_api:
        lines.append(
            "\n> ⚠️ `OPENAI_API_KEY` 가 없어 **OpenAI text-embedding-3-small** 케이스는 건너뛰었습니다. "
            "키 설정 후 재실행하면 API 행이 채워집니다.\n"
        )

    # 대시보드 이미지를 마크다운에 삽입
    lines.append("\n## 📊 2. 시각화 대시보드\n")
    lines.append(f"![벤치마크 시각화 차트]({visualization_path.name})\n")
    lines.append("*※ 각 막대는 Chunk Size(300, 600, 1000)를 의미합니다. (평균/P95 레이턴시, VRAM, 인덱싱, Hit@3, MRR@3)*\n")

    lines.append("\n## 3. 결과 요약 표\n")
    lines.append(
        "| 모델 | Chunk | 로드 | 인덱싱 | 검색 평균 | 검색 P95 | VRAM | 인덱스 크기 | 임베딩 처리량 | Hit@3 | MRR@3 | 토큰 수 | 비용 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"
    )
    for m in metrics:
        tok = str(m.token_count) if m.token_count is not None else "—"
        lines.append(
            f"| {m.model_label} | {m.chunk_size} | {format_load_time(m)} | "
            f"{m.index_time_sec:.2f}s | {m.retrieval_avg_sec:.4f}s | {m.retrieval_p95_sec:.4f}s | "
            f"{format_vram(m)} | {m.index_size_mb:.2f} MB | {format_rate(m.embedding_docs_per_sec)} | "
            f"{m.hit_at_3:.3f} | {m.mrr_at_3:.3f} | {tok} | {m.estimated_cost_usd} |\n"
        )

    lines.append("\n## 4. 추가 지표 요약\n")
    lines.append(
        "- **검색 P95:** 꼬리 지연(느린 구간) 확인용 지표\n"
        "- **인덱스 크기(MB):** 저장소/배포 비용 추정 지표\n"
        "- **임베딩 처리량(docs/s):** 대량 적재 속도 비교 지표\n"
        "- **Hit@3, MRR@3:** Top-3 기준 검색 품질 정량 지표\n"
    )

    lines.append("\n## 5. 쿼리별 검색 품질 (Top-3)\n")
    for qkey, _, kws in TEST_QUERIES:
        lines.append(f"\n### {qkey} — 기대 키워드: {', '.join(kws)}\n")
        lines.append("\n| 모델 | Chunk | Rank(정답 히트) | Rank1 유사도 | Rank1 미리보기 |\n|---|---:|---:|---:|---|\n")
        for m in metrics:
            ranks = m.query_results.get(qkey, [])
            hit_rank: str = "—"
            for row in ranks:
                if doc_matches_keywords(row["text"], kws):
                    hit_rank = str(row["rank"])
                    break
            sim1 = ranks[0]["cosine_similarity"] if ranks else float("nan")
            prev = (ranks[0]["text"][:120] + "…") if ranks else ""
            prev = prev.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {m.model_label} | {m.chunk_size} | {hit_rank} | "
                f"{sim1:.4f} | {prev} |\n"
            )

        lines.append("\n<details>\n<summary>전체 Top-3 원문 및 점수</summary>\n\n")
        for m in metrics:
            lines.append(f"\n#### {m.model_label} — chunk {m.chunk_size}\n")
            for row in m.query_results.get(qkey, []):
                txt = row["text"].replace("\n", " ")[:800]
                lines.append(
                    f"- **Rank {row['rank']}** · sim≈{row['cosine_similarity']:.4f} · dist={row['chroma_distance']:.4f}\n"
                    f"  > {txt}\n"
                )
        lines.append("\n</details>\n")

    lines.append("\n## 6. 종합 결론 및 추천\n")
    if metrics:
        combo, why = recommend_combo(metrics)
        lines.append(
            f"\n**추천 조합(휴리스틱):** **{combo}**\n\n{why}\n\n"
            "- **5초 레이턴시:** 벤치 기준으로는 대부분의 **단일 검색 호출**이 수~수십 ms 수준이었을 가능성이 높습니다. "
            "실서비스에서는 LLM TTFT·TTS·네트워크가 동시에 붙으므로, **end-to-end 예산** 안에서 임베딩·검색 예산을 백로그로 남겨두는 것이 안전합니다.\n"
            "- **16GB RAM / RTX 계열:** 피크 VRAM이 여유롭지 않다면 **청크 크기를 줄이거나** 배치 크기(`EMBED_BATCH_SIZE`)를 낮추고, "
            "가능하면 **양자화 모델** 또는 **더 작은 임베딩 차원**을 후속 실험하세요.\n"
        )
    lines.append(
        "\n---\n\n*이 글은 `embedding_benchmark.py` 실행 결과를 바탕으로 생성되었습니다.*\n"
    )

    report_path.write_text("".join(lines), encoding="utf-8")
    log.info("리포트 저장: %s", report_path)
    return report_path