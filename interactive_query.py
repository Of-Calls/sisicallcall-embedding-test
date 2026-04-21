from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
import urllib.error
import urllib.request

SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_CHUNK_SIZE = 600
DEFAULT_TOP_K = 3
REPORTS_DIR = Path("reports")


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def print_result_block(one: dict) -> None:
    status = one.get("status")
    model_id = one.get("model_id", "unknown")
    chunk_size = one.get("chunk_size", "?")
    if status != "ok":
        print("=" * 72)
        print(f"[실패] {model_id} | chunk={chunk_size}")
        print(f"- error_type: {one.get('error_type')}")
        print(f"- error_message: {one.get('error_message')}")
        print("=" * 72)
        return

    print("=" * 72)
    print(f"[모델] {model_id} | chunk={chunk_size} | latency={one.get('latency_ms')} ms")
    print(f"[질문] {one.get('question')}")
    print("")
    print("Top-k 결과")
    for row in one.get("results", []):
        preview = " ".join(str(row.get("text", "")).split())
        preview = preview[:150] + ("..." if len(preview) > 150 else "")
        print(
            f"{row.get('rank')}) {row.get('chunk_id')} | "
            f"sim={row.get('cosine_similarity')} | text=\"{preview}\""
        )
    print("=" * 72)


def init_markdown_report(chunk_size: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPORTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "interactive_query_report.md"
    header = [
        "# Interactive Query Report\n\n",
        f"- generated_at: {datetime.now().isoformat(timespec='seconds')}\n",
        f"- server_url: {SERVER_URL}\n",
        f"- chunk_size: {chunk_size}\n",
        f"- top_k: {DEFAULT_TOP_K}\n\n",
        "---\n\n",
    ]
    md_path.write_text("".join(header), encoding="utf-8")
    return md_path


def append_markdown_result(md_path: Path, question: str, all_results: list[dict]) -> None:
    lines: list[str] = []
    lines.append(f"## 질문: {question}\n\n")
    for one in all_results:
        model_id = one.get("model_id", "unknown")
        chunk_size = one.get("chunk_size", "?")
        status = one.get("status")
        lines.append(f"### {model_id} | chunk={chunk_size} | status={status}\n")
        if status != "ok":
            lines.append(f"- error_type: `{one.get('error_type')}`\n")
            lines.append(f"- error_message: {one.get('error_message')}\n\n")
            continue
        lines.append(f"- latency_ms: {one.get('latency_ms')}\n")
        for row in one.get("results", []):
            preview = " ".join(str(row.get("text", "")).split())
            preview = preview[:180] + ("..." if len(preview) > 180 else "")
            lines.append(
                f"- rank {row.get('rank')} | {row.get('chunk_id')} | "
                f"sim={row.get('cosine_similarity')}\n"
            )
            lines.append(f"  - {preview}\n")
        lines.append("\n")
    lines.append("---\n\n")
    with md_path.open("a", encoding="utf-8") as f:
        f.write("".join(lines))


def main() -> None:
    chunk_size = DEFAULT_CHUNK_SIZE

    save_md_raw = input("질문/결과를 markdown으로 저장할까요? (y/N): ").strip().lower()
    save_md = save_md_raw == "y"
    md_path: Path | None = None
    if save_md:
        md_path = init_markdown_report(chunk_size)
        print(f"리포트 저장 경로: {md_path}")

    print("질문을 입력하세요. 종료하려면 빈 줄 입력.\n")
    while True:
        question = input("질문> ").strip()
        if not question:
            print("종료합니다.")
            break
        payload = {"question": question, "chunk_size": chunk_size, "k": DEFAULT_TOP_K}
        try:
            out = post_json(f"{SERVER_URL}/query_all", payload)
        except urllib.error.URLError as e:
            print(f"서버 연결 실패: {e}")
            print("먼저 `uvicorn server:app --host 0.0.0.0 --port 8000` 실행하세요.")
            break
        except Exception as e:
            print(f"요청 실패: {e}")
            continue

        if out.get("status") != "ok":
            print(out)
            continue
        results = out.get("results", [])
        for one in results:
            print_result_block(one)
        if save_md and md_path is not None:
            append_markdown_result(md_path, question, results)


if __name__ == "__main__":
    main()
