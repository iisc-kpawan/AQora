"""Run the public questions-only benchmark against an AQora API."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import httpx


def load_questions(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "question" not in rows[0]:
        raise ValueError("Benchmark must contain a column named 'question'")
    return [row["question"].strip() for row in rows if row["question"].strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark/aqa_eval_300_questions.csv"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("evaluation-results.jsonl"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    questions = load_questions(args.input)
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=180) as client, args.output.open(
        "w", encoding="utf-8"
    ) as output:
        for run in range(1, args.runs + 1):
            for index, question in enumerate(questions, 1):
                payload = {"question": question, "provider": args.provider, "model": args.model}
                started = time.perf_counter()
                try:
                    response = client.post("/api/chat", json=payload)
                    latency_ms = round((time.perf_counter() - started) * 1000, 3)
                    body = response.json()
                    trace = body.get("model_trace", {}) if isinstance(body, dict) else {}
                    record = {
                        "question_number": index,
                        "run": run,
                        "provider": args.provider,
                        "model": args.model,
                        "status_code": response.status_code,
                        "completed": response.is_success,
                        "latency_ms": latency_ms,
                        "prompt_tokens": trace.get("prompt_tokens"),
                        "completion_tokens": trace.get("completion_tokens"),
                        "response": body,
                    }
                except Exception as exc:
                    record = {
                        "question_number": index,
                        "run": run,
                        "provider": args.provider,
                        "model": args.model,
                        "completed": False,
                        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                        "error": str(exc),
                    }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()

    print(json.dumps({"questions": len(questions), "runs": args.runs, "output": str(args.output)}))


if __name__ == "__main__":
    main()

