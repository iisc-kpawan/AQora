"""Export a one-column public benchmark from the private benchmark JSONL."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def export_questions(source: Path, destination: Path) -> int:
    questions: list[str] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        question = record.get("Question") or record.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Missing question on JSONL line {line_number}")
        questions.append(question.strip())

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question"])
        writer.writeheader()
        writer.writerows({"question": question} for question in questions)
    return len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("evaluation/benchmark_300.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/aqa_eval_300_questions.csv"))
    args = parser.parse_args()
    count = export_questions(args.input, args.output)
    print(f"Exported {count} questions to {args.output}")


if __name__ == "__main__":
    main()

