from __future__ import annotations

import argparse
import json
from datetime import datetime

from .agent import ConversationalAgent
from .api import main as serve
from .catalog import Catalog
from .evaluation import evaluate_file
from .ingestion import IngestionPipeline
from .queryspec import QuerySpec


def main() -> None:
    parser = argparse.ArgumentParser(prog="aqa")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("catalog")
    query = commands.add_parser("query"); query.add_argument("question")
    commands.add_parser("serve")
    evaluate = commands.add_parser("evaluate"); evaluate.add_argument("input"); evaluate.add_argument("--provider"); evaluate.add_argument("--model")
    ingest = commands.add_parser("ingest"); ingest.add_argument("dataset", choices=["CPCB", "CNEMC"]); ingest.add_argument("--source"); ingest.add_argument("--start"); ingest.add_argument("--end"); ingest.add_argument("--latest", action="store_true"); ingest.add_argument("--dry-run", action="store_true"); ingest.add_argument("files", nargs="*")
    args = parser.parse_args()
    if args.command == "catalog":
        catalog = Catalog(); print(json.dumps({"stations": len(catalog.stations), "series": len(catalog.series), "path": str(catalog.persist())}, indent=2))
    elif args.command == "query":
        print(json.dumps(ConversationalAgent().ask(args.question), ensure_ascii=False, indent=2))
    elif args.command == "serve":
        serve()
    elif args.command == "evaluate":
        print(json.dumps(evaluate_file(args.input, provider=args.provider, model=args.model), indent=2))
    else:
        pipeline = IngestionPipeline()
        start = _parse_datetime(args.start) if args.start else None
        end = _parse_datetime(args.end, end_of_day=True) if args.end else None
        if args.files:
            if not start or not end:
                parser.error("--start and --end are required when ingesting local files")
            source = args.source or pipeline.config["datasets"][args.dataset]["source_priority"][0]
            result = pipeline.ingest_files(args.dataset, source, args.files, start=start, end=end, dry_run=args.dry_run)
        else:
            if not args.latest and (not start or not end):
                parser.error("use --latest or supply --start and --end")
            result = pipeline.fetch_and_ingest(args.dataset, start=start, end=end, dry_run=args.dry_run)
        print(json.dumps(result, indent=2))


def _parse_datetime(value: str, *, end_of_day: bool = False) -> datetime:
    parsed = datetime.fromisoformat(value)
    if end_of_day and len(value) == 10:
        return parsed.replace(hour=23, minute=59, second=59)
    return parsed


if __name__ == "__main__":
    main()
