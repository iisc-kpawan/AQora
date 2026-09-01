from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
from urllib.parse import urlencode

from ..errors import IngestionError


def fetch_cnemc(start: date, end: date, destination: Path, url_template: str, *, dry_run: bool = False) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    current = start
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        while current <= end:
            url = url_template.format(date=current.strftime("%Y%m%d"))
            path = destination / f"china_sites_{current:%Y%m%d}.csv"
            if not dry_run:
                response = client.get(url)
                if response.status_code == 404:
                    current += timedelta(days=1)
                    continue
                response.raise_for_status()
                if not response.content.startswith(b"date,hour,type"):
                    raise IngestionError(f"Unexpected CNEMC artifact at {url}")
                path.write_bytes(response.content)
            outputs.append(path)
            current += timedelta(days=1)
    return outputs


def fetch_cpcb_year(artifact_names: list[str], year: int, destination: Path, base_url: str, *, dry_run: bool = False) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    with httpx.Client(timeout=120, follow_redirects=True, verify=True) as client:
        for artifact in artifact_names:
            remote = f"Raw_data/15Min/{year}/{artifact}"
            url = base_url + "?" + urlencode({"file_name": remote})
            path = destination / f"{year}_{artifact}"
            if not dry_run:
                response = client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                first_line = response.content.splitlines()[0] if response.content else b""
                if b"Timestamp" not in first_line:
                    raise IngestionError(f"Unexpected CPCB artifact at {url}")
                path.write_bytes(response.content)
            outputs.append(path)
    return outputs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
