"""Convenience script to POST data/campaigns_raw.json to a running API.

Usage:
    python seed.py                       # uses http://localhost:8000
    python seed.py http://host:8000 data/campaigns_raw.json
"""
import json
import sys

import httpx


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    path = sys.argv[2] if len(sys.argv) > 2 else "data/campaigns_raw.json"

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    resp = httpx.post(f"{base}/campaigns/ingest", json=payload, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))


if __name__ == "__main__":
    main()
