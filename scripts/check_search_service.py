#!/usr/bin/env python3
"""Fail fast when the SearchQA retrieval service is unavailable or malformed."""

import argparse
import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "query": "What is the capital of France?",
            "topk": 1,
            "return_scores": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = response.read()
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
    ) as exc:
        print(
            f"ERROR: SearchQA retrieval service is unavailable at {args.url}: {exc}",
            file=sys.stderr,
        )
        print(
            "Start examples/search/retriever/retrieval_launch.sh first, "
            "or set SEARCH_URL to the correct /retrieve endpoint.",
            file=sys.stderr,
        )
        return 1

    try:
        decoded = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(
            f"ERROR: retrieval service returned invalid JSON: {exc}",
            file=sys.stderr,
        )
        return 1

    result = decoded.get("result") if isinstance(decoded, dict) else None
    if (
        not isinstance(result, list)
        or not result
        or not isinstance(result[0], list)
        or not result[0]
    ):
        print(
            "ERROR: retrieval service functional check returned no documents; "
            f"got: {decoded!r}",
            file=sys.stderr,
        )
        return 1

    print(f"SearchQA retrieval service is ready: {args.url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
