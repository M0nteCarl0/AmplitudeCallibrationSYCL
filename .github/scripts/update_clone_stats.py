#!/usr/bin/env python3
"""Accumulate GitHub clone traffic into a shields.io endpoint badge payload.

The GitHub Traffic API exposes only the last 14 days and is readable only with
push access, so the window is merged into a lifetime per-day history that is
published to the orphan ``traffic`` branch by
``.github/workflows/clone-stats.yml``:

* ``clones-daily.json`` - accumulated per-day history and lifetime totals
  (source of truth, never shrinks).
* ``clones.json`` - shields.io endpoint payload rendered by the README badge.

Stdlib only. ``--traffic-file`` / ``--history-file`` replace the two API reads
with local JSON fixtures, which allows an offline dry run.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NoReturn

API_ROOT = "https://api.github.com"
BRANCH = "traffic"
HISTORY_PATH = "clones-daily.json"
BADGE_PATH = "clones.json"
BADGE_LABEL = "clones"
BADGE_COLOR = "blue"
BADGE_CACHE_SECONDS = 3600
TOKEN_ENV = "GH_TRAFFIC_TOKEN"


class ApiError(RuntimeError):
    """Non-404 response from the GitHub API."""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} from {url}: {body.strip()[:400]}")
        self.status = status


def fail(message: str) -> NoReturn:
    """Report a fatal problem (as a GitHub annotation when running in Actions)."""
    prefix = "::error title=clone statistics::" if os.environ.get("GITHUB_ACTIONS") else "error: "
    print(prefix + message, file=sys.stderr)
    raise SystemExit(1)


def get_json(url: str, token: str) -> tuple[int, Any]:
    """GET ``url`` and decode the JSON body; returns ``(404, None)`` when missing."""
    request = urllib.request.Request(url)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "clone-statistics-workflow")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return 404, None
        raise ApiError(error.code, url, error.read().decode("utf-8", "replace")) from error


def read_json_file(path: Path) -> Any:
    """Read a JSON fixture (tolerating the BOM a Windows editor may add)."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        fail(f"cannot read JSON from {path}: {error}")


def traffic_from_api(repo: str, token: str) -> list[dict[str, Any]]:
    url = f"{API_ROOT}/repos/{repo}/traffic/clones?per=day"
    try:
        _, payload = get_json(url, token)
    except ApiError as error:
        if error.status in (401, 403):
            fail(
                f"cannot read the traffic API ({error}). The token in {TOKEN_ENV} must have push "
                "access to the repository: a classic token with the 'repo' scope, or a "
                "fine-grained token with 'Contents: read' and 'Administration: read'."
            )
        fail(f"cannot read the traffic API: {error}")
    buckets = (payload or {}).get("clones")
    if not isinstance(buckets, list):
        fail(f"unexpected traffic API response, no 'clones' array: {payload!r}")
    return buckets


def history_from_api(repo: str, token: str) -> dict[str, Any] | None:
    """Read the published history from the ``traffic`` branch (``None`` on first run)."""
    url = f"{API_ROOT}/repos/{repo}/contents/{HISTORY_PATH}?ref={BRANCH}"
    try:
        status, payload = get_json(url, token)
    except ApiError as error:
        fail(f"cannot read the published history: {error}")
    if status == 404 or not payload:
        return None
    try:
        return json.loads(base64.b64decode(payload["content"]))
    except (KeyError, ValueError) as error:
        fail(f"published history {HISTORY_PATH} is not readable JSON: {error}")


def merge_days(
    previous: dict[str, dict[str, int]], clones: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, int]], int]:
    """Merge the 14-day window into the accumulated history.

    GitHub reports every UTC day once, and a day already seen may come back with
    a larger (final) value. Keeping the maximum per day makes repeated runs
    idempotent instead of double counting.
    """
    merged = {day: dict(values) for day, values in previous.items()}
    added = 0
    for bucket in clones:
        day = str(bucket.get("timestamp", ""))[:10]
        if len(day) != 10:
            fail(f"unexpected traffic bucket without a timestamp: {bucket!r}")
        try:
            count, uniques = int(bucket.get("count") or 0), int(bucket.get("uniques") or 0)
        except (TypeError, ValueError):
            fail(f"unexpected traffic bucket with non-numeric counts: {bucket!r}")
        if day not in merged:
            added += 1
        current = merged.setdefault(day, {"count": 0, "uniques": 0})
        current["count"] = max(current["count"], count)
        current["uniques"] = max(current["uniques"], uniques)
    return dict(sorted(merged.items())), added


def totals(days: dict[str, dict[str, int]]) -> tuple[int, int]:
    return (
        sum(entry["count"] for entry in days.values()),
        sum(entry["uniques"] for entry in days.values()),
    )


def validate_history(document: Any) -> dict[str, dict[str, int]]:
    """Return the published per-day history, aborting if it is malformed.

    The branch is generated, but a hand-edited or truncated file must never be
    merged into and silently republished as if it were sound data.
    """
    days = document.get("days") if isinstance(document, dict) else None
    if not isinstance(days, dict):
        fail(f"published history {HISTORY_PATH} has no 'days' mapping; refusing to overwrite it")
    validated: dict[str, dict[str, int]] = {}
    for day, entry in days.items():
        try:
            datetime.strptime(str(day), "%Y-%m-%d")
            count, uniques = int(entry["count"]), int(entry["uniques"])
        except (KeyError, TypeError, ValueError):
            fail(f"published history {HISTORY_PATH} entry {day!r} is malformed: {entry!r}")
        if count < 0 or uniques < 0:
            fail(f"published history {HISTORY_PATH} entry {day!r} has negative counts: {entry!r}")
        validated[str(day)] = {"count": count, "uniques": uniques}
    return validated


def write_outputs(out_dir: Path, days: dict[str, dict[str, int]], generated: str) -> tuple[int, int]:
    total, total_uniques = totals(days)
    history = {
        "schemaVersion": 1,
        "generated": generated,
        "total": total,
        "total_uniques": total_uniques,
        "days": days,
    }
    badge = {
        "schemaVersion": 1,
        "label": BADGE_LABEL,
        "message": str(total),
        "color": BADGE_COLOR,
        "cacheSeconds": BADGE_CACHE_SECONDS,
    }
    for relative, document in ((HISTORY_PATH, history), (BADGE_PATH, badge)):
        target = out_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return total, total_uniques


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", required=True, type=Path, help="directory to write the branch files into")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name (default: $GITHUB_REPOSITORY)")
    parser.add_argument("--token", default=os.environ.get(TOKEN_ENV), help=f"PAT with traffic access (default: ${TOKEN_ENV})")
    parser.add_argument("--traffic-file", type=Path, help="use this traffic JSON fixture instead of the API")
    parser.add_argument("--history-file", type=Path, help="use this history JSON fixture instead of the API")
    args = parser.parse_args(argv)

    if args.traffic_file:
        clones = read_json_file(args.traffic_file)
        clones = list(clones.get("clones") or clones) if isinstance(clones, dict) else list(clones)
    else:
        if not args.repo:
            fail("GITHUB_REPOSITORY is not set; pass --repo owner/name")
        if not args.token:
            fail(
                f"{TOKEN_ENV} is not set. Create a personal access token that can read the traffic "
                "endpoints (classic: 'repo' scope; fine-grained: 'Contents: read' + "
                f"'Administration: read') and store it as the {TOKEN_ENV} repository secret."
            )
        clones = traffic_from_api(args.repo, args.token)

    if args.history_file:
        previous = read_json_file(args.history_file) if args.history_file.is_file() else None
    elif args.token and args.repo:
        previous = history_from_api(args.repo, args.token)
    else:
        previous = None

    if previous is None:
        previous_days: dict[str, dict[str, int]] = {}
    else:
        previous_days = validate_history(previous)
    previous_total = totals(previous_days)[0]

    days, added = merge_days(previous_days, clones)
    total, total_uniques = totals(days)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_outputs(args.output, days, generated)

    print(
        f"traffic window: {len(clones)} day(s); stored days: {len(days)} (+{added}); "
        f"clones: {previous_total} -> {total}; unique clones (summed per day): {total_uniques}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
