#!/usr/bin/env python3
"""Summarize JSON probe reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPORT_DIR = Path("reports")


def iter_cases(group: Any):
    if isinstance(group, dict) and "status" in group:
        yield group
    elif isinstance(group, dict):
        for value in group.values():
            yield from iter_cases(value)


def summarize_report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    env = metadata.get("environment", {})
    cases = list(iter_cases(data.get("tests", {})))
    modules = data.get("modules", {})
    present_modules = sorted(
        name for name, info in modules.items()
        if isinstance(info, dict) and info.get("status") == "ok"
    )
    return {
        "file": path.name,
        "stage": data.get("stage", "?"),
        "device": data.get("device", "?"),
        "sandbox": env.get("CODEX_SANDBOX") or "none",
        "mps": metadata.get("mps_available", "?"),
        "ok": sum(1 for case in cases if case.get("status") == "ok"),
        "native": sum(1 for case in cases if case.get("impl") == "native"),
        "cpu-assist": sum(1 for case in cases if case.get("impl") == "cpu-assisted"),
        "skipped": sum(1 for case in cases if case.get("status") == "skipped"),
        "unsupported": sum(1 for case in cases if case.get("status") == "unsupported"),
        "failed": sum(1 for case in cases if case.get("status") == "failed"),
        "modules": ",".join(present_modules),
    }


def main() -> int:
    paths = sorted(REPORT_DIR.glob("*.json"))
    if not paths:
        print("No reports found.")
        return 0

    rows = [summarize_report(path) for path in paths]
    headers = [
        "stage",
        "device",
        "sandbox",
        "mps",
        "ok",
        "native",
        "cpu-assist",
        "skipped",
        "unsupported",
        "failed",
        "modules",
        "file",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
