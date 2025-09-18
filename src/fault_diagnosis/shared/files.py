"""Filesystem helpers for writing deterministic artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            if not line.endswith("\n"):
                handle.write("\n")


__all__ = ["ensure_dir", "write_json", "write_text", "write_lines"]
