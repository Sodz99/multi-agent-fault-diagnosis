"""Console helpers for the telecom_ops CLI."""
from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Sequence


class SessionConsole:
    """Utility for printing to stdout while recording lines for the session log."""

    def __init__(self) -> None:
        self._lines: List[str] = []

    @property
    def lines(self) -> List[str]:
        return self._lines

    def _record(self, text: str) -> None:
        self._lines.append(text)
        print(text)

    def banner(self, workflow: str, crew: Sequence[str], seed: int, session_path: str) -> None:
        self._record(f"Workflow: {workflow} | Crew: {', '.join(crew)}")
        self._record(f"Deterministic seed: {seed} | Session: {session_path}")
        self._record("")

    def fixture_progress(self, fixture_id: str, description: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._record(f"[{timestamp}] Loading fixture {fixture_id}: {description}")

    def stage_update(self, stage: str, status: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._record(f"[{timestamp}] {stage:<20} {status}")

    def stream(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._record(line)


__all__ = ["SessionConsole"]
