"""Data loading utilities for the fault diagnosis workflow."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


@dataclass
class Fixture:
    fixture_id: str
    path: Path
    description: str

    def load(self) -> Any:
        if self.path.suffix.lower() == ".json":
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return self.path.read_text(encoding="utf-8")


class FixtureLoader:
    """Loads demo fixtures from the data directory."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.manifest_path = base_path / "fixtures_manifest.json"

    def iter_fixtures(self) -> Iterable[Fixture]:
        manifest = self._load_manifest()
        for entry in manifest.get("fixtures", []):
            path = self.base_path / entry["path"]
            yield Fixture(
                fixture_id=entry["id"],
                path=path,
                description=entry.get("description", ""),
            )

    def load_index(self) -> List[Dict[str, Any]]:
        index: List[Dict[str, Any]] = []
        for fixture in self.iter_fixtures():
            content = fixture.load()
            index.append(
                {
                    "id": fixture.fixture_id,
                    "path": str(fixture.path),
                    "description": fixture.description,
                    "kind": fixture.path.suffix.lstrip("."),
                    "preview": self._preview(content),
                }
            )
        return index

    def _load_manifest(self) -> Dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _preview(content: Any, limit: int = 120) -> str:
        text = json.dumps(content, ensure_ascii=False) if isinstance(content, (dict, list)) else str(content)
        return text[:limit]


__all__ = ["Fixture", "FixtureLoader"]
