"""Crew scripts for the deterministic demo run."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class Hypothesis:
    index: int
    statement: str
    confidence: float
    verdict: str
    citation: str


@dataclass
class CrewMessage:
    speaker: str
    text: str

    def render(self) -> str:
        return f"[{self.speaker}] {self.text}"


class CrewTranscriptBuilder:
    """Constructs a deterministic transcript for the demo run."""

    def build(self, hypotheses: Sequence[Hypothesis], seed_fixture_id: str) -> List[str]:
        messages: List[CrewMessage] = [
            CrewMessage("Planner", f"Seeding alert context from fixture {seed_fixture_id}"),
            CrewMessage("Retriever", "Indexed fixtures → runbook_rf_002.md, incident_core_011.json"),
        ]
        for hypothesis in hypotheses:
            if hypothesis.verdict == "Grounded":
                verdict = "grounded ✔"
            elif hypothesis.verdict == "Retry":
                verdict = "retry ✖"
            elif hypothesis.verdict == "Rejected":
                verdict = "rejected ✖"
            else:
                verdict = hypothesis.verdict
            messages.append(
                CrewMessage(
                    "Reasoner",
                    f"Hypothesis {hypothesis.index} ({hypothesis.confidence:.2f} confidence) {verdict} via {hypothesis.citation}",
                )
            )
            if hypothesis.index == 1:
                messages.append(
                    CrewMessage(
                        "Validator: traffic_probe_agent",
                        "Packet loss spike confirmed (p95 latency 180ms)",
                    )
                )
            elif hypothesis.index == 2:
                messages.append(
                    CrewMessage(
                        "Validator: config_diff_checker",
                        "No drift detected across recent change set",
                    )
                )
        messages.append(
            CrewMessage(
                "Reporter",
                "Drafting remediation steps → artifacts/.../fault_diagnosis_report.html",
            )
        )
        return [message.render() for message in messages]


__all__ = ["CrewTranscriptBuilder", "CrewMessage", "Hypothesis"]
