"""Deterministic workflow implementation for the fault diagnosis demo."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from ..shared.console import SessionConsole
from ..shared.files import ensure_dir, write_lines

from ..artifacts.generators import (
    write_alert_context,
    write_hypothesis_board,
    write_log_bundle,
    write_manifest,
    write_plot_placeholders,
    write_rag_index,
    write_remediation_plan,
    write_report,
    write_synthetic_data,
    write_topology_view,
    write_validation_trace,
)
from .crew import CrewTranscriptBuilder, Hypothesis
from .data import Fixture, FixtureLoader


@dataclass
class FaultDiagnosisSettings:
    project_root: Path
    seed: int = 4242
    session_label: str | None = None
    demo: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "fault_diagnosis" / "fixtures"

    @property
    def artifacts_root(self) -> Path:
        return self.project_root / "artifacts"


class FaultDiagnosisRunner:
    def __init__(self, settings: FaultDiagnosisSettings) -> None:
        self.settings = settings
        self.rng = random.Random(settings.seed)

    def run(self) -> Path:
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        session_name = f"{timestamp}"
        if self.settings.session_label:
            session_name = f"{session_name}_{self.settings.session_label}"

        session_dir = ensure_dir(self.settings.artifacts_root / "sessions" / session_name)
        console = SessionConsole()
        crew = ["Planner", "Retriever", "Reasoner", "Reporter"]
        console.banner("Fault Diagnosis", crew, self.settings.seed, str(session_dir))
        console.stage_update("Launch", "Completed")

        loader = FixtureLoader(self.settings.data_dir)
        fixtures = list(loader.iter_fixtures())

        payloads: Dict[str, object] = {}
        rag_index: List[Dict[str, str]] = []
        console.stage_update("Fixture Replay", "Started")
        for fixture in fixtures:
            console.fixture_progress(fixture.fixture_id, fixture.description)
            content = fixture.load()
            payloads[fixture.fixture_id] = content
            rag_index.append(
                {
                    "id": fixture.fixture_id,
                    "path": str(fixture.path),
                    "description": fixture.description,
                    "preview": self._preview(content),
                }
            )
        rag_index_path = write_rag_index(session_dir, rag_index)
        console.stage_update("Fixture Replay", "Completed")

        alert = self._require_payload(payloads, "FD-ALRT-017")
        console.stage_update("Alert Intake", "Started")
        alert_path = write_alert_context(session_dir, alert)
        console.stage_update("Alert Intake", "Completed")

        console.stage_update("Evidence Sweep", "Started")
        log_bundle_path = write_log_bundle(session_dir, alert)
        topology = self._require_payload(payloads, "topology_view_05")
        topology_path = write_topology_view(session_dir, topology)
        console.stage_update("Evidence Sweep", "Completed")

        incident = self._require_payload(payloads, "incident_core_011")
        runbook_raw = payloads.get("runbook_rf_002")
        runbook = runbook_raw if isinstance(runbook_raw, str) else str(runbook_raw)
        kpis = self._require_payload(payloads, "kpi_rollup_may")

        hypotheses = self._build_hypotheses(alert, incident, topology)
        console.stage_update("Hypothesis Board", "Started")
        board_path = write_hypothesis_board(session_dir, hypotheses)
        seed_fixture = fixtures[0].fixture_id if fixtures else "FD-ALRT-017"
        transcript_lines = CrewTranscriptBuilder().build(hypotheses, seed_fixture)
        console.stream(transcript_lines)
        console.stage_update("Hypothesis Board", "Completed")

        console.stage_update("Validation Loop", "Started")
        validation_path = write_validation_trace(session_dir, hypotheses)
        console.stage_update("Validation Loop", "Completed")

        console.stage_update("Resolution Snapshot", "Started")
        remediation_plan = self._build_remediation_plan(runbook, incident)
        remediation_path = write_remediation_plan(session_dir, remediation_plan)
        console.stage_update("Resolution Snapshot", "Completed")

        console.stage_update("Wrap-Up", "Started")
        report_paths = write_report(
            session_dir,
            hypotheses,
            citations=["incident_core_011.json", "runbook_rf_002.md", "traffic_probe_agent.log"],
        )
        plot_paths = write_plot_placeholders(self.settings.artifacts_root / "plots")
        synthetic_payload = {
            "alert": alert,
            "incident": incident,
            "topology": topology,
            "kpis": kpis,
            "session": session_name,
        }
        synthetic_path = write_synthetic_data(self.settings.artifacts_root / "data", session_name, synthetic_payload)
        artifact_paths = [
            rag_index_path,
            alert_path,
            log_bundle_path,
            topology_path,
            board_path,
            validation_path,
            remediation_path,
            *report_paths,
            synthetic_path,
            *plot_paths,
        ]
        session_log_path = session_dir / "session.log"
        manifest_payload = [*artifact_paths, session_log_path]
        manifest_path = write_manifest(session_dir, manifest_payload)
        manifest_payload.append(manifest_path)
        console.stage_update("Wrap-Up", "Completed")

        # Echo manifest for the operator
        console.stream(["", "Artifacts generated:"])
        for path in manifest_payload:
            console.stream([f" - {path}"])

        write_lines(session_log_path, console.lines)

        return session_dir

    def _build_hypotheses(
        self,
        alert: Dict[str, object],
        incident: Dict[str, object],
        topology: Dict[str, object],
    ) -> List[Hypothesis]:
        confidence_primary = 0.82 + self.rng.random() * 0.03
        confidence_secondary = 0.55 + self.rng.random() * 0.05
        confidence_tertiary = 0.30 + self.rng.random() * 0.04
        hypotheses = [
            Hypothesis(
                index=1,
                statement=(
                    "RAN congestion across sector 12A is driving packet loss; align with prior incident "
                    f"{incident.get('incident_id', 'INC-CORE-011')} mitigation playbook."
                ),
                confidence=confidence_primary,
                verdict="Grounded",
                citation="incident_core_011.json",
            ),
            Hypothesis(
                index=2,
                statement="Backhaul degradation on fiber route to MME-CLSTR-3 could explain latency uplift.",
                confidence=confidence_secondary,
                verdict="Retry",
                citation="topology_snapshot.json",
            ),
            Hypothesis(
                index=3,
                statement="QoS misconfiguration on SGW-CORE-2 is the primary driver of congestion.",
                confidence=confidence_tertiary,
                verdict="Rejected",
                citation="change_manager_audit.log",
            ),
        ]
        return hypotheses

    @staticmethod
    def _build_remediation_plan(runbook: object, incident: Dict[str, object]) -> List[str]:
        steps = [
            "- Validate traffic probe telemetry remains stable for the impacted sector.",
            "- Rebalance carriers per runbook_rf_002 step 4 if congestion persists >10 minutes.",
            "- Notify change_manager to coordinate with maintenance window CHG-5567.",
            "- Prepare stakeholder summary referencing incident "
            f"{incident.get('incident_id', 'INC-CORE-011')} for knowledge-base updates.",
        ]
        if isinstance(runbook, str):
            steps.append("- Cross-check demo remediation notes with runbook excerpt:\n\n" + runbook.strip().split("\n", 1)[0])
        return steps

    @staticmethod
    def _preview(content: object, limit: int = 120) -> str:
        try:
            text = json.dumps(content, ensure_ascii=False)
        except TypeError:
            text = str(content)
        return text[:limit]

    @staticmethod
    def _require_payload(payloads: Dict[str, object], key: str) -> Dict[str, object]:
        payload = payloads.get(key)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Fixture {key} missing or not JSON payload in demo data")
        return payload


__all__ = ["FaultDiagnosisRunner", "FaultDiagnosisSettings"]
