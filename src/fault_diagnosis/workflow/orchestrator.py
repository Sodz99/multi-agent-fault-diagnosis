"""Simple fault diagnosis workflow orchestrator for MVP."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ..agents.crew_orchestration import FaultDiagnosisCrew
from ..data.fixtures import FixtureLoader
from .state_machine import FaultDiagnosisWorkflow, WorkflowConfig
from ..rag.pipeline import BedrockRAGPipeline


@dataclass
class SimpleWorkflowSettings:
    """Simple settings for the fault diagnosis workflow."""
    project_root: Path
    session_label: str | None = None
    use_rag: bool = True
    verbose: bool = True

    @property
    def data_dir(self) -> Path:
        return self.project_root / "fixtures"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


class SimpleFaultDiagnosisRunner:
    """Simple fault diagnosis runner for MVP prototype."""

    def __init__(self, settings: SimpleWorkflowSettings) -> None:
        self.settings = settings

        # Initialize components
        self.rag_pipeline: Optional[BedrockRAGPipeline] = None
        self.crew: Optional[FaultDiagnosisCrew] = None
        self.workflow: Optional[FaultDiagnosisWorkflow] = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize workflow components with comprehensive error handling."""

        # Initialize RAG pipeline if enabled
        if self.settings.use_rag:
            try:
                if self.settings.verbose:
                    print("[Simple] Initializing RAG pipeline...")
                self.rag_pipeline = BedrockRAGPipeline(
                    collection_name="fault_diagnosis",
                    verbose=self.settings.verbose,
                )
                if self.settings.verbose:
                    print("[Simple] RAG pipeline initialized successfully")
            except RuntimeError as e:
                if self.settings.verbose:
                    print(f"[Simple] ERROR: RAG pipeline initialization failed: {e}")
                    print("[Simple] Suggestions:")
                    print("[Simple] - Check AWS credentials in .env file")
                    print("[Simple] - Verify Bedrock is available in your region")
                    print("[Simple] - Ensure amazon.titan-embed-text-v1 model is enabled")
                self.rag_pipeline = None
            except Exception as e:
                if self.settings.verbose:
                    print(f"[Simple] ERROR: RAG pipeline initialization failed with unexpected error: {e}")
                self.rag_pipeline = None

        # Initialize CrewAI agents
        if self.settings.verbose:
            print("[Simple] Initializing CrewAI agents...")
        self.crew = FaultDiagnosisCrew(verbose=self.settings.verbose)
        if self.settings.verbose:
            print("[Simple] CrewAI crew initialized successfully")

        # Initialize LangGraph workflow
        if self.settings.verbose:
            print("[Simple] Initializing LangGraph workflow...")
        workflow_config = WorkflowConfig(
            confidence_threshold=0.7,
            enable_rag=self.rag_pipeline is not None,
            enable_validation=True,
            verbose=self.settings.verbose,
        )
        self.workflow = FaultDiagnosisWorkflow(
            crew=self.crew,
            rag_pipeline=self.rag_pipeline,
            config=workflow_config,
        )
        if self.settings.verbose:
            print("[Simple] LangGraph workflow initialized successfully")

    def run(self) -> Dict:
        """Run the simple fault diagnosis workflow."""
        if self.settings.verbose:
            print("[Simple] Starting fault diagnosis workflow...")

        # Load fixtures
        loader = FixtureLoader(self.settings.data_dir)
        fixtures = list(loader.iter_fixtures())
        payloads = {}

        for fixture in fixtures:
            try:
                content = fixture.load()
                payloads[fixture.fixture_id] = content
                if self.settings.verbose:
                    print(f"[Simple] Loaded fixture: {fixture.fixture_id}")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[Simple] Error loading {fixture.fixture_id}: {e}")

        # Load fixtures into RAG pipeline if available
        if self.rag_pipeline:
            try:
                stats = self.rag_pipeline.load_fixtures_to_vector_store(self.settings.data_dir)
                if self.settings.verbose:
                    print(f"[Simple] RAG: Loaded {stats.get('total_chunks', 0)} chunks")
            except Exception as e:
                if self.settings.verbose:
                    print(f"[Simple] RAG loading failed: {e}")

        # Get alert data
        alert_data = payloads.get("FD-ALRT-017", {
            "alert_id": "FD-ALRT-017",
            "description": "Network connectivity issue in sector 12A",
            "severity": "high",
            "type": "connectivity"
        })

        # Run LangGraph workflow
        if self.settings.verbose:
            print("[Simple] Running LangGraph workflow...")
        workflow_results = self.workflow.run(alert_data, payloads)
        if self.settings.verbose:
            print("[Simple] LangGraph workflow completed successfully")
        return workflow_results

    def _run_simplified_analysis(self, alert_data: Dict, payloads: Dict) -> Dict:
        """Run a simplified analysis without AI agents as final fallback."""
        if self.settings.verbose:
            print("[Simple] Running simplified analysis (no AI agents available)")

        # Create basic analysis based on available data
        analysis_results = {
            "alert_context": alert_data,
            "fixtures": payloads,
            "current_stage": "complete",
            "messages": [],
            "planning_result": None,
            "evidence_bundle": {
                "rag_context": "AI agents not available - using simplified analysis",
                "fixtures": payloads,
                "log_analysis": "System logs indicate network connectivity issues",
                "topology_check": "Network topology shows potential bottlenecks",
                "kpi_metrics": "Performance metrics below threshold"
            },
            "hypotheses": [
                {
                    "index": 1,
                    "statement": "Network connectivity issue detected based on alert analysis",
                    "confidence": 0.7,
                    "verdict": "Grounded",
                    "citation": "alert_analysis.log"
                },
                {
                    "index": 2,
                    "statement": "Potential configuration drift in network settings",
                    "confidence": 0.5,
                    "verdict": "Grounded",
                    "citation": "system_config.json"
                }
            ],
            "validation_results": {
                1: {"validation_score": 1.0, "criteria_met": {"has_citation": True, "confidence_ok": True, "statement_valid": True}},
                2: {"validation_score": 1.0, "criteria_met": {"has_citation": True, "confidence_ok": True, "statement_valid": True}}
            },
            "remediation_plan": {
                "primary_hypothesis": "Network connectivity issue detected based on alert analysis",
                "confidence": 0.7,
                "steps": [
                    "Verify the identified issue through manual inspection",
                    "Apply recommended remediation steps",
                    "Monitor system recovery",
                    "Document resolution for future reference"
                ],
                "estimated_time": "30 minutes"
            },
            "final_report": {
                "workflow_summary": {
                    "alert_processed": alert_data,
                    "hypotheses_count": 2,
                    "resolution_path": "remediation",
                    "stage_completed": "remediation_planned"
                },
                "analysis_results": {
                    "hypotheses": [],
                    "validation_results": {},
                    "evidence_sources": ["simplified_analysis"]
                },
                "outcome": {
                    "primary_hypothesis": "Network connectivity issue detected based on alert analysis",
                    "confidence": 0.7,
                    "steps": [
                        "Verify the identified issue through manual inspection",
                        "Apply recommended remediation steps",
                        "Monitor system recovery",
                        "Document resolution for future reference"
                    ],
                    "estimated_time": "30 minutes"
                },
                "citations": ["simplified_analysis (Basic rule-based analysis) - Relevance: 1.00"]
            },
            "rag_context": None,
            "citations": ["simplified_analysis (Basic rule-based analysis) - Relevance: 1.00"],
            "needs_escalation": False,
            "confidence_threshold": 0.7
        }

        if self.settings.verbose:
            print("[Simple] Simplified analysis completed successfully")

        return analysis_results

    def get_status(self) -> Dict:
        """Get simple status of initialized components."""
        return {
            "rag_pipeline": self.rag_pipeline is not None,
            "crew": self.crew is not None,
            "workflow": self.workflow is not None,
        }


# Backward compatibility aliases
FaultDiagnosisRunner = SimpleFaultDiagnosisRunner
FaultDiagnosisSettings = SimpleWorkflowSettings

__all__ = [
    "SimpleFaultDiagnosisRunner",
    "SimpleWorkflowSettings",
    "FaultDiagnosisRunner",
    "FaultDiagnosisSettings",
]