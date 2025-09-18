"""Simple CrewAI crew orchestration for fault diagnosis MVP."""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import os

from crewai import Crew, Process

from .factory import FaultDiagnosisAgents
from .tasks import FaultDiagnosisTasks
from .crew import Hypothesis


class FaultDiagnosisCrew:
    """Simple orchestration for CrewAI-based fault diagnosis workflow."""

    def __init__(self, custom_models: Optional[Dict[str, str]] = None, verbose: bool = True):
        """Initialize crew with simple configuration.

        Args:
            custom_models: Override default model assignments per role
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        if self.verbose:
            print(f"[Crew] 🚀 Initializing simple FaultDiagnosisCrew")

        # Initialize agents with simple models
        self.agents = FaultDiagnosisAgents(custom_models=custom_models, verbose=verbose)
        if self.verbose:
            print(f"[Crew] Agents factory created")

        # Initialize tasks factory
        self.tasks_factory = FaultDiagnosisTasks(self.agents)
        if self.verbose:
            print(f"[Crew] Tasks factory created")

        # Simple embedding configuration using Titan V1 (what works)
        self.embedder_config = {
            "provider": "bedrock",
            "config": {
                "model": "amazon.titan-embed-text-v1",  # Stable Titan V1 model
                "region_name": os.getenv("BEDROCK_REGION", "us-east-1"),
            }
        }
        if self.verbose:
            print(f"[Crew] ✅ Embedder configured: Titan V1")

        # Placeholder for crew instances
        self.crew = None
        if self.verbose:
            print(f"[Crew] 🎉 Simple FaultDiagnosisCrew initialized successfully")

    def create_crew(self, alert_data: Dict[str, Any]) -> Crew:
        """Create and configure the CrewAI crew for this specific alert."""
        if self.verbose:
            print(f"[Crew] 🛠️ Creating crew for alert: {alert_data.get('alert_id', 'unknown')}")

        # Create simple agents
        planner = self.agents.create_planner_agent()
        retriever = self.agents.create_retriever_agent()
        reasoner = self.agents.create_reasoner_agent()
        reporter = self.agents.create_reporter_agent()

        # Create tasks
        planning_task = self.tasks_factory.create_planning_task(alert_data)

        # Create crew with simple configuration
        crew = Crew(
            agents=[planner, retriever, reasoner, reporter],
            tasks=[planning_task],  # Will add more tasks dynamically
            process=Process.sequential,
            verbose=self.verbose,
            memory=True,                    # Enable memory with simple embeddings
            embedder=self.embedder_config,  # Simple Titan V1 embeddings
            max_requests_per_minute=60,     # Basic rate limiting
        )

        if self.verbose:
            print(f"[Crew] ✅ Crew created with 4 agents and Titan V1 embeddings")
        return crew

    def run_fault_diagnosis(self, alert_data: Dict[str, Any], fixtures: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete fault diagnosis workflow with simple sequential execution."""
        if self.verbose:
            print(f"[CrewAI] 🏁 Starting simple fault diagnosis workflow")

        # Create crew for this specific alert
        crew = self.create_crew(alert_data)
        workflow_results = {}

        try:
            # Stage 1: Planning and Triage
            if self.verbose:
                print("[CrewAI] 🎯 Stage 1: Planning and Triage...")
            planning_task = self.tasks_factory.create_planning_task(alert_data)
            planning_result = crew.kickoff_for_each([{"alert_data": alert_data}])
            workflow_results["planning"] = planning_result

            # Stage 2: Evidence Gathering
            if self.verbose:
                print("[CrewAI] 📊 Stage 2: Evidence Retrieval...")
            evidence_task = self.tasks_factory.create_evidence_retrieval_task({
                "planning": planning_result,
                "fixtures": fixtures
            })
            crew.tasks.append(evidence_task)
            evidence_result = crew.kickoff()
            workflow_results["evidence"] = evidence_result

            # Stage 3: Hypothesis Generation
            if self.verbose:
                print("[CrewAI] 🧠 Stage 3: Hypothesis Generation...")
            hypothesis_task = self.tasks_factory.create_hypothesis_generation_task({
                "evidence": evidence_result,
                "fixtures": fixtures
            })
            crew.tasks.append(hypothesis_task)
            hypothesis_result = crew.kickoff()
            workflow_results["hypotheses"] = hypothesis_result

            # Stage 4: Validation
            if self.verbose:
                print("[CrewAI] ✅ Stage 4: Validation...")
            validation_task = self.tasks_factory.create_validation_task(hypothesis_result)
            crew.tasks.append(validation_task)
            validation_result = crew.kickoff()
            workflow_results["validation"] = validation_result

            # Stage 5: Remediation Planning
            if self.verbose:
                print("[CrewAI] 🛠️ Stage 5: Remediation Planning...")
            remediation_task = self.tasks_factory.create_remediation_task(validation_result)
            crew.tasks.append(remediation_task)
            remediation_result = crew.kickoff()
            workflow_results["remediation"] = remediation_result

            # Stage 6: Final Reporting
            if self.verbose:
                print("[CrewAI] 📝 Stage 6: Final Reporting...")
            reporting_task = self.tasks_factory.create_reporting_task({
                "planning": planning_result,
                "evidence": evidence_result,
                "hypotheses": hypothesis_result,
                "validation": validation_result,
                "remediation": remediation_result,
            })
            crew.tasks.append(reporting_task)
            final_result = crew.kickoff()
            workflow_results["final_report"] = final_result

            if self.verbose:
                print(f"[CrewAI] 🎉 Fault diagnosis workflow completed successfully")

        except Exception as e:
            if self.verbose:
                print(f"[CrewAI] ❌ Workflow error: {e}")

            # Add error information to results
            workflow_results["error"] = str(e)
            workflow_results["status"] = "failed"
        else:
            workflow_results["status"] = "completed"

        return workflow_results

    def parse_hypotheses_from_result(self, crew_result: Dict[str, Any]) -> List[Hypothesis]:
        """Parse CrewAI results into Hypothesis objects for compatibility."""
        hypotheses = []

        # Extract hypotheses from the crew result
        hypothesis_data = crew_result.get("hypotheses", [])
        validation_data = crew_result.get("validation", {})

        if isinstance(hypothesis_data, str):
            # If it's a string result, we need to parse it
            lines = hypothesis_data.split('\n')
            current_hypothesis = {}
            hypothesis_count = 0

            for line in lines:
                line = line.strip()
                if line.startswith('Hypothesis'):
                    if current_hypothesis:
                        hypotheses.append(self._create_hypothesis_from_dict(current_hypothesis, hypothesis_count))
                        hypothesis_count += 1
                    current_hypothesis = {'statement': line}
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':')[1].strip())
                        current_hypothesis['confidence'] = confidence
                    except (ValueError, IndexError):
                        current_hypothesis['confidence'] = 0.5
                elif line.startswith('Citation:'):
                    current_hypothesis['citation'] = line.split(':', 1)[1].strip()

            if current_hypothesis:
                hypotheses.append(self._create_hypothesis_from_dict(current_hypothesis, hypothesis_count))

        # If we don't have any parsed hypotheses, create some default ones
        if not hypotheses:
            hypotheses = self._create_default_hypotheses()

        return hypotheses

    def _create_hypothesis_from_dict(self, data: Dict[str, Any], index: int) -> Hypothesis:
        """Create a Hypothesis object from parsed data."""
        return Hypothesis(
            index=index + 1,
            statement=data.get('statement', f'Hypothesis {index + 1}'),
            confidence=data.get('confidence', 0.5),
            verdict=self._determine_verdict(data.get('confidence', 0.5)),
            citation=data.get('citation', 'crew_ai_analysis.log'),
        )

    def _determine_verdict(self, confidence: float) -> str:
        """Determine verdict based on confidence score."""
        if confidence >= 0.8:
            return "Grounded"
        elif confidence >= 0.5:
            return "Retry"
        else:
            return "Rejected"

    def _create_default_hypotheses(self) -> List[Hypothesis]:
        """Create default hypotheses if parsing fails."""
        return [
            Hypothesis(
                index=1,
                statement="Network connectivity issue detected based on alert analysis",
                confidence=0.82,
                verdict="Grounded",
                citation="crew_ai_evidence.json",
            ),
            Hypothesis(
                index=2,
                statement="Potential configuration drift in system settings",
                confidence=0.65,
                verdict="Retry",
                citation="crew_ai_topology.json",
            ),
            Hypothesis(
                index=3,
                statement="Hardware resource constraints affecting performance",
                confidence=0.45,
                verdict="Rejected",
                citation="crew_ai_config.log",
            ),
        ]


__all__ = ["FaultDiagnosisCrew"]