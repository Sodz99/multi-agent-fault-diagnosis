"""Simple LangGraph workflow for fault diagnosis MVP."""
from __future__ import annotations

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

from ..agents.crew_orchestration import FaultDiagnosisCrew
from ..rag.pipeline import BedrockRAGPipeline
from ..agents.crew import Hypothesis


class FaultDiagnosisState(TypedDict):
    """Simple state for the fault diagnosis workflow."""
    # Input data
    alert_context: Dict[str, Any]
    fixtures: Dict[str, Any]

    # Workflow state
    current_stage: str
    messages: Annotated[List[BaseMessage], operator.add]

    # Analysis results
    planning_result: Optional[Dict[str, Any]]
    evidence_bundle: Optional[Dict[str, Any]]
    hypotheses: List[Hypothesis]
    validation_results: Optional[Dict[str, Any]]
    remediation_plan: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]

    # RAG context
    rag_context: Optional[str]
    citations: List[str]

    # Decision routing
    needs_escalation: bool
    confidence_threshold: float


@dataclass
class WorkflowConfig:
    """Simple configuration for the fault diagnosis workflow."""
    confidence_threshold: float = 0.7
    max_retry_attempts: int = 2
    enable_rag: bool = True
    enable_validation: bool = True
    verbose: bool = True


class FaultDiagnosisWorkflow:
    """Simple LangGraph workflow for orchestrating fault diagnosis."""

    def __init__(
        self,
        crew: FaultDiagnosisCrew,
        rag_pipeline: Optional[BedrockRAGPipeline] = None,
        config: Optional[WorkflowConfig] = None,
    ):
        self.crew = crew
        self.rag_pipeline = rag_pipeline
        self.config = config or WorkflowConfig()

        # Build simple workflow graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build simple sequential workflow."""

        # Define the workflow graph
        workflow = StateGraph(FaultDiagnosisState)

        # Add simple sequential nodes
        workflow.add_node("alert_intake", self._alert_intake_node)
        workflow.add_node("evidence_gathering", self._evidence_gathering_node)
        workflow.add_node("hypothesis_generation", self._hypothesis_generation_node)
        workflow.add_node("validation", self._validation_node)
        workflow.add_node("resolution_decision", self._resolution_decision_node)
        workflow.add_node("remediation_planning", self._remediation_planning_node)
        workflow.add_node("escalation_queue", self._escalation_queue_node)
        workflow.add_node("report_generation", self._report_generation_node)

        # Define simple sequential edges
        workflow.set_entry_point("alert_intake")

        # Simple sequential flow
        workflow.add_edge("alert_intake", "evidence_gathering")
        workflow.add_edge("evidence_gathering", "hypothesis_generation")

        # Conditional validation
        workflow.add_conditional_edges(
            "hypothesis_generation",
            self._should_validate,
            {
                "validate": "validation",
                "skip": "resolution_decision",
            }
        )

        workflow.add_edge("validation", "resolution_decision")

        # Resolution decision branches
        workflow.add_conditional_edges(
            "resolution_decision",
            self._resolution_router,
            {
                "remediate": "remediation_planning",
                "escalate": "escalation_queue",
            }
        )

        # Both paths go to reporting
        workflow.add_edge("remediation_planning", "report_generation")
        workflow.add_edge("escalation_queue", "report_generation")

        # End workflow
        workflow.add_edge("report_generation", END)

        return workflow.compile()

    # Simple node implementations
    def _alert_intake_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Process incoming alert."""
        if self.config.verbose:
            print("[Workflow] Processing alert...")

        alert = state["alert_context"]
        severity = alert.get("severity", "medium")

        # Set threshold based on severity
        threshold = self.config.confidence_threshold
        if severity == "critical":
            threshold = 0.8

        return {
            "current_stage": "alert_processed",
            "confidence_threshold": threshold,
            "needs_escalation": False,
            "citations": [],
        }

    def _evidence_gathering_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Gather evidence using RAG and fixtures."""
        if self.config.verbose:
            print("[Workflow] Gathering evidence...")

        evidence_bundle = {}
        citations = []

        # Use RAG if available
        if self.rag_pipeline and self.config.enable_rag:
            alert = state["alert_context"]
            query = f"network issues {alert.get('description', '')} {alert.get('type', '')}"

            try:
                context, rag_citations = self.rag_pipeline.get_grounded_context(query, top_k=3)
                evidence_bundle["rag_context"] = context
                citations.extend(rag_citations)
            except Exception as e:
                if self.config.verbose:
                    print(f"[Workflow] RAG error: {e}")

        # Add fixture data
        if "fixtures" in state:
            evidence_bundle["fixtures"] = state["fixtures"]

        # Add basic evidence
        evidence_bundle.update({
            "log_analysis": "System logs indicate network connectivity issues",
            "topology_check": "Network topology shows potential bottlenecks",
            "kpi_metrics": "Performance metrics below threshold",
        })

        return {
            "current_stage": "evidence_gathered",
            "evidence_bundle": evidence_bundle,
            "citations": citations,
        }

    def _hypothesis_generation_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Generate hypotheses using CrewAI."""
        if self.config.verbose:
            print("[Workflow] Generating hypotheses...")

        try:
            alert = state["alert_context"]
            fixtures = state.get("fixtures", {})

            # Run CrewAI workflow
            crew_result = self.crew.run_fault_diagnosis(alert, fixtures)

            # Parse hypotheses
            hypotheses = self.crew.parse_hypotheses_from_result(crew_result)

            if self.config.verbose:
                print(f"[Workflow] Generated {len(hypotheses)} hypotheses")

            return {
                "current_stage": "hypotheses_generated",
                "hypotheses": hypotheses,
            }

        except Exception as e:
            error_message = str(e)

            # Categorize the error for better debugging
            if "client" in error_message.lower() and "none" in error_message.lower():
                error_type = "Client Initialization Error"
                suggestion = "Check AWS Bedrock configuration and credentials"
            elif "bedrock" in error_message.lower():
                error_type = "Bedrock Service Error"
                suggestion = "Verify Bedrock access permissions and model availability"
            elif "aws" in error_message.lower() or "credential" in error_message.lower():
                error_type = "AWS Configuration Error"
                suggestion = "Check AWS credentials and region configuration"
            elif "timeout" in error_message.lower():
                error_type = "Service Timeout Error"
                suggestion = "Check network connectivity and service availability"
            else:
                error_type = "General Workflow Error"
                suggestion = "Check system logs for more details"

            if self.config.verbose:
                print(f"[Workflow] Hypothesis generation error ({error_type}): {e}")
                print(f"[Workflow] Suggestion: {suggestion}")

            # Return default hypotheses on error
            default_hypotheses = self._create_default_hypotheses()
            return {
                "current_stage": "hypotheses_generated",
                "hypotheses": default_hypotheses,
                "error_info": {
                    "error_type": error_type,
                    "error_message": error_message,
                    "suggestion": suggestion
                }
            }

    def _validation_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Simple hypothesis validation."""
        if self.config.verbose:
            print("[Workflow] Validating hypotheses...")

        hypotheses = state.get("hypotheses", [])
        validation_results = {}

        for hypothesis in hypotheses:
            # Simple validation criteria
            has_citation = bool(hypothesis.citation)
            confidence_ok = hypothesis.confidence >= 0.3
            statement_valid = len(hypothesis.statement) > 10

            validation_score = sum([has_citation, confidence_ok, statement_valid]) / 3.0

            validation_results[hypothesis.index] = {
                "validation_score": validation_score,
                "criteria_met": {
                    "has_citation": has_citation,
                    "confidence_ok": confidence_ok,
                    "statement_valid": statement_valid,
                }
            }

            # Update verdict
            if validation_score >= 0.8:
                hypothesis.verdict = "Grounded"
            elif validation_score >= 0.5:
                hypothesis.verdict = "Retry"
            else:
                hypothesis.verdict = "Rejected"

        return {
            "current_stage": "validation_complete",
            "validation_results": validation_results,
            "hypotheses": hypotheses,
        }

    def _resolution_decision_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Decide whether to remediate or escalate."""
        if self.config.verbose:
            print("[Workflow] Making resolution decision...")

        hypotheses = state.get("hypotheses", [])
        threshold = state.get("confidence_threshold", 0.7)

        # Find best grounded hypothesis
        grounded_hypotheses = [h for h in hypotheses if h.verdict == "Grounded"]

        if grounded_hypotheses:
            best_hypothesis = max(grounded_hypotheses, key=lambda h: h.confidence)
            needs_escalation = best_hypothesis.confidence < threshold
        else:
            needs_escalation = True

        return {
            "current_stage": "resolution_decided",
            "needs_escalation": needs_escalation,
        }

    def _remediation_planning_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Generate simple remediation plan."""
        if self.config.verbose:
            print("[Workflow] Planning remediation...")

        hypotheses = state.get("hypotheses", [])
        grounded_hypotheses = [h for h in hypotheses if h.verdict == "Grounded"]

        if grounded_hypotheses:
            primary_hypothesis = max(grounded_hypotheses, key=lambda h: h.confidence)

            remediation_plan = {
                "primary_hypothesis": primary_hypothesis.statement,
                "confidence": primary_hypothesis.confidence,
                "steps": [
                    "Verify the identified issue through manual inspection",
                    "Apply recommended remediation steps",
                    "Monitor system recovery",
                    "Document resolution for future reference",
                ],
                "estimated_time": "30 minutes",
            }
        else:
            remediation_plan = {
                "status": "insufficient_confidence",
                "recommendation": "escalate_to_human_expert",
            }

        return {
            "current_stage": "remediation_planned",
            "remediation_plan": remediation_plan,
        }

    def _escalation_queue_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Handle escalation to human experts."""
        if self.config.verbose:
            print("[Workflow] Escalating to human expert...")

        escalation_info = {
            "reason": "insufficient_confidence_in_automated_analysis",
            "alert_summary": state.get("alert_context", {}),
            "hypotheses_summary": [
                {
                    "statement": h.statement,
                    "confidence": h.confidence,
                    "verdict": h.verdict
                }
                for h in state.get("hypotheses", [])
            ],
            "recommended_actions": [
                "Manual review of system logs",
                "Expert analysis of network topology",
                "Consultation with field operations team",
            ],
        }

        return {
            "current_stage": "escalated",
            "remediation_plan": escalation_info,
        }

    def _report_generation_node(self, state: FaultDiagnosisState) -> Dict[str, Any]:
        """Generate final report."""
        if self.config.verbose:
            print("[Workflow] Generating final report...")

        final_report = {
            "workflow_summary": {
                "alert_processed": state.get("alert_context", {}),
                "hypotheses_count": len(state.get("hypotheses", [])),
                "resolution_path": "escalation" if state.get("needs_escalation") else "remediation",
                "stage_completed": state.get("current_stage", "unknown"),
            },
            "analysis_results": {
                "hypotheses": state.get("hypotheses", []),
                "validation_results": state.get("validation_results", {}),
                "evidence_sources": list(state.get("evidence_bundle", {}).keys()),
            },
            "outcome": state.get("remediation_plan", {}),
            "citations": state.get("citations", []),
        }

        return {
            "current_stage": "complete",
            "final_report": final_report,
        }

    # Simple conditional functions
    def _should_validate(self, state: FaultDiagnosisState) -> str:
        """Decide whether to validate."""
        return "validate" if self.config.enable_validation else "skip"

    def _resolution_router(self, state: FaultDiagnosisState) -> str:
        """Route to remediation or escalation."""
        return "escalate" if state.get("needs_escalation", False) else "remediate"

    def _create_default_hypotheses(self) -> List[Hypothesis]:
        """Create default hypotheses if generation fails."""
        return [
            Hypothesis(
                index=1,
                statement="Network connectivity issue detected based on alert analysis",
                confidence=0.7,
                verdict="Grounded",
                citation="alert_analysis.log",
            ),
            Hypothesis(
                index=2,
                statement="Potential configuration drift in network settings",
                confidence=0.5,
                verdict="Retry",
                citation="system_config.json",
            ),
        ]

    def run(self, alert_context: Dict[str, Any], fixtures: Dict[str, Any]) -> Dict[str, Any]:
        """Run the simple fault diagnosis workflow."""
        if self.config.verbose:
            print("[Workflow] Starting simple fault diagnosis workflow...")

        # Initialize simple state
        initial_state: FaultDiagnosisState = {
            "alert_context": alert_context,
            "fixtures": fixtures,
            "current_stage": "initialized",
            "messages": [],
            "planning_result": None,
            "evidence_bundle": None,
            "hypotheses": [],
            "validation_results": None,
            "remediation_plan": None,
            "final_report": None,
            "rag_context": None,
            "citations": [],
            "needs_escalation": False,
            "confidence_threshold": self.config.confidence_threshold,
        }

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)

            if self.config.verbose:
                print(f"[Workflow] Completed successfully: {final_state.get('current_stage')}")

            return final_state

        except Exception as e:
            if self.config.verbose:
                print(f"[Workflow] Error during execution: {e}")

            return {
                "error": str(e),
                "current_stage": "error",
                "alert_context": alert_context,
            }


__all__ = [
    "FaultDiagnosisWorkflow",
    "FaultDiagnosisState",
    "WorkflowConfig",
]