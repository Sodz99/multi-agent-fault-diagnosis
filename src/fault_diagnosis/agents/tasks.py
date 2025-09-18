"""CrewAI task definitions for fault diagnosis workflow."""
from __future__ import annotations

from typing import Dict, Any, List
from crewai import Task
from .factory import FaultDiagnosisAgents


class FaultDiagnosisTasks:
    """Factory for creating fault diagnosis workflow tasks."""

    def __init__(self, agents: FaultDiagnosisAgents):
        self.agents = agents

    def create_planning_task(self, alert_data: Dict[str, Any]) -> Task:
        """Create the initial planning and triage task."""
        return Task(
            description=f"""
            Analyze the incoming telecom alert and create an initial assessment plan:

            Alert Data: {alert_data}

            Your responsibilities:
            1. Parse the alert details including severity, affected assets, and KPIs
            2. Classify the incident type (RAN congestion, backhaul failure, core network issue, etc.)
            3. Determine priority level and potential impact scope
            4. Create an initial action plan with recommended next steps
            5. Identify which specialist teams should be engaged

            Output should include:
            - Alert classification and severity assessment
            - Initial hypothesis about the problem domain
            - Recommended evidence gathering priorities
            - Stakeholder notification requirements

            Be systematic and thorough in your analysis.
            """,
            agent=self.agents.create_planner_agent(),
            expected_output="A structured assessment report with alert classification, priority, and initial action plan",
        )

    def create_evidence_retrieval_task(self, context: Dict[str, Any]) -> Task:
        """Create the evidence gathering and retrieval task."""
        return Task(
            description=f"""
            Based on the planning assessment, retrieve relevant technical evidence and documentation:

            Context from Planning: {context}

            Your responsibilities:
            1. Search historical incident database for similar patterns
            2. Retrieve relevant runbooks and remediation procedures
            3. Gather topology information for affected network segments
            4. Collect KPI data and performance metrics
            5. Find relevant technical documentation and best practices

            For each piece of evidence:
            - Provide proper citations with source identifiers
            - Include relevance score and rationale
            - Extract key technical details
            - Note any gaps that need additional investigation

            Ensure all evidence is authoritative and well-documented.
            """,
            agent=self.agents.create_retriever_agent(),
            expected_output="Comprehensive evidence package with citations, relevance scores, and technical details",
        )

    def create_hypothesis_generation_task(self, evidence: Dict[str, Any]) -> Task:
        """Create the hypothesis generation and analysis task."""
        return Task(
            description=f"""
            Synthesize the gathered evidence into ranked hypotheses about the root cause:

            Evidence Package: {evidence}

            Your responsibilities:
            1. Analyze patterns across all evidence sources
            2. Generate 3-5 ranked hypotheses about the root cause
            3. Assign confidence scores (0.0-1.0) based on evidence strength
            4. Identify required validation steps for each hypothesis
            5. Flag any hypotheses that lack sufficient evidence

            For each hypothesis:
            - Provide a clear technical statement of the proposed root cause
            - List supporting evidence with citations
            - Identify potential validation methods
            - Assess confidence level with justification
            - Consider alternative explanations

            Focus on technical accuracy and avoid unsupported conclusions.
            """,
            agent=self.agents.create_reasoner_agent(),
            expected_output="Ranked list of hypotheses with confidence scores, supporting evidence, and validation requirements",
        )

    def create_validation_task(self, hypotheses: List[Dict[str, Any]]) -> Task:
        """Create the hypothesis validation task."""
        return Task(
            description=f"""
            Validate the proposed hypotheses through systematic analysis:

            Hypotheses to Validate: {hypotheses}

            Your responsibilities:
            1. Review each hypothesis against available evidence
            2. Check for logical consistency and technical feasibility
            3. Verify that all claims are properly supported by citations
            4. Identify any gaps in reasoning or missing evidence
            5. Recommend additional data collection if needed

            Validation criteria:
            - Technical accuracy of the hypothesis
            - Strength of supporting evidence
            - Consistency with known network behaviors
            - Completeness of the analysis

            Mark each hypothesis as: Grounded ✓, Needs More Evidence ?, or Rejected ✗
            """,
            agent=self.agents.create_reasoner_agent(),
            expected_output="Validation results for each hypothesis with clear verdicts and justification",
        )

    def create_remediation_task(self, validated_hypotheses: List[Dict[str, Any]]) -> Task:
        """Create the remediation planning task."""
        return Task(
            description=f"""
            Develop actionable remediation plans based on validated hypotheses:

            Validated Hypotheses: {validated_hypotheses}

            Your responsibilities:
            1. Create step-by-step remediation procedures for the top hypotheses
            2. Identify required maintenance windows and change approvals
            3. Estimate MTTR (Mean Time To Recovery) for each approach
            4. Assess potential risks and rollback procedures
            5. Prepare stakeholder communications

            For each remediation plan:
            - Provide clear, executable steps
            - Include timing estimates and resource requirements
            - Reference relevant runbooks and procedures
            - Identify success criteria and validation checks
            - Consider impact on other network services

            Ensure plans are practical and follow established change management processes.
            """,
            agent=self.agents.create_reporter_agent(),
            expected_output="Detailed remediation plan with steps, timelines, risks, and success criteria",
        )

    def create_reporting_task(self, analysis_results: Dict[str, Any]) -> Task:
        """Create the final reporting and documentation task."""
        return Task(
            description=f"""
            Create comprehensive incident report and stakeholder communications:

            Analysis Results: {analysis_results}

            Your responsibilities:
            1. Synthesize all findings into a clear narrative
            2. Create executive summary for leadership
            3. Document technical details for engineering teams
            4. Prepare customer communications if needed
            5. Update knowledge base with lessons learned

            Report sections:
            - Executive Summary (impact, resolution, prevention)
            - Technical Analysis (root cause, evidence, validation)
            - Remediation Plan (steps, timeline, resources)
            - Post-incident Actions (monitoring, preventive measures)
            - Appendices (detailed data, citations, references)

            Ensure the report is professional, accurate, and actionable for all audiences.
            """,
            agent=self.agents.create_reporter_agent(),
            expected_output="Complete incident report with executive summary, technical analysis, and remediation plan",
        )


__all__ = ["FaultDiagnosisTasks"]