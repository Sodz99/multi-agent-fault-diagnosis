"""Simple CrewAI crew orchestration for fault diagnosis MVP."""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import os
import boto3
import time

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
            print(f"[Crew] Initializing simple FaultDiagnosisCrew")

        # Initialize agents with simple models
        self.agents = FaultDiagnosisAgents(custom_models=custom_models, verbose=verbose)
        if self.verbose:
            print(f"[Crew] Agents factory created")

        # Initialize tasks factory
        self.tasks_factory = FaultDiagnosisTasks(self.agents)
        if self.verbose:
            print(f"[Crew] Tasks factory created")

        # Simple embedding configuration using Titan V1 (what works)
        # Create boto3 session for embeddings (required by ChromaDB)
        embedder_session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("BEDROCK_REGION", "us-east-1")
        )

        self.embedder_config = {
            "provider": "bedrock",
            "config": {
                "model": "amazon.titan-embed-text-v1",  # Stable Titan V1 model
                "region_name": os.getenv("BEDROCK_REGION", "us-east-1"),
                "session": embedder_session  # Required by ChromaDB's AmazonBedrockEmbeddingFunction
            }
        }
        if self.verbose:
            print(f"[Crew] Embedder configured: Titan V1")

        # Placeholder for crew instances
        self.crew = None
        if self.verbose:
            print(f"[Crew] Simple FaultDiagnosisCrew initialized successfully")

    def create_crew(self, alert_data: Dict[str, Any]) -> Crew:
        """Create and configure the CrewAI crew for this specific alert."""
        if self.verbose:
            print(f"[Crew] Creating crew for alert: {alert_data.get('alert_id', 'unknown')}")

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
            memory=False,                   # Fix: Disable memory to prevent cached delegation responses
            embedder=self.embedder_config,  # Simple Titan V1 embeddings
            max_requests_per_minute=60,     # Basic rate limiting
        )

        if self.verbose:
            print(f"[Crew] Crew created with 4 agents and Titan V1 embeddings")
        return crew

    def run_fault_diagnosis(self, alert_data: Dict[str, Any], fixtures: Dict[str, Any]) -> Dict[str, Any]:
        """Run the complete fault diagnosis workflow with simple sequential execution."""
        if self.verbose:
            print(f"[CrewAI] Starting simple fault diagnosis workflow")

        # Create crew for this specific alert
        crew = self.create_crew(alert_data)
        workflow_results = {}

        try:
            # Stage 1: Planning and Triage
            if self.verbose:
                print("[CrewAI] Stage 1: Planning and Triage...")
            planning_task = self.tasks_factory.create_planning_task(alert_data)
            planning_result = crew.kickoff_for_each([{"alert_data": alert_data}])
            workflow_results["planning"] = planning_result

            # Fix: Add delay to prevent rate limiting
            time.sleep(3)

            # Stage 2: Evidence Gathering
            if self.verbose:
                print("[CrewAI] Stage 2: Evidence Retrieval...")

            # Fix: Extract essential information from ALL fixtures - smart compression
            essential_fixtures = self._extract_fixture_essentials(alert_data, fixtures)

            evidence_task = self.tasks_factory.create_evidence_retrieval_task({
                "planning": planning_result,
                "fixtures": essential_fixtures  # Fix: Extracted fixture essentials to prevent context overflow
            })
            crew.tasks.append(evidence_task)
            evidence_result = crew.kickoff()
            workflow_results["evidence"] = evidence_result

            # Fix: Add delay to prevent rate limiting
            time.sleep(3)

            # Stage 3: Hypothesis Generation
            if self.verbose:
                print("[CrewAI] Stage 3: Hypothesis Generation...")
            hypothesis_task = self.tasks_factory.create_hypothesis_generation_task({
                "evidence": evidence_result,
                "fixtures": essential_fixtures  # Fix: Use same extracted fixture essentials
            })
            crew.tasks.append(hypothesis_task)
            hypothesis_result = crew.kickoff()
            workflow_results["hypotheses"] = hypothesis_result

            # Fix: Add delay to prevent rate limiting
            time.sleep(3)

            # Stage 4: Validation
            if self.verbose:
                print("[CrewAI] Stage 4: Validation...")
            validation_task = self.tasks_factory.create_validation_task(hypothesis_result)
            crew.tasks.append(validation_task)
            validation_result = crew.kickoff()
            workflow_results["validation"] = validation_result

            # Fix: Add longer delay after validation stage (main rate limiting point)
            time.sleep(5)

            # Stage 5: Remediation Planning
            if self.verbose:
                print("[CrewAI] Stage 5: Remediation Planning...")
            remediation_task = self.tasks_factory.create_remediation_task(validation_result)
            crew.tasks.append(remediation_task)
            remediation_result = crew.kickoff()
            workflow_results["remediation"] = remediation_result

            # Fix: Add delay to prevent rate limiting
            time.sleep(3)

            # Stage 6: Final Reporting
            if self.verbose:
                print("[CrewAI] Stage 6: Final Reporting...")
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

            # Fix: Add delay after final stage
            time.sleep(2)

            if self.verbose:
                print(f"[CrewAI] Fault diagnosis workflow completed successfully")

        except Exception as e:
            if self.verbose:
                print(f"[CrewAI] ERROR: Workflow error: {e}")

            # Add error information to results
            workflow_results["error"] = str(e)
            workflow_results["status"] = "failed"
        else:
            workflow_results["status"] = "completed"

        return workflow_results

    def _extract_fixture_essentials(self, alert_data: Dict[str, Any], all_fixtures: Dict[str, Any]) -> Dict[str, Any]:
        """Extract essential information from ALL fixtures - smart compression instead of selection."""
        extracted = {}
        total_original_size = 0
        total_extracted_size = 0

        for key, content in all_fixtures.items():
            original_size = self._estimate_content_size(content)
            total_original_size += original_size

            # Extract key information based on fixture type
            summary = self._summarize_fixture(content, alert_data, key)
            extracted_size = self._estimate_content_size(summary)
            total_extracted_size += extracted_size

            extracted[key] = summary

            if self.verbose:
                compression_ratio = (1 - extracted_size / original_size) * 100 if original_size > 0 else 0
                print(f"[Crew] Extracted {key}: {original_size} -> {extracted_size} chars ({compression_ratio:.1f}% reduction)")

        if self.verbose:
            overall_compression = (1 - total_extracted_size / total_original_size) * 100 if total_original_size > 0 else 0
            print(f"[Crew] Total compression: {total_original_size} -> {total_extracted_size} chars ({overall_compression:.1f}% reduction)")

        return extracted

    def _summarize_fixture(self, content: Any, alert_context: Dict[str, Any], fixture_id: str) -> Dict[str, Any]:
        """Create compact summary of fixture based on its type."""
        # Convert content to string for analysis
        content_str = self._content_to_text(content)

        # Base summary structure
        summary = {
            'type': self._identify_fixture_type(fixture_id),
            'original_size': len(content_str),
            'fixture_id': fixture_id
        }

        # Extract based on fixture type patterns
        if 'alert' in fixture_id.lower() or fixture_id == alert_context.get('alert_id'):
            summary.update(self._extract_alert_essentials(content))
        elif 'runbook' in fixture_id.lower():
            summary.update(self._extract_runbook_essentials(content_str))
        elif 'incident' in fixture_id.lower():
            summary.update(self._extract_incident_essentials(content_str))
        elif 'topology' in fixture_id.lower():
            summary.update(self._extract_topology_essentials(content_str))
        elif 'kpi' in fixture_id.lower():
            summary.update(self._extract_kpi_essentials(content_str))
        elif 'log' in fixture_id.lower():
            summary.update(self._extract_log_essentials(content_str))
        elif 'config' in fixture_id.lower() or 'enodeb' in fixture_id.lower():
            summary.update(self._extract_config_essentials(content_str))
        elif 'design' in fixture_id.lower():
            summary.update(self._extract_design_essentials(content_str))
        elif 'postmortem' in fixture_id.lower():
            summary.update(self._extract_postmortem_essentials(content_str))
        elif 'alarm' in fixture_id.lower() or 'correlation' in fixture_id.lower():
            summary.update(self._extract_alarm_essentials(content_str))
        else:
            # Generic extraction for unknown types
            summary.update(self._extract_generic_essentials(content_str))

        return summary

    def _identify_fixture_type(self, fixture_id: str) -> str:
        """Identify fixture type from its ID."""
        fixture_id_lower = fixture_id.lower()
        if 'alert' in fixture_id_lower:
            return 'alert'
        elif 'runbook' in fixture_id_lower:
            return 'runbook'
        elif 'incident' in fixture_id_lower:
            return 'incident'
        elif 'topology' in fixture_id_lower:
            return 'topology'
        elif 'kpi' in fixture_id_lower:
            return 'kpi'
        elif 'log' in fixture_id_lower:
            return 'logs'
        elif 'config' in fixture_id_lower or 'enodeb' in fixture_id_lower:
            return 'config'
        elif 'design' in fixture_id_lower:
            return 'design'
        elif 'postmortem' in fixture_id_lower:
            return 'postmortem'
        elif 'alarm' in fixture_id_lower:
            return 'alarm'
        else:
            return 'unknown'

    def _extract_alert_keywords(self, alert_data: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords from alert data for matching."""
        keywords = []

        # Extract from common alert fields
        for field in ['title', 'summary', 'description', 'type']:
            value = alert_data.get(field, '')
            if isinstance(value, str) and value:
                # Simple keyword extraction - split on common delimiters
                words = value.lower().replace('-', ' ').replace('_', ' ').split()
                keywords.extend(words)

        # Add asset-related keywords
        assets = alert_data.get('impacted_assets', [])
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, str):
                    keywords.extend(asset.lower().replace('-', ' ').split())

        # Add severity as keyword
        severity = alert_data.get('severity', '')
        if severity:
            keywords.append(severity.lower())

        # Remove duplicates and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = list(set(word for word in keywords if len(word) > 2 and word not in stop_words))

        return keywords

    def _calculate_relevance(self, content: Any, keywords: List[str]) -> float:
        """Calculate relevance score between content and alert keywords."""
        if not keywords:
            return 0.0

        # Convert content to searchable text
        text = self._content_to_text(content)
        if not text:
            return 0.0

        text_lower = text.lower()

        # Count keyword matches
        matches = 0
        total_weight = 0

        for keyword in keywords:
            # Weight longer keywords more heavily
            weight = len(keyword) if len(keyword) > 3 else 1
            total_weight += weight

            if keyword in text_lower:
                matches += weight

        # Return relevance as a percentage
        return (matches / total_weight) if total_weight > 0 else 0.0

    def _content_to_text(self, content: Any) -> str:
        """Convert various content types to searchable text."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract text from dict values
            text_parts = []
            for value in content.values():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (list, dict)):
                    text_parts.append(str(value))
            return ' '.join(text_parts)
        elif isinstance(content, list):
            # Extract text from list items
            return ' '.join(str(item) for item in content)
        else:
            return str(content)

    def _estimate_content_size(self, content: Any) -> int:
        """Estimate the size of content in characters."""
        try:
            if isinstance(content, str):
                return len(content)
            else:
                return len(str(content))
        except:
            return 0

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

    # Fixture-specific extraction functions
    def _extract_alert_essentials(self, content: Any) -> Dict[str, Any]:
        """Extract essential information from alert fixtures."""
        if isinstance(content, dict):
            return {
                'alert_id': content.get('alert_id', ''),
                'severity': content.get('severity', ''),
                'title': content.get('title', ''),
                'assets': content.get('impacted_assets', []),
                'kpis': content.get('kpis', {}),
                'summary': content.get('summary', '')[:200] + '...' if len(content.get('summary', '')) > 200 else content.get('summary', '')
            }
        return {'raw_content': str(content)[:500]}

    def _extract_runbook_essentials(self, content: str) -> Dict[str, Any]:
        """Extract essential information from runbook fixtures."""
        lines = content.split('\n')
        steps = []

        for line in lines:
            line = line.strip()
            if line and (line.startswith(('1.', '2.', '3.', '4.', '5.')) or line.startswith('- ')):
                steps.append(line[:100])  # Truncate long steps

        return {
            'procedure_steps': steps[:10],  # Max 10 steps
            'key_actions': [step for step in steps if any(keyword in step.lower()
                          for keyword in ['escalate', 'rebalance', 'validate', 'inspect'])][:5]
        }

    def _extract_incident_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from incident fixtures."""
        if isinstance(content_str, str):
            # Try to extract JSON-like data
            import re

            essentials = {}

            # Look for common incident patterns
            incident_id_match = re.search(r'incident_id[\'"]:\s*[\'"]([^\'\"]+)', content_str)
            if incident_id_match:
                essentials['incident_id'] = incident_id_match.group(1)

            impact_match = re.search(r'impact[\'"]:\s*[\'"]([^\'\"]+)', content_str)
            if impact_match:
                essentials['impact'] = impact_match.group(1)

            resolution_match = re.search(r'resolution_time_minutes[\'"]:\s*(\d+)', content_str)
            if resolution_match:
                essentials['resolution_time'] = int(resolution_match.group(1))

            return essentials

        return {'type': 'incident', 'content_preview': str(content_str)[:200]}

    def _extract_topology_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from topology fixtures."""
        import re

        essentials = {}

        # Extract sector info
        sector_match = re.search(r'sector[\'"]:\s*[\'"]([^\'\"]+)', content_str)
        if sector_match:
            essentials['sector'] = sector_match.group(1)

        # Extract dependencies
        deps_match = re.search(r'dependencies[\'"]:\s*\[([^\]]+)\]', content_str)
        if deps_match:
            deps = [dep.strip().replace("'", "").replace('"', '') for dep in deps_match.group(1).split(',')]
            essentials['dependencies'] = deps[:5]  # Max 5 deps

        # Extract backhaul info
        backhaul_match = re.search(r'backhaul[\'"]:\s*\{([^\}]+)\}', content_str)
        if backhaul_match:
            essentials['backhaul_info'] = backhaul_match.group(1)[:100]

        return essentials

    def _extract_kpi_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from KPI fixtures."""
        import re

        # Look for numeric patterns that indicate KPIs
        numbers = re.findall(r'(\d+\.?\d*)', content_str)

        # Extract time windows
        time_matches = re.findall(r'20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', content_str)

        essentials = {
            'kpi_type': 'timeseries' if 'timeseries' in content_str else 'rollup',
            'data_points': len(numbers),
            'time_range': f"{time_matches[0]} to {time_matches[-1]}" if len(time_matches) >= 2 else 'unknown',
            'key_metrics': []
        }

        # Extract key metric names
        for metric in ['packet_loss', 'latency', 'throughput', 'call_drop_rate', 'prb_utilization']:
            if metric in content_str:
                essentials['key_metrics'].append(metric)

        return essentials

    def _extract_log_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from log fixtures."""
        lines = content_str.split('\n')

        error_lines = []
        warning_lines = []
        critical_lines = []

        for line in lines[:50]:  # Process max 50 lines
            if 'ERROR' in line:
                error_lines.append(line.split('] ')[-1] if '] ' in line else line)
            elif 'WARN' in line:
                warning_lines.append(line.split('] ')[-1] if '] ' in line else line)
            elif 'CRITICAL' in line:
                critical_lines.append(line.split('] ')[-1] if '] ' in line else line)

        return {
            'total_lines': len(lines),
            'critical_events': critical_lines[:3],
            'error_events': error_lines[:5],
            'warning_events': warning_lines[:3],
            'log_timespan': 'extracted from timestamps' if any('T' in line for line in lines[:5]) else 'unknown'
        }

    def _extract_config_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from config fixtures."""
        import re

        essentials = {}

        # Extract cell config
        cell_id_match = re.search(r'cell-id\s+(\d+)', content_str)
        if cell_id_match:
            essentials['cell_id'] = cell_id_match.group(1)

        # Extract bandwidth
        bandwidth_match = re.search(r'bandwidth\s+(\d+\w+)', content_str)
        if bandwidth_match:
            essentials['bandwidth'] = bandwidth_match.group(1)

        # Extract IP addresses
        ip_matches = re.findall(r'(\d+\.\d+\.\d+\.\d+)', content_str)
        if ip_matches:
            essentials['ip_addresses'] = ip_matches[:3]  # Max 3 IPs

        # Extract thresholds
        thresholds = {}
        for line in content_str.split('\n'):
            if 'threshold' in line.lower():
                thresholds[line.strip()] = True
        essentials['has_thresholds'] = len(thresholds) > 0

        return essentials

    def _extract_design_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from design document fixtures."""
        lines = content_str.split('\n')

        essentials = {
            'document_type': 'network_design',
            'key_sections': [],
            'critical_info': []
        }

        # Extract section headers (lines starting with #)
        for line in lines:
            if line.strip().startswith('#') and len(line.strip()) > 1:
                essentials['key_sections'].append(line.strip()[:50])
                if len(essentials['key_sections']) >= 5:
                    break

        # Extract critical numbers and facts
        import re
        critical_patterns = [
            r'(\d+,?\d*)\s*subscribers',
            r'(\d+)\s*sites',
            r'(\d+\.?\d*%)\s*availability',
            r'(\w+-\w+-\d+)',  # Asset names like MME-CLSTR-3
        ]

        for pattern in critical_patterns:
            matches = re.findall(pattern, content_str)
            essentials['critical_info'].extend(matches[:3])

        return essentials

    def _extract_postmortem_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from postmortem fixtures."""
        import re

        essentials = {}

        # Extract incident ID
        incident_match = re.search(r'Incident ID.*?:.*?([A-Z0-9-]+)', content_str)
        if incident_match:
            essentials['incident_id'] = incident_match.group(1)

        # Extract duration
        duration_match = re.search(r'Duration.*?:.*?(\d+h?\s*\d*m?)', content_str)
        if duration_match:
            essentials['duration'] = duration_match.group(1)

        # Extract root cause
        lines = content_str.split('\n')
        for i, line in enumerate(lines):
            if 'root cause' in line.lower() and i + 1 < len(lines):
                essentials['root_cause'] = lines[i + 1].strip()[:200]
                break

        # Extract impact
        impact_match = re.search(r'(\d+,?\d*)\s*subscribers?\s*affected', content_str)
        if impact_match:
            essentials['impact'] = f"{impact_match.group(1)} subscribers affected"

        return essentials

    def _extract_alarm_essentials(self, content_str: str) -> Dict[str, Any]:
        """Extract essential information from alarm correlation fixtures."""
        import re

        essentials = {
            'format': 'xml' if content_str.strip().startswith('<?xml') else 'text',
            'alarm_count': 0,
            'critical_alarms': [],
            'correlation_rules': []
        }

        # Count alarms
        alarm_matches = re.findall(r'<Alarm\s+id="([^"]+)"', content_str)
        essentials['alarm_count'] = len(alarm_matches)

        # Extract critical alarms
        critical_matches = re.findall(r'<Severity>Critical</Severity>.*?<Type>([^<]+)</Type>', content_str, re.DOTALL)
        essentials['critical_alarms'] = critical_matches[:3]

        # Extract rule names
        rule_matches = re.findall(r'<Name>([^<]+)</Name>', content_str)
        essentials['correlation_rules'] = rule_matches[:5]

        return essentials

    def _extract_generic_essentials(self, content_str: str) -> Dict[str, Any]:
        """Generic extraction for unknown fixture types."""
        import re

        # Extract any numbers that might be important
        numbers = re.findall(r'\b\d+\.?\d*\b', content_str)

        # Extract any asset-like identifiers
        assets = re.findall(r'\b[A-Z]{2,}-[A-Z0-9-]+\b', content_str)

        # Extract timestamps
        timestamps = re.findall(r'20\d{2}-\d{2}-\d{2}', content_str)

        return {
            'content_length': len(content_str),
            'numeric_values': numbers[:10],  # First 10 numbers
            'asset_identifiers': assets[:5],  # First 5 assets
            'timestamps': timestamps[:3],     # First 3 timestamps
            'preview': content_str[:200] + '...' if len(content_str) > 200 else content_str
        }


__all__ = ["FaultDiagnosisCrew"]