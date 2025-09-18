"""Validator subgraphs for hypothesis validation in fault diagnosis."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from ..agents.crew import Hypothesis


class ValidationVerdict(Enum):
    """Validation verdict options."""
    GROUNDED = "Grounded"
    RETRY = "Retry"
    REJECTED = "Rejected"


@dataclass
class ValidationResult:
    """Result of hypothesis validation."""
    hypothesis_index: int
    verdict: ValidationVerdict
    confidence_score: float
    validation_details: Dict[str, Any]
    validator_name: str
    citations_verified: List[str]
    issues_found: List[str]
    recommendations: List[str]


class BaseValidator(ABC):
    """Base class for hypothesis validators."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def validate(
        self,
        hypothesis: Hypothesis,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate a hypothesis and return results."""
        pass

    def _extract_technical_claims(self, statement: str) -> List[str]:
        """Extract technical claims from hypothesis statement."""
        # Simple pattern matching for technical terms
        patterns = [
            r'(\w+\s+congestion)',
            r'(packet loss.*?%)',
            r'(latency.*?ms)',
            r'(\w+\s+degradation)',
            r'(sector\s+\w+)',
            r'(\w+-\w+-\d+)',  # Equipment identifiers
        ]

        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, statement, re.IGNORECASE)
            claims.extend(matches)

        return claims


class TrafficProbeValidator(BaseValidator):
    """Validates hypotheses related to traffic and network performance."""

    def __init__(self):
        super().__init__("traffic_probe_agent")

    def validate(
        self,
        hypothesis: Hypothesis,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate traffic-related claims in the hypothesis."""

        issues = []
        verified_citations = []
        confidence_factors = []

        # Check for traffic-related claims
        statement_lower = hypothesis.statement.lower()
        has_traffic_claim = any(term in statement_lower for term in [
            'packet loss', 'congestion', 'latency', 'throughput', 'traffic'
        ])

        if not has_traffic_claim:
            # Not applicable to this validator
            return ValidationResult(
                hypothesis_index=hypothesis.index,
                verdict=ValidationVerdict.GROUNDED,
                confidence_score=hypothesis.confidence,
                validation_details={"applicable": False},
                validator_name=self.name,
                citations_verified=[],
                issues_found=["Not applicable to traffic validation"],
                recommendations=[]
            )

        # Validate packet loss claims
        if 'packet loss' in statement_lower:
            packet_loss_valid = self._validate_packet_loss_claim(
                hypothesis.statement, evidence
            )
            confidence_factors.append(packet_loss_valid)

            if packet_loss_valid:
                verified_citations.append("traffic_probe_metrics")
            else:
                issues.append("Packet loss claim not supported by evidence")

        # Validate latency claims
        if 'latency' in statement_lower:
            latency_valid = self._validate_latency_claim(
                hypothesis.statement, evidence
            )
            confidence_factors.append(latency_valid)

            if latency_valid:
                verified_citations.append("latency_measurements")
            else:
                issues.append("Latency claim not supported by evidence")

        # Validate congestion claims
        if 'congestion' in statement_lower:
            congestion_valid = self._validate_congestion_claim(
                hypothesis.statement, evidence
            )
            confidence_factors.append(congestion_valid)

            if congestion_valid:
                verified_citations.append("congestion_indicators")
            else:
                issues.append("Congestion claim lacks supporting metrics")

        # Calculate overall confidence
        if confidence_factors:
            validation_confidence = sum(confidence_factors) / len(confidence_factors)
        else:
            validation_confidence = 0.5

        # Determine verdict
        if validation_confidence >= 0.8 and len(issues) == 0:
            verdict = ValidationVerdict.GROUNDED
        elif validation_confidence >= 0.5:
            verdict = ValidationVerdict.RETRY
        else:
            verdict = ValidationVerdict.REJECTED

        recommendations = self._generate_traffic_recommendations(
            hypothesis.statement, issues
        )

        return ValidationResult(
            hypothesis_index=hypothesis.index,
            verdict=verdict,
            confidence_score=validation_confidence,
            validation_details={
                "packet_loss_validated": 'packet loss' in statement_lower,
                "latency_validated": 'latency' in statement_lower,
                "congestion_validated": 'congestion' in statement_lower,
                "metrics_available": len(verified_citations) > 0,
            },
            validator_name=self.name,
            citations_verified=verified_citations,
            issues_found=issues,
            recommendations=recommendations,
        )

    def _validate_packet_loss_claim(self, statement: str, evidence: Dict[str, Any]) -> float:
        """Validate packet loss claims against evidence."""
        # Extract percentage if mentioned
        packet_loss_match = re.search(r'(\d+(?:\.\d+)?)%?\s*packet\s*loss', statement, re.IGNORECASE)

        # Check evidence for packet loss data
        log_analysis = evidence.get("log_analysis", "")
        if "packet loss" in log_analysis.lower():
            # Extract percentage from evidence
            evidence_match = re.search(r'(\d+(?:\.\d+)?)%', log_analysis)
            if evidence_match and packet_loss_match:
                claimed_loss = float(packet_loss_match.group(1))
                evidence_loss = float(evidence_match.group(1))
                # Validate if claim is within reasonable range of evidence
                if abs(claimed_loss - evidence_loss) <= 2.0:  # Within 2% tolerance
                    return 0.9
                else:
                    return 0.3
            elif evidence_match:
                return 0.7  # Evidence exists but no specific claim
            else:
                return 0.5  # General packet loss mentioned
        else:
            return 0.2  # No supporting evidence

    def _validate_latency_claim(self, statement: str, evidence: Dict[str, Any]) -> float:
        """Validate latency claims against evidence."""
        latency_match = re.search(r'(\d+(?:\.\d+)?)\s*ms', statement, re.IGNORECASE)

        log_analysis = evidence.get("log_analysis", "")
        if "latency" in log_analysis.lower():
            evidence_match = re.search(r'(\d+(?:\.\d+)?)\s*ms', log_analysis)
            if evidence_match and latency_match:
                claimed_latency = float(latency_match.group(1))
                evidence_latency = float(evidence_match.group(1))
                # Validate if claim is reasonable
                if abs(claimed_latency - evidence_latency) <= 20:  # Within 20ms tolerance
                    return 0.9
                else:
                    return 0.3
            else:
                return 0.6  # General latency issue mentioned
        else:
            return 0.2

    def _validate_congestion_claim(self, statement: str, evidence: Dict[str, Any]) -> float:
        """Validate congestion claims against evidence."""
        # Check for congestion indicators in evidence
        indicators = ['congestion', 'utilization', 'capacity', 'overload']

        log_analysis = evidence.get("log_analysis", "").lower()
        kpi_metrics = evidence.get("kpi_metrics", "").lower()

        evidence_score = 0.0
        for indicator in indicators:
            if indicator in log_analysis or indicator in kpi_metrics:
                evidence_score += 0.25

        return min(evidence_score, 1.0)

    def _generate_traffic_recommendations(
        self, statement: str, issues: List[str]
    ) -> List[str]:
        """Generate recommendations for traffic-related validation."""
        recommendations = []

        if "Packet loss claim not supported" in str(issues):
            recommendations.append("Verify packet loss measurements with network monitoring tools")

        if "Latency claim not supported" in str(issues):
            recommendations.append("Cross-reference latency measurements with probe data")

        if "Congestion claim lacks" in str(issues):
            recommendations.append("Review utilization metrics and capacity planning data")

        if not recommendations:
            recommendations.append("Traffic validation passed - proceed with hypothesis")

        return recommendations


class ConfigDiffValidator(BaseValidator):
    """Validates hypotheses related to configuration changes and drift."""

    def __init__(self):
        super().__init__("config_diff_checker")

    def validate(
        self,
        hypothesis: Hypothesis,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate configuration-related claims."""

        issues = []
        verified_citations = []
        statement_lower = hypothesis.statement.lower()

        # Check for configuration-related claims
        config_terms = ['configuration', 'config', 'qos', 'policy', 'parameter', 'setting']
        has_config_claim = any(term in statement_lower for term in config_terms)

        if not has_config_claim:
            return ValidationResult(
                hypothesis_index=hypothesis.index,
                verdict=ValidationVerdict.GROUNDED,
                confidence_score=hypothesis.confidence,
                validation_details={"applicable": False},
                validator_name=self.name,
                citations_verified=[],
                issues_found=["Not applicable to configuration validation"],
                recommendations=[]
            )

        # Check for recent configuration changes
        change_evidence = self._check_recent_changes(evidence)
        config_consistency = self._check_config_consistency(evidence)

        confidence_score = (change_evidence + config_consistency) / 2.0

        # Special handling for QoS configuration claims
        if 'qos' in statement_lower:
            qos_validation = self._validate_qos_claims(statement_lower, evidence)
            confidence_score = (confidence_score + qos_validation) / 2.0

            if qos_validation > 0.6:
                verified_citations.append("qos_policy_audit")
            else:
                issues.append("QoS configuration claim requires additional verification")

        # Determine verdict
        if confidence_score >= 0.7 and len(issues) == 0:
            verdict = ValidationVerdict.GROUNDED
        elif confidence_score >= 0.4:
            verdict = ValidationVerdict.RETRY
        else:
            verdict = ValidationVerdict.REJECTED

        recommendations = [
            "Review recent configuration change logs",
            "Compare current configuration with baseline",
            "Verify configuration consistency across network elements"
        ]

        return ValidationResult(
            hypothesis_index=hypothesis.index,
            verdict=verdict,
            confidence_score=confidence_score,
            validation_details={
                "recent_changes_detected": change_evidence > 0.5,
                "config_consistency_score": config_consistency,
                "qos_specific": 'qos' in statement_lower,
            },
            validator_name=self.name,
            citations_verified=verified_citations,
            issues_found=issues,
            recommendations=recommendations,
        )

    def _check_recent_changes(self, evidence: Dict[str, Any]) -> float:
        """Check for evidence of recent configuration changes."""
        # Simulate change detection
        fixtures = evidence.get("fixtures", {})

        # Look for change-related indicators in evidence
        change_indicators = 0
        total_checks = 3

        # Check 1: Change logs or audit trails
        log_analysis = evidence.get("log_analysis", "").lower()
        if any(term in log_analysis for term in ["change", "config", "update", "modify"]):
            change_indicators += 1

        # Check 2: Configuration drift indicators
        if "drift" in log_analysis or "inconsistency" in log_analysis:
            change_indicators += 1

        # Check 3: Recent maintenance activities
        if any(term in log_analysis for term in ["maintenance", "upgrade", "patch"]):
            change_indicators += 1

        return change_indicators / total_checks

    def _check_config_consistency(self, evidence: Dict[str, Any]) -> float:
        """Check configuration consistency across network elements."""
        # Simulate consistency checking
        topology_check = evidence.get("topology_check", "").lower()

        if "no drift detected" in topology_check:
            return 0.9
        elif "minor inconsistencies" in topology_check:
            return 0.6
        elif "significant drift" in topology_check:
            return 0.2
        else:
            return 0.5  # Unknown state

    def _validate_qos_claims(self, statement: str, evidence: Dict[str, Any]) -> float:
        """Validate QoS-specific configuration claims."""
        # Look for QoS-related evidence
        qos_indicators = ['qos', 'priority', 'bandwidth', 'scheduling', 'traffic class']

        evidence_text = " ".join([
            evidence.get("log_analysis", ""),
            evidence.get("kpi_metrics", ""),
            str(evidence.get("fixtures", {}))
        ]).lower()

        matches = sum(1 for indicator in qos_indicators if indicator in evidence_text)
        return min(matches / len(qos_indicators), 1.0)


class TopologyValidator(BaseValidator):
    """Validates hypotheses related to network topology and connectivity."""

    def __init__(self):
        super().__init__("topology_analyzer")

    def validate(
        self,
        hypothesis: Hypothesis,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Validate topology-related claims."""

        issues = []
        verified_citations = []
        statement_lower = hypothesis.statement.lower()

        # Check for topology-related claims
        topology_terms = ['sector', 'backhaul', 'fiber', 'mme', 'sgw', 'topology', 'route', 'link']
        has_topology_claim = any(term in statement_lower for term in topology_terms)

        if not has_topology_claim:
            return ValidationResult(
                hypothesis_index=hypothesis.index,
                verdict=ValidationVerdict.GROUNDED,
                confidence_score=hypothesis.confidence,
                validation_details={"applicable": False},
                validator_name=self.name,
                citations_verified=[],
                issues_found=["Not applicable to topology validation"],
                recommendations=[]
            )

        # Validate specific topology claims
        sector_validation = self._validate_sector_claims(statement_lower, evidence)
        backhaul_validation = self._validate_backhaul_claims(statement_lower, evidence)
        equipment_validation = self._validate_equipment_references(statement_lower, evidence)

        validations = [v for v in [sector_validation, backhaul_validation, equipment_validation] if v is not None]

        if validations:
            confidence_score = sum(validations) / len(validations)
        else:
            confidence_score = 0.5

        # Check for supporting topology data
        topology_check = evidence.get("topology_check", "")
        if topology_check and any(term in topology_check.lower() for term in topology_terms):
            verified_citations.append("topology_snapshot")
            confidence_score = min(confidence_score + 0.1, 1.0)

        # Determine verdict
        if confidence_score >= 0.75:
            verdict = ValidationVerdict.GROUNDED
        elif confidence_score >= 0.5:
            verdict = ValidationVerdict.RETRY
        else:
            verdict = ValidationVerdict.REJECTED

        return ValidationResult(
            hypothesis_index=hypothesis.index,
            verdict=verdict,
            confidence_score=confidence_score,
            validation_details={
                "sector_claims_validated": sector_validation is not None,
                "backhaul_claims_validated": backhaul_validation is not None,
                "equipment_references_validated": equipment_validation is not None,
            },
            validator_name=self.name,
            citations_verified=verified_citations,
            issues_found=issues,
            recommendations=[
                "Verify topology data accuracy",
                "Cross-check equipment identifiers",
                "Validate network connectivity paths"
            ],
        )

    def _validate_sector_claims(self, statement: str, evidence: Dict[str, Any]) -> Optional[float]:
        """Validate sector-specific claims."""
        sector_match = re.search(r'sector\s+(\w+)', statement)
        if not sector_match:
            return None

        sector_id = sector_match.group(1)

        # Check if sector exists in topology data
        topology_check = evidence.get("topology_check", "").lower()
        if sector_id.lower() in topology_check:
            return 0.8
        else:
            return 0.3

    def _validate_backhaul_claims(self, statement: str, evidence: Dict[str, Any]) -> Optional[float]:
        """Validate backhaul-related claims."""
        if 'backhaul' not in statement:
            return None

        # Look for backhaul evidence
        topology_check = evidence.get("topology_check", "").lower()
        if any(term in topology_check for term in ['backhaul', 'fiber', 'link', 'degradation']):
            return 0.7
        else:
            return 0.4

    def _validate_equipment_references(self, statement: str, evidence: Dict[str, Any]) -> Optional[float]:
        """Validate equipment identifier references."""
        # Look for equipment patterns like MME-CLSTR-3, SGW-CORE-2
        equipment_pattern = r'(\w{2,5}-\w{2,8}-\d+)'
        equipment_matches = re.findall(equipment_pattern, statement, re.IGNORECASE)

        if not equipment_matches:
            return None

        topology_check = evidence.get("topology_check", "").lower()
        validated_count = 0

        for equipment in equipment_matches:
            if equipment.lower() in topology_check:
                validated_count += 1

        return validated_count / len(equipment_matches) if equipment_matches else 0.5


class HypothesisValidatorOrchestrator:
    """Orchestrates multiple validators for comprehensive hypothesis validation."""

    def __init__(self):
        self.validators = [
            TrafficProbeValidator(),
            ConfigDiffValidator(),
            TopologyValidator(),
        ]

    def validate_hypothesis(
        self,
        hypothesis: Hypothesis,
        evidence: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """Run all applicable validators on a hypothesis."""

        if context is None:
            context = {}

        results = []

        for validator in self.validators:
            try:
                result = validator.validate(hypothesis, evidence, context)
                results.append(result)
            except Exception as e:
                # Log error and continue with other validators
                print(f"[Validator] Error in {validator.name}: {e}")
                continue

        return results

    def get_consensus_verdict(
        self,
        validation_results: List[ValidationResult]
    ) -> Tuple[ValidationVerdict, float, List[str]]:
        """Determine consensus verdict from multiple validators."""

        applicable_results = [r for r in validation_results
                            if r.validation_details.get("applicable", True)]

        if not applicable_results:
            return ValidationVerdict.GROUNDED, 0.5, ["No applicable validators"]

        # Calculate weighted consensus
        verdict_scores = {
            ValidationVerdict.GROUNDED: 0,
            ValidationVerdict.RETRY: 0,
            ValidationVerdict.REJECTED: 0,
        }

        total_confidence = 0.0
        all_issues = []

        for result in applicable_results:
            verdict_scores[result.verdict] += result.confidence_score
            total_confidence += result.confidence_score
            all_issues.extend(result.issues_found)

        avg_confidence = total_confidence / len(applicable_results)

        # Determine consensus verdict
        max_verdict = max(verdict_scores.items(), key=lambda x: x[1])
        consensus_verdict = max_verdict[0]

        return consensus_verdict, avg_confidence, all_issues


__all__ = [
    "BaseValidator",
    "TrafficProbeValidator",
    "ConfigDiffValidator",
    "TopologyValidator",
    "HypothesisValidatorOrchestrator",
    "ValidationResult",
    "ValidationVerdict",
]