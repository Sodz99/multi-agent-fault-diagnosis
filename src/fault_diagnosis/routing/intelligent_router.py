"""Intelligent model routing for cost optimization and performance enhancement."""
from __future__ import annotations

import re
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics

from crewai import LLM


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    CRITICAL = "critical"


class ModelTier(Enum):
    """Model performance tiers."""
    COST_OPTIMIZED = "cost_optimized"    # Titan Express - fastest, cheapest
    BALANCED = "balanced"                # Titan Lite - basic tasks
    PERFORMANCE = "performance"          # Titan Premier - best reasoning
    PREMIUM = "premium"                  # Reserved for future premium models


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_string: str
    tier: ModelTier
    cost_per_1k_tokens: float
    avg_response_time_ms: float
    reasoning_capability: float  # 0.0 to 1.0
    context_window: int
    max_tokens: int
    description: str


@dataclass
class TaskAnalysis:
    """Analysis results for a specific task."""
    complexity: TaskComplexity
    confidence: float
    reasoning_required: float  # 0.0 to 1.0
    urgency: float  # 0.0 to 1.0
    token_estimate: int
    analysis_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Routing decision with rationale and cost estimates."""
    selected_model: str
    tier: ModelTier
    confidence: float
    estimated_cost: float
    estimated_time_ms: float
    rationale: str
    fallback_model: Optional[str] = None
    decision_factors: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingMetrics:
    """Comprehensive metrics for routing performance."""
    total_requests: int = 0
    successful_routes: int = 0
    cost_savings_usd: float = 0.0
    avg_decision_time_ms: float = 0.0
    complexity_distribution: Dict[TaskComplexity, int] = field(default_factory=lambda: {c: 0 for c in TaskComplexity})
    tier_usage: Dict[ModelTier, int] = field(default_factory=lambda: {t: 0 for t in ModelTier})
    total_estimated_cost: float = 0.0
    fallback_rate: float = 0.0


class IntelligentModelRouter:
    """Intelligent routing system for optimizing model selection based on task complexity and cost constraints."""

    # Predefined model configurations for Amazon Titan models
    MODEL_CONFIGS = {
        "bedrock/amazon.titan-text-express-v1": ModelConfig(
            model_string="bedrock/amazon.titan-text-express-v1",
            tier=ModelTier.COST_OPTIMIZED,
            cost_per_1k_tokens=0.0002,   # Cost-optimized Titan Express
            avg_response_time_ms=900,    # Fast response
            reasoning_capability=0.70,   # Good for most tasks
            context_window=8000,
            max_tokens=8192,
            description="Titan Text Express - Fast response, cost-optimized"
        ),
        "bedrock/amazon.titan-text-premier-v1:0": ModelConfig(
            model_string="bedrock/amazon.titan-text-premier-v1:0",
            tier=ModelTier.PERFORMANCE,
            cost_per_1k_tokens=0.0005,   # Premium Titan model
            avg_response_time_ms=2000,   # Balanced response time
            reasoning_capability=0.90,   # Superior reasoning
            context_window=32000,
            max_tokens=4096,
            description="Titan Text Premier - Superior reasoning, optimized for RAG"
        ),
        "bedrock/amazon.titan-text-lite-v1": ModelConfig(
            model_string="bedrock/amazon.titan-text-lite-v1",
            tier=ModelTier.BALANCED,
            cost_per_1k_tokens=0.00015,  # Lightweight model
            avg_response_time_ms=1200,   # Standard response time
            reasoning_capability=0.60,   # Basic reasoning
            context_window=4000,
            max_tokens=4096,
            description="Titan Text Lite - Lightweight, basic tasks"
        )
    }

    def __init__(
        self,
        cost_budget_per_hour: float = 5.0,
        performance_priority: float = 0.5,  # 0.0 = cost priority, 1.0 = performance priority
        enable_fallback: bool = True,
        verbose: bool = True
    ):
        """Initialize the intelligent model router.

        Args:
            cost_budget_per_hour: Maximum cost budget per hour in USD
            performance_priority: Balance between cost (0.0) and performance (1.0)
            enable_fallback: Enable automatic fallback to simpler models
            verbose: Enable detailed logging
        """
        self.cost_budget_per_hour = cost_budget_per_hour
        self.performance_priority = performance_priority
        self.enable_fallback = enable_fallback
        self.verbose = verbose

        # Initialize metrics
        self.metrics = RoutingMetrics()

        # Cost tracking
        self.hourly_cost_tracker = []
        self.current_hour_start = time.time()
        self.current_hour_cost = 0.0

        # Performance tracking
        self.model_performance_history = {model: [] for model in self.MODEL_CONFIGS.keys()}

        if self.verbose:
            print(f"[Router] 🧠 Intelligent Model Router initialized")
            print(f"[Router] 💰 Budget: ${cost_budget_per_hour:.2f}/hour")
            print(f"[Router] ⚖️ Priority: {performance_priority:.1f} (0=cost, 1=performance)")
            print(f"[Router] 🔄 Fallback: {'enabled' if enable_fallback else 'disabled'}")

    def analyze_task_complexity(
        self,
        task_description: str,
        agent_role: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskAnalysis:
        """Analyze task complexity to inform routing decisions."""
        analysis_start = time.time()

        # Initialize analysis factors
        factors = {}

        # Text-based complexity indicators
        text_length = len(task_description)
        word_count = len(task_description.split())
        sentence_count = len(re.split(r'[.!?]+', task_description))

        factors["text_metrics"] = {
            "length": text_length,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": word_count / max(sentence_count, 1)
        }

        # Complexity keywords analysis
        complex_keywords = {
            "analysis": 0.3, "synthesize": 0.4, "evaluate": 0.3, "compare": 0.3,
            "reasoning": 0.4, "inference": 0.4, "hypothesis": 0.3, "validate": 0.3,
            "root cause": 0.5, "diagnose": 0.4, "troubleshoot": 0.4,
            "optimization": 0.4, "strategy": 0.3, "decision": 0.3,
            "complex": 0.3, "sophisticated": 0.3, "advanced": 0.3,
            "multiple": 0.2, "various": 0.2, "comprehensive": 0.3
        }

        simple_keywords = {
            "list": -0.2, "show": -0.2, "display": -0.2, "fetch": -0.2,
            "get": -0.2, "retrieve": -0.2, "find": -0.2, "search": -0.2,
            "simple": -0.3, "basic": -0.3, "quick": -0.3, "fast": -0.2
        }

        # Calculate keyword-based complexity
        text_lower = task_description.lower()
        complexity_score = 0.0

        for keyword, weight in complex_keywords.items():
            if keyword in text_lower:
                complexity_score += weight

        for keyword, weight in simple_keywords.items():
            if keyword in text_lower:
                complexity_score += weight

        factors["keyword_complexity"] = complexity_score

        # Agent role-based complexity
        role_complexity = {
            "planner": 0.2,        # Usually straightforward orchestration
            "retriever": 0.1,      # Simple information retrieval
            "reasoner": 0.8,       # Complex analysis and reasoning
            "reporter": 0.3,       # Moderate synthesis and documentation
            "validator": 0.6,      # Validation requires careful analysis
            "escalation": 0.4      # Moderate complexity for human handoff
        }

        role_factor = role_complexity.get(agent_role.lower(), 0.5)
        factors["role_complexity"] = role_factor

        # Context-based complexity
        context_complexity = 0.0
        if context:
            # More data sources = higher complexity
            data_sources = len(context.get("fixtures", {}))
            if data_sources > 5:
                context_complexity += 0.2
            elif data_sources > 10:
                context_complexity += 0.4

            # Evidence complexity
            evidence_types = len(context.get("evidence_bundle", {}))
            context_complexity += min(evidence_types * 0.1, 0.3)

            # Alert severity impacts urgency, not necessarily complexity
            severity = context.get("alert_context", {}).get("severity", "medium")
            urgency_factor = {"low": 0.1, "medium": 0.5, "high": 0.8, "critical": 1.0}.get(severity, 0.5)
        else:
            urgency_factor = 0.5

        factors["context_complexity"] = context_complexity
        factors["urgency"] = urgency_factor

        # Calculate overall complexity score
        base_complexity = (
            min(text_length / 1000, 0.3) +      # Text length (normalized)
            min(word_count / 200, 0.3) +        # Word count (normalized)
            complexity_score +                   # Keyword analysis
            role_factor +                        # Agent role
            context_complexity                   # Context factors
        )

        # Normalize complexity score to 0-1 range
        normalized_complexity = min(max(base_complexity, 0.0), 1.0)

        # Determine complexity category
        if normalized_complexity < 0.3:
            complexity = TaskComplexity.SIMPLE
        elif normalized_complexity < 0.6:
            complexity = TaskComplexity.MODERATE
        elif normalized_complexity < 0.8:
            complexity = TaskComplexity.COMPLEX
        else:
            complexity = TaskComplexity.CRITICAL

        # Calculate reasoning requirement
        reasoning_required = min(max(
            complexity_score + role_factor + (normalized_complexity * 0.5),
            0.0
        ), 1.0)

        # Estimate token usage
        estimated_tokens = max(
            word_count * 1.5,  # Rough token estimate
            100  # Minimum token estimate
        )

        analysis_time = time.time() - analysis_start

        if self.verbose:
            print(f"[Router] 📊 Task analysis: {complexity.value} (score: {normalized_complexity:.2f})")
            print(f"[Router] 🧠 Reasoning required: {reasoning_required:.2f}")
            print(f"[Router] ⚡ Urgency: {urgency_factor:.2f}")
            print(f"[Router] 🎯 Estimated tokens: {estimated_tokens}")

        return TaskAnalysis(
            complexity=complexity,
            confidence=0.85,  # High confidence in analysis
            reasoning_required=reasoning_required,
            urgency=urgency_factor,
            token_estimate=estimated_tokens,
            analysis_factors=factors
        )

    def route_model(
        self,
        task_analysis: TaskAnalysis,
        agent_role: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """Route to the optimal model based on task analysis and constraints."""
        routing_start = time.time()

        # Check budget constraints
        self._update_hourly_cost_tracking()
        budget_remaining = self.cost_budget_per_hour - self.current_hour_cost
        budget_pressure = 1.0 - (budget_remaining / self.cost_budget_per_hour)

        # Get available models sorted by capability
        available_models = list(self.MODEL_CONFIGS.keys())

        # Calculate scores for each model
        model_scores = {}
        for model_string in available_models:
            config = self.MODEL_CONFIGS[model_string]
            score = self._calculate_model_score(task_analysis, config, budget_pressure)
            model_scores[model_string] = score

        # Select best model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        selected_model = best_model[0]
        best_config = self.MODEL_CONFIGS[selected_model]

        # Calculate cost and time estimates
        estimated_cost = (task_analysis.token_estimate / 1000.0) * best_config.cost_per_1k_tokens
        estimated_time = best_config.avg_response_time_ms * (1.0 + (task_analysis.reasoning_required * 0.5))

        # Determine fallback model
        fallback_model = None
        if self.enable_fallback:
            fallback_model = self._select_fallback_model(selected_model, task_analysis)

        # Build rationale
        rationale_parts = []

        if task_analysis.complexity == TaskComplexity.SIMPLE:
            rationale_parts.append("Simple task → cost-optimized model")
        elif task_analysis.complexity == TaskComplexity.COMPLEX:
            rationale_parts.append("Complex task → high-reasoning model")
        else:
            rationale_parts.append("Moderate task → balanced model")

        if budget_pressure > 0.7:
            rationale_parts.append("high budget pressure")
        elif budget_pressure < 0.3:
            rationale_parts.append("low budget pressure")

        if task_analysis.urgency > 0.8:
            rationale_parts.append("high urgency → fast model")

        rationale = ", ".join(rationale_parts)

        routing_time = time.time() - routing_start

        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.successful_routes += 1
        self.metrics.complexity_distribution[task_analysis.complexity] += 1
        self.metrics.tier_usage[best_config.tier] += 1
        self.metrics.total_estimated_cost += estimated_cost
        self.metrics.avg_decision_time_ms = (
            (self.metrics.avg_decision_time_ms * (self.metrics.total_requests - 1) + routing_time * 1000)
            / self.metrics.total_requests
        )

        decision = RoutingDecision(
            selected_model=selected_model,
            tier=best_config.tier,
            confidence=0.9,  # High confidence in routing logic
            estimated_cost=estimated_cost,
            estimated_time_ms=estimated_time,
            rationale=rationale,
            fallback_model=fallback_model,
            decision_factors={
                "budget_pressure": budget_pressure,
                "model_scores": model_scores,
                "routing_time_ms": routing_time * 1000
            }
        )

        if self.verbose:
            print(f"[Router] 🎯 Selected: {best_config.description}")
            print(f"[Router] 💰 Estimated cost: ${estimated_cost:.4f}")
            print(f"[Router] ⏱️ Estimated time: {estimated_time:.0f}ms")
            print(f"[Router] 🔍 Rationale: {rationale}")
            if fallback_model:
                print(f"[Router] 🔄 Fallback: {self.MODEL_CONFIGS[fallback_model].description}")

        return decision

    def _calculate_model_score(
        self,
        task_analysis: TaskAnalysis,
        model_config: ModelConfig,
        budget_pressure: float
    ) -> float:
        """Calculate a score for a model based on task requirements and constraints."""

        # Base capability score
        capability_match = min(
            model_config.reasoning_capability / max(task_analysis.reasoning_required, 0.1),
            2.0  # Cap the advantage
        )

        # Cost efficiency score (higher is better for lower cost)
        max_cost = max(config.cost_per_1k_tokens for config in self.MODEL_CONFIGS.values())
        cost_efficiency = (max_cost - model_config.cost_per_1k_tokens) / max_cost

        # Speed score (higher is better for lower latency)
        max_time = max(config.avg_response_time_ms for config in self.MODEL_CONFIGS.values())
        speed_score = (max_time - model_config.avg_response_time_ms) / max_time

        # Urgency factor
        urgency_weight = task_analysis.urgency

        # Apply user preference weighting
        performance_weight = self.performance_priority
        cost_weight = 1.0 - self.performance_priority

        # Apply budget pressure
        cost_weight *= (1.0 + budget_pressure)

        # Calculate weighted score
        score = (
            capability_match * 0.4 +
            cost_efficiency * cost_weight * 0.35 +
            speed_score * (performance_weight + urgency_weight * 0.5) * 0.25
        )

        # Penalty for overkill (using high-end model for simple tasks)
        if (task_analysis.complexity == TaskComplexity.SIMPLE and
            model_config.tier in [ModelTier.PERFORMANCE, ModelTier.PREMIUM]):
            score *= 0.7  # 30% penalty

        return score

    def _select_fallback_model(self, primary_model: str, task_analysis: TaskAnalysis) -> Optional[str]:
        """Select an appropriate fallback model."""
        primary_config = self.MODEL_CONFIGS[primary_model]

        # For complex tasks, fallback to a simpler but still capable model
        if task_analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.CRITICAL]:
            # Fallback from Sonnet v2 to Sonnet v1, or from Sonnet to Haiku
            if primary_config.tier == ModelTier.PERFORMANCE:
                candidates = [m for m, c in self.MODEL_CONFIGS.items()
                             if c.tier == ModelTier.BALANCED and m != primary_model]
            else:
                candidates = [m for m, c in self.MODEL_CONFIGS.items()
                             if c.tier == ModelTier.COST_OPTIMIZED and m != primary_model]
        else:
            # For simple tasks, always fallback to the cheapest option
            candidates = [m for m, c in self.MODEL_CONFIGS.items()
                         if c.tier == ModelTier.COST_OPTIMIZED and m != primary_model]

        return candidates[0] if candidates else None

    def _update_hourly_cost_tracking(self):
        """Update hourly cost tracking and reset if needed."""
        current_time = time.time()

        # Check if an hour has passed
        if current_time - self.current_hour_start >= 3600:  # 1 hour
            # Archive current hour data
            self.hourly_cost_tracker.append({
                "hour_start": self.current_hour_start,
                "cost": self.current_hour_cost
            })

            # Reset for new hour
            self.current_hour_start = current_time
            self.current_hour_cost = 0.0

            # Keep only last 24 hours of data
            if len(self.hourly_cost_tracker) > 24:
                self.hourly_cost_tracker = self.hourly_cost_tracker[-24:]

    def record_actual_performance(
        self,
        model_string: str,
        actual_cost: float,
        actual_time_ms: float,
        success: bool
    ):
        """Record actual performance to improve future routing decisions."""
        self.current_hour_cost += actual_cost

        if model_string in self.model_performance_history:
            self.model_performance_history[model_string].append({
                "cost": actual_cost,
                "time_ms": actual_time_ms,
                "success": success,
                "timestamp": time.time()
            })

            # Keep only recent history (last 100 requests per model)
            if len(self.model_performance_history[model_string]) > 100:
                self.model_performance_history[model_string] = \
                    self.model_performance_history[model_string][-100:]

    def create_llm_for_agent(
        self,
        agent_role: str,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[LLM, RoutingDecision]:
        """Create an optimally routed LLM instance for a specific agent and task."""

        # Analyze task complexity
        task_analysis = self.analyze_task_complexity(task_description, agent_role, context)

        # Route to optimal model
        routing_decision = self.route_model(task_analysis, agent_role, context)

        # Create LLM instance
        llm = LLM(model=routing_decision.selected_model)

        if self.verbose:
            print(f"[Router] 🤖 Created LLM for {agent_role}: {routing_decision.selected_model}")

        return llm, routing_decision

    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing performance metrics."""
        total_requests = max(self.metrics.total_requests, 1)

        # Calculate cost savings compared to always using the most expensive model
        max_cost = max(config.cost_per_1k_tokens for config in self.MODEL_CONFIGS.values())
        theoretical_max_cost = total_requests * max_cost * 1000  # Assume 1000 tokens per request
        actual_cost = self.metrics.total_estimated_cost
        cost_savings = max(0, theoretical_max_cost - actual_cost)

        return {
            "performance_summary": {
                "total_requests": self.metrics.total_requests,
                "successful_routes": self.metrics.successful_routes,
                "success_rate_percent": (self.metrics.successful_routes / total_requests) * 100,
                "avg_decision_time_ms": round(self.metrics.avg_decision_time_ms, 2)
            },
            "cost_optimization": {
                "total_estimated_cost_usd": round(self.metrics.total_estimated_cost, 4),
                "estimated_cost_savings_usd": round(cost_savings, 4),
                "current_hour_cost_usd": round(self.current_hour_cost, 4),
                "budget_utilization_percent": round((self.current_hour_cost / self.cost_budget_per_hour) * 100, 1),
                "avg_cost_per_request_usd": round(self.metrics.total_estimated_cost / total_requests, 4)
            },
            "usage_distribution": {
                "complexity_distribution": {k.value: v for k, v in self.metrics.complexity_distribution.items()},
                "tier_usage": {k.value: v for k, v in self.metrics.tier_usage.items()},
                "fallback_rate_percent": round(self.metrics.fallback_rate * 100, 1)
            },
            "model_configurations": {
                model: {
                    "tier": config.tier.value,
                    "cost_per_1k_tokens": config.cost_per_1k_tokens,
                    "avg_response_time_ms": config.avg_response_time_ms,
                    "reasoning_capability": config.reasoning_capability
                }
                for model, config in self.MODEL_CONFIGS.items()
            }
        }


__all__ = [
    "IntelligentModelRouter",
    "TaskComplexity",
    "ModelTier",
    "TaskAnalysis",
    "RoutingDecision",
    "RoutingMetrics",
]