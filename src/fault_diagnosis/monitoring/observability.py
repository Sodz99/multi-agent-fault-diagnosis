"""Comprehensive monitoring and observability for fault diagnosis multi-agent system."""
from __future__ import annotations

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading
import uuid

import boto3
from botocore.exceptions import ClientError


class MetricType(Enum):
    """Types of metrics collected by the monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    dimensions: Dict[str, str] = field(default_factory=dict)
    unit: str = "Count"


@dataclass
class Alert:
    """System alert with context and severity."""
    id: str
    message: str
    severity: AlertSeverity
    timestamp: float
    source: str
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


@dataclass
class WorkflowMetrics:
    """Comprehensive metrics for a workflow execution."""
    workflow_id: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "running"

    # Stage timing
    stage_times: Dict[str, float] = field(default_factory=dict)
    stage_success: Dict[str, bool] = field(default_factory=dict)

    # Agent performance
    agent_execution_times: Dict[str, float] = field(default_factory=dict)
    agent_token_usage: Dict[str, int] = field(default_factory=dict)
    agent_costs: Dict[str, float] = field(default_factory=dict)
    agent_model_used: Dict[str, str] = field(default_factory=dict)

    # Results quality
    hypothesis_count: int = 0
    grounded_hypothesis_count: int = 0
    confidence_scores: List[float] = field(default_factory=list)

    # Resource usage
    total_cost: float = 0.0
    total_tokens: int = 0
    total_rag_queries: int = 0
    cache_hits: int = 0

    # Errors and issues
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)


class FaultDiagnosisMonitor:
    """Comprehensive monitoring and observability system for fault diagnosis workflows."""

    def __init__(
        self,
        enable_cloudwatch: bool = True,
        enable_local_logging: bool = True,
        log_directory: Optional[Path] = None,
        cloudwatch_namespace: str = "FaultDiagnosis",
        alert_threshold_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """Initialize the monitoring system.

        Args:
            enable_cloudwatch: Enable CloudWatch metrics publishing
            enable_local_logging: Enable local file logging
            log_directory: Directory for local log files
            cloudwatch_namespace: CloudWatch namespace for metrics
            alert_threshold_config: Configuration for alert thresholds
            verbose: Enable detailed console logging
        """
        self.enable_cloudwatch = enable_cloudwatch
        self.enable_local_logging = enable_local_logging
        self.cloudwatch_namespace = cloudwatch_namespace
        self.verbose = verbose

        # Initialize CloudWatch client
        self.cloudwatch_client = None
        if enable_cloudwatch:
            try:
                self.cloudwatch_client = boto3.client('cloudwatch')
                if verbose:
                    print("[Monitor] ☁️ CloudWatch monitoring enabled")
            except Exception as e:
                if verbose:
                    print(f"[Monitor] ⚠️ CloudWatch initialization failed: {e}")
                self.enable_cloudwatch = False

        # Initialize local logging
        self.log_directory = log_directory or Path("./logs")
        if enable_local_logging:
            self.log_directory.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"[Monitor] 📝 Local logging enabled: {self.log_directory}")

        # Alert thresholds
        self.alert_thresholds = {
            "cost_per_hour": 10.0,           # USD
            "workflow_duration": 600.0,      # seconds (10 minutes)
            "error_rate": 0.1,               # 10%
            "avg_response_time": 5000.0,     # milliseconds
            "hypothesis_quality": 0.5,       # minimum average confidence
            "token_usage_per_hour": 100000,  # tokens
            **alert_threshold_config or {}
        }

        # In-memory metric storage
        self.metrics_buffer = deque(maxlen=10000)  # Keep last 10k metrics
        self.active_workflows = {}
        self.completed_workflows = deque(maxlen=1000)  # Keep last 1000 workflows
        self.active_alerts = []
        self.resolved_alerts = deque(maxlen=500)

        # Performance tracking
        self.hourly_metrics = defaultdict(lambda: {
            "workflows": 0,
            "cost": 0.0,
            "tokens": 0,
            "errors": 0,
            "avg_duration": 0.0
        })

        # Thread-safe metric collection
        self._metrics_lock = threading.Lock()

        # Background metric publishing (if CloudWatch enabled)
        self._publishing_thread = None
        self._stop_publishing = threading.Event()

        if self.enable_cloudwatch:
            self._start_metric_publishing()

        if verbose:
            print("[Monitor] 🔍 Fault Diagnosis Monitor initialized")
            print(f"[Monitor] 📊 Alert thresholds: {len(self.alert_thresholds)} configured")

    def start_workflow(self, workflow_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """Start monitoring a new workflow execution."""
        if not workflow_id:
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"

        workflow_metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            start_time=time.time()
        )

        with self._metrics_lock:
            self.active_workflows[workflow_id] = workflow_metrics

        self.record_metric("workflow_started", 1.0, MetricType.COUNTER, {
            "workflow_id": workflow_id
        })

        if self.verbose:
            print(f"[Monitor] 🚀 Started monitoring workflow: {workflow_id}")

        return workflow_id

    def end_workflow(
        self,
        workflow_id: str,
        status: str = "completed",
        final_results: Optional[Dict[str, Any]] = None
    ):
        """End monitoring for a workflow execution."""
        with self._metrics_lock:
            if workflow_id not in self.active_workflows:
                return

            workflow_metrics = self.active_workflows[workflow_id]
            workflow_metrics.end_time = time.time()
            workflow_metrics.status = status

            # Extract metrics from final results if provided
            if final_results:
                self._extract_metrics_from_results(workflow_metrics, final_results)

            # Calculate total duration
            duration = workflow_metrics.end_time - workflow_metrics.start_time

            # Record completion metrics
            self.record_metric("workflow_completed", 1.0, MetricType.COUNTER, {
                "workflow_id": workflow_id,
                "status": status
            })

            self.record_metric("workflow_duration", duration, MetricType.TIMER, {
                "workflow_id": workflow_id,
                "status": status
            })

            if workflow_metrics.total_cost > 0:
                self.record_metric("workflow_cost", workflow_metrics.total_cost, MetricType.GAUGE, {
                    "workflow_id": workflow_id
                })

            # Move to completed workflows
            self.completed_workflows.append(workflow_metrics)
            del self.active_workflows[workflow_id]

            # Check for alerts
            self._check_workflow_alerts(workflow_metrics)

        if self.verbose:
            print(f"[Monitor] ✅ Completed monitoring workflow: {workflow_id}")
            print(f"[Monitor] ⏱️ Duration: {duration:.2f}s, Cost: ${workflow_metrics.total_cost:.4f}")

    def record_stage_start(self, workflow_id: str, stage_name: str):
        """Record the start of a workflow stage."""
        with self._metrics_lock:
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].stage_times[f"{stage_name}_start"] = time.time()

        self.record_metric("stage_started", 1.0, MetricType.COUNTER, {
            "workflow_id": workflow_id,
            "stage": stage_name
        })

    def record_stage_end(self, workflow_id: str, stage_name: str, success: bool = True):
        """Record the end of a workflow stage."""
        end_time = time.time()

        with self._metrics_lock:
            if workflow_id in self.active_workflows:
                workflow_metrics = self.active_workflows[workflow_id]
                start_key = f"{stage_name}_start"

                if start_key in workflow_metrics.stage_times:
                    duration = end_time - workflow_metrics.stage_times[start_key]
                    workflow_metrics.stage_times[stage_name] = duration
                    workflow_metrics.stage_success[stage_name] = success

                    self.record_metric("stage_duration", duration, MetricType.TIMER, {
                        "workflow_id": workflow_id,
                        "stage": stage_name,
                        "success": str(success)
                    })

    def record_agent_performance(
        self,
        workflow_id: str,
        agent_role: str,
        execution_time: float,
        token_usage: int,
        cost: float,
        model_used: str,
        success: bool = True
    ):
        """Record performance metrics for an agent execution."""
        with self._metrics_lock:
            if workflow_id in self.active_workflows:
                workflow_metrics = self.active_workflows[workflow_id]
                workflow_metrics.agent_execution_times[agent_role] = execution_time
                workflow_metrics.agent_token_usage[agent_role] = token_usage
                workflow_metrics.agent_costs[agent_role] = cost
                workflow_metrics.agent_model_used[agent_role] = model_used
                workflow_metrics.total_cost += cost
                workflow_metrics.total_tokens += token_usage

        # Record individual agent metrics
        dimensions = {
            "workflow_id": workflow_id,
            "agent_role": agent_role,
            "model": model_used,
            "success": str(success)
        }

        self.record_metric("agent_execution_time", execution_time, MetricType.TIMER, dimensions)
        self.record_metric("agent_token_usage", token_usage, MetricType.GAUGE, dimensions)
        self.record_metric("agent_cost", cost, MetricType.GAUGE, dimensions)

        if self.verbose:
            print(f"[Monitor] 🤖 Agent {agent_role}: {execution_time:.2f}s, {token_usage} tokens, ${cost:.4f}")

    def record_rag_performance(
        self,
        workflow_id: str,
        query_time: float,
        results_count: int,
        cache_hit: bool = False,
        cost: float = 0.0
    ):
        """Record RAG pipeline performance metrics."""
        with self._metrics_lock:
            if workflow_id in self.active_workflows:
                workflow_metrics = self.active_workflows[workflow_id]
                workflow_metrics.total_rag_queries += 1
                if cache_hit:
                    workflow_metrics.cache_hits += 1
                workflow_metrics.total_cost += cost

        dimensions = {
            "workflow_id": workflow_id,
            "cache_hit": str(cache_hit)
        }

        self.record_metric("rag_query_time", query_time, MetricType.TIMER, dimensions)
        self.record_metric("rag_results_count", results_count, MetricType.GAUGE, dimensions)
        if cost > 0:
            self.record_metric("rag_cost", cost, MetricType.GAUGE, dimensions)

    def record_error(
        self,
        workflow_id: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record an error occurrence."""
        error_data = {
            "type": error_type,
            "message": error_message,
            "timestamp": time.time(),
            "context": context or {}
        }

        with self._metrics_lock:
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id].errors.append(error_data)

        self.record_metric("error_occurred", 1.0, MetricType.COUNTER, {
            "workflow_id": workflow_id,
            "error_type": error_type
        })

        # Create alert for errors
        alert = Alert(
            id=f"error_{uuid.uuid4().hex[:8]}",
            message=f"Error in workflow {workflow_id}: {error_message}",
            severity=AlertSeverity.ERROR,
            timestamp=time.time(),
            source="workflow_execution",
            context={"workflow_id": workflow_id, "error_type": error_type}
        )
        self._add_alert(alert)

        if self.verbose:
            print(f"[Monitor] ❌ Error in {workflow_id}: {error_type} - {error_message}")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        dimensions: Optional[Dict[str, str]] = None,
        unit: str = "Count"
    ):
        """Record a custom metric."""
        metric = MetricPoint(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            dimensions=dimensions or {},
            unit=unit
        )

        with self._metrics_lock:
            self.metrics_buffer.append(metric)

        # Log locally if enabled
        if self.enable_local_logging:
            self._log_metric_locally(metric)

    def get_workflow_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a workflow."""
        workflow_metrics = None

        with self._metrics_lock:
            # Check active workflows first
            if workflow_id in self.active_workflows:
                workflow_metrics = self.active_workflows[workflow_id]
            else:
                # Check completed workflows
                for completed in self.completed_workflows:
                    if completed.workflow_id == workflow_id:
                        workflow_metrics = completed
                        break

        if not workflow_metrics:
            return None

        duration = (workflow_metrics.end_time or time.time()) - workflow_metrics.start_time

        return {
            "workflow_id": workflow_id,
            "status": workflow_metrics.status,
            "duration_seconds": duration,
            "total_cost_usd": workflow_metrics.total_cost,
            "total_tokens": workflow_metrics.total_tokens,
            "stage_performance": {
                stage: {
                    "duration": workflow_metrics.stage_times.get(stage, 0),
                    "success": workflow_metrics.stage_success.get(stage, False)
                }
                for stage in workflow_metrics.stage_success.keys()
            },
            "agent_performance": {
                role: {
                    "execution_time": workflow_metrics.agent_execution_times.get(role, 0),
                    "tokens": workflow_metrics.agent_token_usage.get(role, 0),
                    "cost": workflow_metrics.agent_costs.get(role, 0),
                    "model": workflow_metrics.agent_model_used.get(role, "unknown")
                }
                for role in workflow_metrics.agent_execution_times.keys()
            },
            "quality_metrics": {
                "hypothesis_count": workflow_metrics.hypothesis_count,
                "grounded_count": workflow_metrics.grounded_hypothesis_count,
                "average_confidence": statistics.mean(workflow_metrics.confidence_scores) if workflow_metrics.confidence_scores else 0.0,
                "rag_queries": workflow_metrics.total_rag_queries,
                "cache_hit_rate": workflow_metrics.cache_hits / max(workflow_metrics.total_rag_queries, 1)
            },
            "errors": workflow_metrics.errors,
            "warnings": workflow_metrics.warnings
        }

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        current_time = time.time()
        hour_ago = current_time - 3600

        # Get recent metrics
        recent_workflows = [
            wf for wf in self.completed_workflows
            if wf.end_time and wf.end_time > hour_ago
        ]

        # Calculate system metrics
        total_workflows = len(recent_workflows)
        successful_workflows = len([wf for wf in recent_workflows if wf.status == "completed"])

        avg_duration = statistics.mean([
            wf.end_time - wf.start_time for wf in recent_workflows
        ]) if recent_workflows else 0.0

        total_cost = sum(wf.total_cost for wf in recent_workflows)
        total_tokens = sum(wf.total_tokens for wf in recent_workflows)

        error_count = sum(len(wf.errors) for wf in recent_workflows)

        # Active alerts by severity
        alert_counts = defaultdict(int)
        for alert in self.active_alerts:
            alert_counts[alert.severity.value] += 1

        return {
            "timestamp": current_time,
            "performance": {
                "workflows_last_hour": total_workflows,
                "success_rate": successful_workflows / max(total_workflows, 1),
                "avg_duration_seconds": avg_duration,
                "active_workflows": len(self.active_workflows)
            },
            "cost_metrics": {
                "cost_last_hour_usd": total_cost,
                "tokens_last_hour": total_tokens,
                "avg_cost_per_workflow": total_cost / max(total_workflows, 1)
            },
            "quality_metrics": {
                "error_count_last_hour": error_count,
                "error_rate": error_count / max(total_workflows, 1)
            },
            "alerts": {
                "active_count": len(self.active_alerts),
                "by_severity": dict(alert_counts),
                "recent_critical": [
                    alert.message for alert in self.active_alerts
                    if alert.severity == AlertSeverity.CRITICAL
                ][-5:]  # Last 5 critical alerts
            },
            "resource_usage": {
                "metrics_buffer_size": len(self.metrics_buffer),
                "completed_workflows_stored": len(self.completed_workflows)
            }
        }

    def _extract_metrics_from_results(self, workflow_metrics: WorkflowMetrics, results: Dict[str, Any]):
        """Extract additional metrics from workflow results."""
        # Extract hypothesis metrics
        hypotheses = results.get("hypotheses", [])
        if hypotheses:
            workflow_metrics.hypothesis_count = len(hypotheses)
            workflow_metrics.grounded_hypothesis_count = len([
                h for h in hypotheses if getattr(h, 'verdict', '') == 'Grounded'
            ])

            confidence_scores = [
                getattr(h, 'confidence', 0.0) for h in hypotheses
            ]
            workflow_metrics.confidence_scores = confidence_scores

        # Extract routing metrics if available
        routing_metrics = results.get("routing_metrics", {})
        if routing_metrics:
            cost_info = routing_metrics.get("cost_optimization", {})
            if cost_info.get("total_estimated_cost_usd"):
                workflow_metrics.total_cost = max(
                    workflow_metrics.total_cost,
                    cost_info["total_estimated_cost_usd"]
                )

    def _check_workflow_alerts(self, workflow_metrics: WorkflowMetrics):
        """Check if workflow metrics trigger any alerts."""
        duration = workflow_metrics.end_time - workflow_metrics.start_time

        # Check duration threshold
        if duration > self.alert_thresholds["workflow_duration"]:
            alert = Alert(
                id=f"duration_{uuid.uuid4().hex[:8]}",
                message=f"Workflow {workflow_metrics.workflow_id} exceeded duration threshold: {duration:.2f}s",
                severity=AlertSeverity.WARNING,
                timestamp=time.time(),
                source="performance_monitor",
                context={"workflow_id": workflow_metrics.workflow_id, "duration": duration}
            )
            self._add_alert(alert)

        # Check cost threshold
        if workflow_metrics.total_cost > self.alert_thresholds.get("cost_per_workflow", 1.0):
            alert = Alert(
                id=f"cost_{uuid.uuid4().hex[:8]}",
                message=f"Workflow {workflow_metrics.workflow_id} exceeded cost threshold: ${workflow_metrics.total_cost:.4f}",
                severity=AlertSeverity.WARNING,
                timestamp=time.time(),
                source="cost_monitor",
                context={"workflow_id": workflow_metrics.workflow_id, "cost": workflow_metrics.total_cost}
            )
            self._add_alert(alert)

        # Check hypothesis quality
        if workflow_metrics.confidence_scores:
            avg_confidence = statistics.mean(workflow_metrics.confidence_scores)
            if avg_confidence < self.alert_thresholds["hypothesis_quality"]:
                alert = Alert(
                    id=f"quality_{uuid.uuid4().hex[:8]}",
                    message=f"Workflow {workflow_metrics.workflow_id} has low hypothesis quality: {avg_confidence:.2f}",
                    severity=AlertSeverity.INFO,
                    timestamp=time.time(),
                    source="quality_monitor",
                    context={"workflow_id": workflow_metrics.workflow_id, "avg_confidence": avg_confidence}
                )
                self._add_alert(alert)

    def _add_alert(self, alert: Alert):
        """Add a new alert to the system."""
        with self._metrics_lock:
            self.active_alerts.append(alert)

            # Keep only recent alerts (last 100 active)
            if len(self.active_alerts) > 100:
                oldest_alert = self.active_alerts.pop(0)
                oldest_alert.resolved = True
                oldest_alert.resolution_timestamp = time.time()
                self.resolved_alerts.append(oldest_alert)

        if self.verbose and alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            severity_icon = "🚨" if alert.severity == AlertSeverity.CRITICAL else "⚠️"
            print(f"[Monitor] {severity_icon} {alert.severity.value.upper()}: {alert.message}")

    def _log_metric_locally(self, metric: MetricPoint):
        """Log metric to local file."""
        if not self.enable_local_logging:
            return

        try:
            log_file = self.log_directory / f"metrics_{time.strftime('%Y%m%d')}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(asdict(metric)) + '\n')
        except Exception as e:
            if self.verbose:
                print(f"[Monitor] ⚠️ Failed to log metric locally: {e}")

    def _start_metric_publishing(self):
        """Start background thread for publishing metrics to CloudWatch."""
        def publish_metrics():
            while not self._stop_publishing.wait(60):  # Publish every minute
                self._publish_metrics_to_cloudwatch()

        self._publishing_thread = threading.Thread(target=publish_metrics, daemon=True)
        self._publishing_thread.start()

    def _publish_metrics_to_cloudwatch(self):
        """Publish buffered metrics to CloudWatch."""
        if not self.cloudwatch_client:
            return

        # Get metrics to publish
        metrics_to_publish = []
        with self._metrics_lock:
            if self.metrics_buffer:
                metrics_to_publish = list(self.metrics_buffer)
                self.metrics_buffer.clear()

        if not metrics_to_publish:
            return

        try:
            # Group metrics by namespace and dimensions for efficient publishing
            metric_data = []

            for metric in metrics_to_publish[-100:]:  # Publish last 100 metrics
                dimensions = [
                    {"Name": k, "Value": v} for k, v in metric.dimensions.items()
                ]

                metric_data.append({
                    "MetricName": metric.name,
                    "Value": metric.value,
                    "Unit": metric.unit,
                    "Timestamp": metric.timestamp,
                    "Dimensions": dimensions
                })

            # Publish in batches (CloudWatch limit is 20 metrics per call)
            for i in range(0, len(metric_data), 20):
                batch = metric_data[i:i + 20]
                self.cloudwatch_client.put_metric_data(
                    Namespace=self.cloudwatch_namespace,
                    MetricData=batch
                )

            if self.verbose:
                print(f"[Monitor] ☁️ Published {len(metric_data)} metrics to CloudWatch")

        except ClientError as e:
            if self.verbose:
                print(f"[Monitor] ⚠️ CloudWatch publishing failed: {e}")

    def shutdown(self):
        """Shutdown the monitoring system gracefully."""
        if self._publishing_thread:
            self._stop_publishing.set()
            self._publishing_thread.join(timeout=5)

        # Publish any remaining metrics
        if self.enable_cloudwatch:
            self._publish_metrics_to_cloudwatch()

        if self.verbose:
            print("[Monitor] 🔍 Fault Diagnosis Monitor shutdown complete")


__all__ = [
    "FaultDiagnosisMonitor",
    "MetricType",
    "AlertSeverity",
    "MetricPoint",
    "Alert",
    "WorkflowMetrics",
]