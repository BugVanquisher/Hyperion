"""
Alert handling and notification system for Hyperion.

Processes alerts from Alertmanager and provides ML-aware alert processing.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AlertStatus(str, Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertComponent(str, Enum):
    GPU = "gpu"
    ML_INFERENCE = "ml-inference"
    BATCH_PROCESSING = "batch-processing"
    CACHE = "cache"
    API = "api"


class AlertmanagerAlert(BaseModel):
    """Single alert from Alertmanager."""

    status: AlertStatus
    labels: Dict[str, str]
    annotations: Dict[str, str]
    startsAt: str
    endsAt: Optional[str] = None
    generatorURL: str
    fingerprint: str


class AlertmanagerWebhook(BaseModel):
    """Webhook payload from Alertmanager."""

    receiver: str
    status: AlertStatus
    alerts: List[AlertmanagerAlert]
    groupLabels: Dict[str, str]
    commonLabels: Dict[str, str]
    commonAnnotations: Dict[str, str]
    externalURL: str
    version: str = "4"
    groupKey: str
    truncatedAlerts: int = 0


class AlertProcessor:
    """Processes and enriches alerts with ML context."""

    def __init__(self):
        self.alert_history: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, AlertmanagerAlert] = {}

    def process_webhook(self, webhook: AlertmanagerWebhook) -> Dict[str, Any]:
        """Process incoming Alertmanager webhook."""

        processed_alerts = []
        for alert in webhook.alerts:
            enriched_alert = self._enrich_alert(alert)
            processed_alerts.append(enriched_alert)

            # Update active alerts tracking
            if alert.status == AlertStatus.FIRING:
                self.active_alerts[alert.fingerprint] = alert
            elif alert.status == AlertStatus.RESOLVED:
                self.active_alerts.pop(alert.fingerprint, None)

        # Store in history
        alert_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "receiver": webhook.receiver,
            "status": webhook.status,
            "alert_count": len(webhook.alerts),
            "group_labels": webhook.groupLabels,
            "alerts": processed_alerts,
        }

        self.alert_history.append(alert_event)

        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        return alert_event

    def _enrich_alert(self, alert: AlertmanagerAlert) -> Dict[str, Any]:
        """Enrich alert with ML-specific context and analysis."""

        enriched = {
            "fingerprint": alert.fingerprint,
            "status": alert.status,
            "severity": alert.labels.get("severity", "unknown"),
            "component": alert.labels.get("component", "unknown"),
            "service": alert.labels.get("service", "unknown"),
            "alertname": alert.labels.get("alertname", "unknown"),
            "summary": alert.annotations.get("summary", ""),
            "description": alert.annotations.get("description", ""),
            "starts_at": alert.startsAt,
            "ends_at": alert.endsAt,
            "labels": alert.labels,
            "annotations": alert.annotations,
            "ml_context": self._extract_ml_context(alert),
        }

        # Add impact assessment
        enriched["impact_assessment"] = self._assess_alert_impact(alert)

        # Add suggested actions
        enriched["suggested_actions"] = self._get_suggested_actions(alert)

        return enriched

    def _extract_ml_context(self, alert: AlertmanagerAlert) -> Dict[str, Any]:
        """Extract ML-specific context from alert."""

        context = {}
        component = alert.labels.get("component", "")
        alertname = alert.labels.get("alertname", "")

        if component == "gpu":
            context.update(
                {
                    "gpu_name": alert.labels.get("gpu_name"),
                    "gpu_memory_usage": alert.labels.get("gpu_memory_usage"),
                    "gpu_utilization": alert.labels.get("gpu_utilization"),
                    "anomaly_type": alert.labels.get("anomaly"),
                }
            )

        elif component == "ml-inference":
            context.update(
                {
                    "model_name": alert.labels.get("model_name"),
                    "inference_time": alert.labels.get("inference_time"),
                    "failure_rate": alert.labels.get("failure_rate"),
                    "anomaly_type": alert.labels.get("anomaly"),
                }
            )

        elif component == "batch-processing":
            context.update(
                {
                    "avg_batch_size": alert.labels.get("avg_batch_size"),
                    "batch_duration": alert.labels.get("batch_duration"),
                    "queue_depth": alert.labels.get("queue_depth"),
                }
            )

        elif component == "cache":
            context.update(
                {
                    "cache_hit_rate": alert.labels.get("cache_hit_rate"),
                    "cache_size": alert.labels.get("cache_size"),
                    "cache_eviction_rate": alert.labels.get("cache_eviction_rate"),
                }
            )

        return context

    def _assess_alert_impact(self, alert: AlertmanagerAlert) -> Dict[str, Any]:
        """Assess the potential impact of the alert."""

        severity = alert.labels.get("severity", "unknown")
        component = alert.labels.get("component", "unknown")
        alertname = alert.labels.get("alertname", "")

        impact = {
            "user_facing": False,
            "performance_degradation": False,
            "cost_impact": False,
            "service_availability": "normal",
            "estimated_recovery_time": "unknown",
        }

        # GPU alerts
        if component == "gpu":
            if "Critical" in alertname or severity == "critical":
                impact.update(
                    {
                        "user_facing": True,
                        "performance_degradation": True,
                        "service_availability": "degraded",
                        "estimated_recovery_time": "5-15 minutes",
                    }
                )
            elif "MemoryHigh" in alertname:
                impact.update(
                    {
                        "performance_degradation": True,
                        "cost_impact": True,
                        "service_availability": "at_risk",
                    }
                )

        # Inference alerts
        elif component == "ml-inference":
            if "Down" in alertname or "Failure" in alertname:
                impact.update(
                    {
                        "user_facing": True,
                        "service_availability": "unavailable",
                        "estimated_recovery_time": "2-10 minutes",
                    }
                )
            elif "Latency" in alertname:
                impact.update(
                    {
                        "user_facing": True,
                        "performance_degradation": True,
                        "service_availability": "degraded",
                    }
                )

        # Batch processing alerts
        elif component == "batch-processing":
            impact.update(
                {
                    "performance_degradation": True,
                    "cost_impact": True,
                    "service_availability": (
                        "degraded" if "Stalled" in alertname else "at_risk"
                    ),
                }
            )

        return impact

    def _get_suggested_actions(self, alert: AlertmanagerAlert) -> List[str]:
        """Get suggested remediation actions for the alert."""

        alertname = alert.labels.get("alertname", "")
        component = alert.labels.get("component", "")
        actions = []

        if component == "gpu":
            if "MemoryHigh" in alertname:
                actions.extend(
                    [
                        "Check for memory leaks in model inference code",
                        "Consider reducing batch size temporarily",
                        "Monitor GPU memory allocation patterns",
                        "Review recent model deployments for memory inefficiencies",
                    ]
                )
            elif "MemoryCritical" in alertname:
                actions.extend(
                    [
                        "IMMEDIATE: Restart GPU-enabled pods to clear memory",
                        "Scale down non-essential GPU workloads",
                        "Implement emergency batch size reduction",
                        "Contact on-call engineer immediately",
                    ]
                )
            elif "MemoryLeak" in alertname:
                actions.extend(
                    [
                        "Profile GPU memory usage over time",
                        "Check for unreleased tensors in model code",
                        "Review garbage collection settings",
                        "Consider rolling restart of inference pods",
                    ]
                )

        elif component == "ml-inference":
            if "HighInferenceLatency" in alertname:
                actions.extend(
                    [
                        "Check GPU utilization and memory pressure",
                        "Review recent model changes or updates",
                        "Monitor batch processing efficiency",
                        "Consider horizontal scaling if sustained",
                    ]
                )
            elif "InferenceDown" in alertname:
                actions.extend(
                    [
                        "Check application logs for errors",
                        "Verify model loading status",
                        "Restart inference service pods",
                        "Check underlying infrastructure health",
                    ]
                )

        elif component == "batch-processing":
            if "BatchProcessingSlow" in alertname:
                actions.extend(
                    [
                        "Review batch size configuration",
                        "Check for processing bottlenecks",
                        "Monitor queue depth trends",
                        "Consider batch timeout adjustments",
                    ]
                )
            elif "BatchProcessingStalled" in alertname:
                actions.extend(
                    [
                        "Restart batch processing components",
                        "Check for deadlocks or blocking operations",
                        "Review batch processing logs for errors",
                        "Verify queue connectivity",
                    ]
                )

        elif component == "cache":
            if "CacheHitRateLow" in alertname:
                actions.extend(
                    [
                        "Review cache TTL settings",
                        "Check for cache key distribution issues",
                        "Monitor cache memory usage",
                        "Consider cache warming strategies",
                    ]
                )

        # Add general actions for all critical alerts
        if alert.labels.get("severity") == "critical":
            actions.insert(0, "CRITICAL: Page on-call engineer immediately")
            actions.insert(1, "Check service health dashboard")

        return actions

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of current alert state."""

        active_count = len(self.active_alerts)
        critical_count = sum(
            1
            for alert in self.active_alerts.values()
            if alert.labels.get("severity") == "critical"
        )

        components_affected = set(
            alert.labels.get("component", "unknown")
            for alert in self.active_alerts.values()
        )

        recent_alerts = len(
            [
                alert
                for alert in self.alert_history[-50:]
                if alert["status"] == AlertStatus.FIRING
            ]
        )

        return {
            "active_alerts": active_count,
            "critical_alerts": critical_count,
            "components_affected": list(components_affected),
            "recent_firing_alerts": recent_alerts,
            "alert_history_size": len(self.alert_history),
            "last_alert": (
                self.alert_history[-1]["timestamp"] if self.alert_history else None
            ),
        }


# Global alert processor instance
alert_processor = AlertProcessor()
