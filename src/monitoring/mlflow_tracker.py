"""
MLflow Tracker + Drift Monitoring
- Logs per-prediction metrics (risk_score, latency, entity_count)
- 5-run rolling drift detection
- Fires alerts when metrics drop >0.05 vs baseline
"""

import logging
import statistics
from collections import deque
from datetime import datetime
from typing import Optional

from src.config import settings

logger = logging.getLogger(__name__)

# Lazy import so missing mlflow doesn't crash startup
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore


class MLflowTracker:
    """
    Tracks triage pipeline metrics in MLflow.
    Maintains a rolling window for drift detection.
    Falls back to no-op if MLflow server is unreachable.
    """

    WINDOW_SIZE = 5
    BASELINE_RISK_SCORE = 0.5  # Expected average risk score
    BASELINE_LATENCY_MS = 3000.0  # Expected p50 latency

    def __init__(self):
        self._enabled = False
        self._experiment_name = settings.mlflow_experiment_name
        self._alerts: list[str] = []

        # ALWAYS initialize rolling windows first — drift detection works without MLflow
        self._risk_window: deque[float] = deque(maxlen=self.WINDOW_SIZE)
        self._latency_window: deque[float] = deque(maxlen=self.WINDOW_SIZE)

        # ── Gate 1: MLflow package installed? ───────────────────────────────
        if not _MLFLOW_AVAILABLE:
            logger.info("MLflow package not installed. Tracking disabled.")
            return

        # ── Gate 2: URI configured? ───────────────────────────────────────────
        uri = (settings.mlflow_tracking_uri or "").strip()
        if not uri:
            logger.info("MLFLOW_TRACKING_URI is empty. MLflow tracking disabled.")
            return

        # ── Gate 3: Can we actually connect? ─────────────────────────────────
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(self._experiment_name)
            self.client = MlflowClient()
            self._experiment = mlflow.get_experiment_by_name(self._experiment_name)
            self._enabled = True
            logger.info(f"MLflow connected to {uri}")
        except Exception as e:
            logger.warning(f"MLflow connection failed ({e}). Tracking disabled.")

    @property
    def experiment_id(self) -> Optional[str]:
        if not self._enabled or mlflow is None:
            return None
        exp = mlflow.get_experiment_by_name(self._experiment_name)
        return exp.experiment_id if exp else None

    def log_prediction(
        self,
        note_id: str,
        risk_score: float,
        risk_level: str,
        latency_ms: float,
        entity_count: int,
    ) -> None:
        """Log a single prediction to MLflow and check for drift."""
        # Always update local windows (drift detection works without MLflow)
        self._risk_window.append(risk_score)
        self._latency_window.append(latency_ms)
        self._check_drift()

        if not self._enabled:
            return

        try:
            with mlflow.start_run(
                run_name=f"pred_{note_id}_{datetime.now().strftime('%H%M%S')}"
            ):
                mlflow.log_metrics(
                    {
                        "risk_score": risk_score,
                        "latency_ms": latency_ms,
                        "entity_count": entity_count,
                    }
                )
                mlflow.log_params(
                    {
                        "note_id": note_id,
                        "risk_level": risk_level,
                    }
                )
        except Exception as e:
            logger.warning(f"MLflow log_prediction failed: {e}")

    def _check_drift(self) -> None:
        """Detect drift when window is full."""
        if len(self._risk_window) < self.WINDOW_SIZE:
            return

        avg_risk = statistics.mean(self._risk_window)
        avg_latency = statistics.mean(self._latency_window)

        alerts = []

        # Risk score drift
        drift_risk = abs(avg_risk - self.BASELINE_RISK_SCORE)
        if drift_risk > settings.drift_alert_threshold:
            msg = (
                f"DRIFT ALERT: Risk score drifted {drift_risk:.3f} from baseline "
                f"(avg={avg_risk:.3f}, baseline={self.BASELINE_RISK_SCORE})"
            )
            alerts.append(msg)
            logger.warning(msg)

        # Latency drift
        if avg_latency > self.BASELINE_LATENCY_MS * 1.5:
            msg = (
                f"LATENCY ALERT: avg latency {avg_latency:.0f}ms "
                f"exceeds 1.5× baseline ({self.BASELINE_LATENCY_MS:.0f}ms)"
            )
            alerts.append(msg)
            logger.warning(msg)

        self._alerts = alerts

        # Log drift metrics to MLflow
        if alerts and self._enabled:
            try:
                with mlflow.start_run(run_name="drift_check"):
                    mlflow.log_metrics(
                        {
                            "drift_risk_score": drift_risk,
                            "avg_latency_ms": avg_latency,
                            "drift_alerts": len(alerts),
                        }
                    )
            except Exception:
                pass

    def get_drift_alerts(self) -> list[str]:
        return self._alerts.copy()

    def get_latest_metrics(self) -> Optional[dict]:
        """Fetch the most recent MLflow run metrics."""
        if not self._enabled:
            return None
        try:
            exp = mlflow.get_experiment_by_name(self._experiment_name)
            if not exp:
                return None

            runs = self.client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=1,
            )

            if not runs:
                return None

            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "start_time": run.info.start_time,
            }
        except Exception as e:
            logger.warning(f"get_latest_metrics failed: {e}")
            return None

    def log_batch_summary(self, results: list[dict]) -> None:
        """Log aggregate metrics for a batch run."""
        if not self._enabled or not results:
            return
        try:
            risk_scores = [r["risk_score"] for r in results if r.get("risk_score")]
            latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]

            with mlflow.start_run(run_name=f"batch_{len(results)}"):
                if risk_scores:
                    mlflow.log_metrics(
                        {
                            "batch_avg_risk_score": statistics.mean(risk_scores),
                            "batch_max_risk_score": max(risk_scores),
                            "batch_count": len(results),
                        }
                    )
                if latencies:
                    mlflow.log_metrics(
                        {
                            "batch_avg_latency_ms": statistics.mean(latencies),
                            "batch_p95_latency_ms": sorted(latencies)[
                                int(len(latencies) * 0.95)
                            ],
                        }
                    )
        except Exception as e:
            logger.warning(f"log_batch_summary failed: {e}")


# Singleton
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker
