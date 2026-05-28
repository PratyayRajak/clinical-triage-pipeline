"""
FastAPI REST API
Endpoints: /triage, /batch, /health, /metrics
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.agents.triage_graph import run_triage
from src.config import settings
from src.monitoring.mlflow_tracker import get_tracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------- #
#  Pydantic Models                                                              #
# --------------------------------------------------------------------------- #
class TriageRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Clinical note text")
    specialty: str = Field(default="GENERAL", description="Medical specialty")
    note_id: Optional[str] = Field(default=None, description="Optional note ID")


class BatchTriageRequest(BaseModel):
    notes: list[TriageRequest] = Field(..., max_length=50)


class TriageResponse(BaseModel):
    note_id: str
    risk_score: float
    risk_level: str
    risk_reasoning: str
    care_pathway: str
    entities: dict
    validation_notes: str
    latency_ms: float
    errors: list[str]


class BatchTriageResponse(BaseModel):
    results: list[TriageResponse]
    total_notes: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str


class MetricsResponse(BaseModel):
    experiment_name: str
    latest_run: Optional[dict]
    drift_alerts: list[str]
    api_calls: Optional[dict]


# --------------------------------------------------------------------------- #
#  App                                                                          #
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Clinical Triage API starting up...")
    if settings.warmup_on_startup:
        try:
            from src.agents.triage_graph import get_triage_graph
            from src.nlp.clinical_ner import get_ner
            from src.nlp.rag_retriever import get_retriever

            get_ner()
            get_triage_graph()
            get_retriever()
            logger.info("Models warmed up successfully.")
        except Exception as e:
            logger.warning(f"Warm-up failed (non-fatal): {e}")
    else:
        logger.info("Model warm-up skipped. Set WARMUP_ON_STARTUP=true to enable it.")
    yield
    logger.info("Clinical Triage API shutting down.")


app = FastAPI(
    title="Clinical NLP Triage API",
    description="GenAI-powered clinical note triage and risk stratification",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --------------------------------------------------------------------------- #
#  Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
async def triage_note(
    request: TriageRequest,
    background_tasks: BackgroundTasks,
):
    """
    Run the full 4-node LangGraph triage pipeline on a single clinical note.
    Returns risk score, level, care pathway, and extracted entities.
    """
    note_id = request.note_id or str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    try:
        result = run_triage(
            note_id=note_id,
            raw_text=request.text,
            specialty=request.specialty.upper(),
        )
    except Exception as e:
        logger.error(f"Triage failed for note_id={note_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    response = TriageResponse(
        note_id=note_id,
        risk_score=round(result.get("risk_score") or 0.0, 3),
        risk_level=result.get("risk_level") or "UNKNOWN",
        risk_reasoning=result.get("risk_reasoning") or "",
        care_pathway=result.get("care_pathway") or "",
        entities=result.get("entities") or {},
        validation_notes=result.get("validation_notes") or "",
        latency_ms=round(latency_ms, 1),
        errors=result.get("errors") or [],
    )

    # Log to MLflow in background (non-blocking)
    background_tasks.add_task(
        _log_to_mlflow,
        note_id=note_id,
        risk_score=response.risk_score,
        risk_level=response.risk_level,
        latency_ms=latency_ms,
        entity_count=sum(len(v) for v in response.entities.values()),
    )

    return response


@app.post("/batch", response_model=BatchTriageResponse, tags=["Triage"])
async def batch_triage(request: BatchTriageRequest, background_tasks: BackgroundTasks):
    """
    Triage multiple clinical notes in a single request (max 50).
    """
    t0 = time.perf_counter()
    results = []

    for note_req in request.notes:
        single_resp = await triage_note(note_req, background_tasks)
        results.append(single_resp)

    total_latency = (time.perf_counter() - t0) * 1000

    return BatchTriageResponse(
        results=results,
        total_notes=len(results),
        total_latency_ms=round(total_latency, 1),
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Return latest MLflow metrics, drift alerts, and API call count."""
    from src.agents.triage_graph import get_api_call_summary

    try:
        tracker = get_tracker()
        latest = tracker.get_latest_metrics()
        alerts = tracker.get_drift_alerts()
        return MetricsResponse(
            experiment_name=tracker.experiment_name,
            latest_run=latest,
            drift_alerts=alerts,
            api_calls=get_api_call_summary(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------- #
#  Background task                                                              #
# --------------------------------------------------------------------------- #
def _log_to_mlflow(
    note_id: str,
    risk_score: float,
    risk_level: str,
    latency_ms: float,
    entity_count: int,
) -> None:
    try:
        tracker = get_tracker()
        tracker.log_prediction(
            note_id=note_id,
            risk_score=risk_score,
            risk_level=risk_level,
            latency_ms=latency_ms,
            entity_count=entity_count,
        )
    except Exception as e:
        logger.warning(f"MLflow logging failed (non-fatal): {e}")
