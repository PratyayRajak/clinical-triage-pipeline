"""
Test Suite for Clinical Triage Pipeline

Tests:
- NER extraction
- RAG retriever
- Risk scoring heuristics
- FastAPI endpoints (mocked)
- MLflow tracker
"""

import json
from unittest.mock import MagicMock, patch

import pytest


# ============================================================
#  NER Tests
# ============================================================
class TestClinicalNER:
    """Tests for the ClinicalNER extractor."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.nlp.clinical_ner import ClinicalNER

        # Patch spaCy load to avoid model download in CI
        with patch("src.nlp.clinical_ner._load_nlp") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []
            mock_doc.__iter__ = MagicMock(return_value=iter([]))
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp
            self.ner = ClinicalNER()
            self.ner.nlp = mock_nlp

    def test_empty_text_returns_empty(self):
        from src.nlp.clinical_ner import ClinicalEntities

        result = self.ner.extract("")
        assert isinstance(result, ClinicalEntities)
        assert result.entity_count() == 0

    def test_medication_keyword_detected(self):
        """Aspirin should be detected via keyword fallback."""
        mock_doc = MagicMock()
        mock_doc.ents = []

        # Create mock tokens
        aspirin_token = MagicMock()
        aspirin_token.text = "aspirin"

        other_token = MagicMock()
        other_token.text = "patient"

        mock_doc.__iter__ = MagicMock(return_value=iter([aspirin_token, other_token]))
        self.ner.nlp = MagicMock(return_value=mock_doc)

        result = self.ner.extract("Patient takes aspirin daily.")
        assert "aspirin" in [m.lower() for m in result.medications]

    def test_procedure_regex_detected(self):
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.__iter__ = MagicMock(return_value=iter([]))
        self.ner.nlp = MagicMock(return_value=mock_doc)

        result = self.ner.extract("Patient underwent MRI scan and biopsy.")
        procs = [p.lower() for p in result.procedures]
        assert any("mri" in p for p in procs) or any("biopsy" in p for p in procs)

    def test_dedup_works(self):
        from src.nlp.clinical_ner import _dedup

        items = ["Aspirin", "aspirin", "ASPIRIN", "ibuprofen"]
        result = _dedup(items)
        assert len(result) == 2


# ============================================================
#  RAG Retriever Tests
# ============================================================
class TestCarePathwayRetriever:
    """Tests for ChromaDB retriever."""

    @pytest.fixture
    def retriever(self, tmp_path):
        with patch("chromadb.PersistentClient") as mock_client:
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )
            mock_collection.query.return_value = {
                "metadatas": [
                    [
                        {
                            "condition": "chest pain",
                            "specialty": "CARDIOLOGY",
                            "risk_level": "HIGH",
                            "pathway": "Immediate ECG + troponin.",
                            "tags": "chest pain cardiac",
                        }
                    ]
                ],
                "distances": [[0.15]],
            }

            from src.nlp.rag_retriever import CarePathwayRetriever

            r = CarePathwayRetriever.__new__(CarePathwayRetriever)
            r.collection = mock_collection
            r.persist_dir = str(tmp_path)
            return r

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("chest pain cardiac", n_results=3)
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_relevance_score_between_0_and_1(self, retriever):
        results = retriever.retrieve("pneumonia fever cough")
        for r in results:
            assert 0.0 <= r["relevance_score"] <= 1.0


# ============================================================
#  Heuristic Scorer Tests
# ============================================================
class TestHeuristicScorer:

    def test_high_risk_entities(self):
        from src.agents.triage_graph import _heuristic_score

        entities = {
            "diagnoses": ["chest pain", "myocardial infarction"],
            "symptoms": [],
            "medications": [],
            "procedures": [],
            "anatomy": [],
        }
        score, level, _ = _heuristic_score(entities)
        assert score >= 0.7
        assert level == "HIGH"

    def test_medium_risk_entities(self):
        from src.agents.triage_graph import _heuristic_score

        entities = {
            "diagnoses": ["pneumonia"],
            "symptoms": ["fever", "cough"],
            "medications": ["azithromycin"],
            "procedures": [],
            "anatomy": [],
        }
        score, level, _ = _heuristic_score(entities)
        assert 0.4 <= score <= 0.75
        assert level == "MEDIUM"

    def test_low_risk_entities(self):
        from src.agents.triage_graph import _heuristic_score

        entities = {
            "diagnoses": [],
            "symptoms": ["mild headache"],
            "medications": [],
            "procedures": [],
            "anatomy": [],
        }
        score, level, _ = _heuristic_score(entities)
        assert score < 0.4
        assert level == "LOW"


# ============================================================
#  FastAPI Tests
# ============================================================
class TestAPI:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        # Patch run_triage to avoid real LLM calls
        with patch("src.api.main.run_triage") as mock_triage:
            mock_triage.return_value = {
                "note_id": "test_001",
                "risk_score": 0.82,
                "risk_level": "HIGH",
                "risk_reasoning": "STEMI indicators present.",
                "care_pathway": "Immediate cath lab activation.",
                "entities": {
                    "diagnoses": ["chest pain"],
                    "medications": ["aspirin"],
                    "procedures": [],
                    "anatomy": [],
                    "symptoms": ["dyspnea"],
                },
                "validation_notes": "PASS",
                "errors": [],
            }
            from src.api.main import app

            with TestClient(app) as c:
                yield c

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_triage_endpoint_success(self, client):
        payload = {
            "text": "58 year old male with substernal chest pain and diaphoresis. EKG shows STEMI.",
            "specialty": "CARDIOLOGY",
        }
        resp = client.post("/triage", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "risk_level" in data
        assert "care_pathway" in data

    def test_triage_rejects_short_text(self, client):
        resp = client.post("/triage", json={"text": "short"})
        assert resp.status_code == 422  # Pydantic min_length

    def test_batch_endpoint(self, client):
        payload = {
            "notes": [
                {
                    "text": "Patient presents with fever cough and shortness of breath.",
                    "specialty": "PULMONOLOGY",
                },
                {
                    "text": "Routine annual physical exam, no acute complaints, vitals stable.",
                    "specialty": "GENERAL",
                },
            ]
        }
        resp = client.post("/batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_notes"] == 2
        assert len(data["results"]) == 2


# ============================================================
#  MLflow Tracker Tests
# ============================================================
class TestMLflowTracker:

    @pytest.fixture
    def tracker(self):
        with patch("mlflow.set_tracking_uri"), patch("mlflow.set_experiment"), patch(
            "mlflow.get_experiment_by_name"
        ) as mock_get_exp, patch("mlflow.start_run"), patch(
            "mlflow.log_metrics"
        ), patch(
            "mlflow.log_params"
        ):
            mock_get_exp.return_value = MagicMock(experiment_id="1")
            from src.monitoring.mlflow_tracker import MLflowTracker

            return MLflowTracker()

    def test_log_prediction_updates_window(self, tracker):
        with patch.object(tracker, "_check_drift"):
            tracker.log_prediction("n1", 0.8, "HIGH", 1200.0, 5)
            assert len(tracker._risk_window) == 1
            assert tracker._risk_window[0] == 0.8

    def test_drift_detected_when_score_diverges(self, tracker):
        """Fill the window with near-zero scores — should trigger drift."""
        with patch("mlflow.start_run"), patch("mlflow.log_metrics"):
            for _ in range(5):
                tracker._risk_window.append(0.01)
                tracker._latency_window.append(1000.0)
            tracker._check_drift()
            assert len(tracker._alerts) > 0

    def test_no_drift_within_threshold(self, tracker):
        with patch("mlflow.start_run"), patch("mlflow.log_metrics"):
            for _ in range(5):
                tracker._risk_window.append(0.5)  # exactly baseline
                tracker._latency_window.append(2000.0)
            tracker._check_drift()
            # Risk drift should be 0 — no alerts from risk
            risk_alerts = [a for a in tracker._alerts if "Risk score drifted" in a]
            assert len(risk_alerts) == 0
