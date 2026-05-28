# Clinical NLP Triage and Risk Stratification Pipeline

An end-to-end clinical AI demo that turns unstructured clinical notes into a
structured triage response: extracted entities, risk score, risk level, care
pathway recommendation, and monitoring metrics.

> This project is for portfolio and interview demonstration use. It is not a
> validated clinical decision system.

## Problem

Clinical notes are long, unstructured, and hard to route quickly. A reviewer may
need to identify symptoms, medications, diagnoses, urgency, and next-step care
recommendations from free text.

## Solution

The project exposes a FastAPI service that runs a four-stage LangGraph workflow:

1. Extract clinical entities with spaCy/scispaCy plus an LLM.
2. Validate the extracted entities with lightweight safety rules.
3. Score patient risk using the configured LLM, with heuristic fallback.
4. Retrieve similar care pathways from ChromaDB and generate a recommendation.

## Architecture

```text
Raw clinical note
  -> Clinical NER
  -> LangGraph: extract -> validate -> score -> recommend
  -> ChromaDB care pathway retrieval
  -> FastAPI response
  -> MLflow prediction metrics
```

The repository also includes a PySpark Bronze/Silver/Gold ingestion pipeline for
batch data preparation. The current Gold layer is a schema-ready placeholder;
the live triage result is produced by the FastAPI/LangGraph path.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| API | FastAPI, Uvicorn, Pydantic |
| Agent workflow | LangGraph, LiteLLM |
| LLM providers | Groq primary, Gemini fallback, optional OpenAI/Mistral/Cohere/Anthropic |
| Clinical NLP | spaCy, scispaCy-compatible extraction, rule fallbacks |
| Retrieval | Keyword fallback by default, optional ChromaDB plus sentence-transformers |
| Monitoring | MLflow local tracking, rolling drift checks |
| Batch pipeline | PySpark, Delta Lake |
| Testing | pytest |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional, for better biomedical NER:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
```

Create `.env` from `.env.example` and set at least one provider key:

```bash
copy .env.example .env
```

Recommended demo configuration:

```env
LITELLM_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_optional_google_fallback_key_here
MLFLOW_TRACKING_URI=file:./mlruns
USE_VECTOR_RETRIEVAL=false
```

Set `USE_VECTOR_RETRIEVAL=true` only when the sentence-transformer model is
already available locally or the machine can download from Hugging Face.

## Run The API

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open the browser UI:

```text
http://localhost:8000/
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## Demo Request

```bash
curl -X POST "http://localhost:8000/triage" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"58-year-old male with substernal chest pain radiating to the left arm. EKG shows ST elevation. Takes aspirin and atorvastatin.\",\"specialty\":\"CARDIOLOGY\"}"
```

Expected output includes:

```json
{
  "risk_score": 0.8,
  "risk_level": "HIGH",
  "risk_reasoning": "Clinical reasoning sentence",
  "care_pathway": "Concise care recommendation",
  "entities": {
    "diagnoses": [],
    "medications": [],
    "procedures": [],
    "anatomy": [],
    "symptoms": []
  },
  "validation_notes": "PASS",
  "errors": []
}
```

## API Endpoints

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | GET | Service health check |
| `/triage` | POST | Triage one clinical note |
| `/batch` | POST | Triage up to 50 notes |
| `/metrics` | GET | Latest MLflow metrics, drift alerts, API call summary |

## Batch Data Pipeline

Generate sample MTSamples-style data:

```bash
python scripts/generate_sample_data.py
```

Run the local Spark/Delta pipeline:

```bash
python src/ingestion/spark_pipeline.py
```

This writes Bronze, Silver, and placeholder Gold tables under `data/delta`.

## Tests

```bash
pytest tests/ -v
```

## Interview Talking Points

- Converts unstructured clinical notes into structured triage outputs.
- Uses a resilient LLM provider chain: configured primary model, Groq, then Gemini when keys are available.
- Has deterministic fallback behavior for scoring if LLM calls fail.
- Uses ChromaDB retrieval to ground care pathway recommendations.
- Logs prediction metrics and drift alerts with MLflow.
- Includes API tests and an eval-gate script for basic quality checks.
