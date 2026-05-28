import json
import logging
import operator
import os
import re
import threading
from typing import Annotated, Optional, TypedDict

import litellm
from langgraph.graph import END, StateGraph

from src.config import settings
from src.nlp.clinical_ner import ClinicalEntities, get_ner
from src.nlp.rag_retriever import get_retriever

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
litellm.suppress_debug_info = True
LLM_TIMEOUT_SECONDS = 25

# ── API call counter ─────────────────────────────────────────────────────────
_api_call_count = 0
_api_call_log = []
_temporarily_failed_models: set[str] = set()


PROVIDER_ENV_KEYS = {
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

PROVIDER_SETTING_NAMES = {
    "google": "google_api_key",
    "gemini": "google_api_key",
    "groq": "groq_api_key",
    "openai": "openai_api_key",
    "mistral": "mistral_api_key",
    "cohere": "cohere_api_key",
    "anthropic": "anthropic_api_key",
}


def _provider_for_model(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[0].lower()
    if model.startswith("gpt-"):
        return "openai"
    return "openai"


def _api_key_for_provider(provider: str) -> str:
    setting_name = PROVIDER_SETTING_NAMES.get(provider, "openai_api_key")
    env_key = PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
    return (getattr(settings, setting_name, "") or os.environ.get(env_key, "")).strip()


def _model_has_key(model: str) -> bool:
    return bool(_api_key_for_provider(_provider_for_model(model)))


def get_api_call_summary() -> dict:
    return {
        "total_api_calls": _api_call_count,
        "calls_per_triage": 3,
        "log": _api_call_log[-20:],
    }


def _call_llm(prompt: str, model: str) -> str:
    """
    Unified LLM call.
    Gemini uses google-genai directly; other providers use LiteLLM.
    """
    provider = _provider_for_model(model)
    if provider in {"gemini", "google"}:
        return _call_gemini(prompt, model)

    env_key = PROVIDER_ENV_KEYS.get(provider, "OPENAI_API_KEY")
    api_key = _api_key_for_provider(provider)

    if api_key:
        os.environ[env_key] = api_key

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
        request_timeout=LLM_TIMEOUT_SECONDS,
        num_retries=0,
    )
    return response.choices[0].message.content.strip()


def _call_gemini(prompt: str, model: str) -> str:
    """Call Gemini directly through google-genai for a simpler demo path."""
    api_key = _api_key_for_provider("gemini")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    model_name = model.split("/", 1)[1] if "/" in model else model

    from google import genai

    client = genai.Client(api_key=api_key)
    result: dict[str, object] = {}

    def worker() -> None:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            result["text"] = (response.text or "").strip()
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(LLM_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise TimeoutError(f"Gemini call timed out after {LLM_TIMEOUT_SECONDS}s")
    if "error" in result:
        raise result["error"]  # type: ignore[misc]

    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _llm_call(prompt: str, node: str) -> str:
    """
    Try the configured model, then Groq and Gemini fallbacks when keys exist.
    The caller handles final non-LLM fallback behavior.
    """
    global _api_call_count, _api_call_log, _temporarily_failed_models

    preferred_models = [
        settings.litellm_model.strip(),
        "groq/llama-3.1-8b-instant",
        "gemini/gemini-2.5-flash",
    ]
    models = list(dict.fromkeys(m for m in preferred_models if m))
    last_error: Exception | None = None

    for idx, model in enumerate(models):
        if model in _temporarily_failed_models:
            logger.info(
                f"[LLM] node={node} | model={model} skipped: earlier call failed"
            )
            continue

        if not _model_has_key(model):
            logger.info(
                f"[LLM] node={node} | model={model} skipped: API key not configured"
            )
            continue

        _api_call_count += 1
        call_num = _api_call_count
        label = "primary" if idx == 0 else "fallback"
        logger.info(f"[API Call #{call_num}] node={node} | model={model} ({label})")

        try:
            text = _call_llm(prompt, model)
            _api_call_log.append(
                {"call": call_num, "node": node, "model": model, "status": "OK"}
            )
            logger.info(f"[API Call #{call_num}] OK ({len(text)} chars)")
            return text
        except Exception as err:
            last_error = err
            _temporarily_failed_models.add(model)
            _api_call_log.append(
                {
                    "call": call_num,
                    "node": node,
                    "model": model,
                    "status": f"FAIL: {type(err).__name__}",
                }
            )
            logger.warning(f"[API Call #{call_num}] failed: {err}")

    if last_error:
        raise last_error
    raise RuntimeError("No LLM API key configured for the primary or fallback models.")


# --------------------------------------------------------------------------- #
#  Agent State                                                                  #
# --------------------------------------------------------------------------- #
class TriageState(TypedDict):
    note_id: str
    raw_text: str
    specialty: str
    entities: Optional[dict]
    validation_notes: Optional[str]
    risk_score: Optional[float]
    risk_level: Optional[str]
    risk_reasoning: Optional[str]
    care_pathway: Optional[str]
    retrieved_pathways: Optional[list]
    errors: Annotated[list[str], operator.add]
    retry_count: int


# --------------------------------------------------------------------------- #
#  NODE 1 — Extract                                                             #
# --------------------------------------------------------------------------- #
def node_extract(state: TriageState) -> TriageState:
    logger.info(f"[Extract] note_id={state['note_id']}")
    try:
        ner = get_ner()
        spacy_entities: ClinicalEntities = ner.extract(state["raw_text"])

        prompt = f"""You are a clinical NLP expert. Extract medical entities from this note.
Return ONLY valid JSON with keys: diagnoses, medications, procedures, anatomy, symptoms.
Each value is a list of strings. No explanation, no markdown, just JSON.

Clinical Note:
{state['raw_text'][:2000]}

JSON:"""

        try:
            raw = _llm_call(prompt, node="extract")
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
            llm_entities = json.loads(raw)
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}. Using spaCy only.")
            llm_entities = {}

        def merge(a, b):
            seen, out = set(), []
            for item in a + (b or []):
                k = str(item).lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    out.append(item)
            return out

        merged = {
            "diagnoses": merge(
                spacy_entities.diagnoses, llm_entities.get("diagnoses", [])
            ),
            "medications": merge(
                spacy_entities.medications, llm_entities.get("medications", [])
            ),
            "procedures": merge(
                spacy_entities.procedures, llm_entities.get("procedures", [])
            ),
            "anatomy": merge(spacy_entities.anatomy, llm_entities.get("anatomy", [])),
            "symptoms": merge(
                spacy_entities.symptoms, llm_entities.get("symptoms", [])
            ),
        }

        return {**state, "entities": merged}

    except Exception as e:
        logger.error(f"[Extract] Error: {e}")
        return {**state, "entities": {}, "errors": [str(e)]}


# --------------------------------------------------------------------------- #
#  NODE 2 — Validate (no LLM call — pure rule-based)                           #
# --------------------------------------------------------------------------- #
def node_validate(state: TriageState) -> TriageState:
    logger.info(f"[Validate] note_id={state['note_id']}")
    entities = state.get("entities", {})
    notes = []

    if not entities.get("diagnoses") and not entities.get("symptoms"):
        notes.append("WARNING: No diagnoses or symptoms found.")

    meds_lower = {m.lower() for m in entities.get("medications", [])}
    if "warfarin" in meds_lower and meds_lower & {"aspirin", "ibuprofen", "naproxen"}:
        notes.append("DRUG INTERACTION FLAG: Warfarin + NSAID/aspirin — bleeding risk.")

    if sum(len(v) for v in entities.values()) < 2:
        notes.append("LOW ENTITY COUNT: Fewer than 2 entities found.")

    return {**state, "validation_notes": "; ".join(notes) if notes else "PASS"}


# --------------------------------------------------------------------------- #
#  NODE 3 — Score                                                               #
# --------------------------------------------------------------------------- #
def node_score(state: TriageState) -> TriageState:
    logger.info(f"[Score] note_id={state['note_id']}")
    entities = state.get("entities", {})
    validation = state.get("validation_notes", "")

    prompt = f"""You are a clinical risk stratification expert.
Assign a risk score to this patient based on extracted entities.

ENTITIES:
{json.dumps(entities, indent=2)}

VALIDATION NOTES: {validation}
SPECIALTY: {state.get('specialty', 'GENERAL')}

Scoring rules:
- 0.0 to 1.0 scale (0=minimal risk, 1=immediately life-threatening)
- LOW < 0.4 | MEDIUM 0.4-0.7 | HIGH 0.7-0.9 | CRITICAL > 0.9
- High weight: chest pain, stroke, cancer, sepsis, overdose, respiratory failure
- Return ONLY valid JSON, no explanation:

{{"risk_score": 0.XX, "risk_level": "HIGH", "reasoning": "One sentence."}}"""

    try:
        raw = _llm_call(prompt, node="score")
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("```").strip()
        result = json.loads(raw)
        risk_score = max(0.0, min(1.0, float(result.get("risk_score", 0.3))))
        risk_level = result.get("risk_level", "LOW").upper()
        reasoning = result.get("reasoning", "")
    except Exception as e:
        logger.warning(f"[Score] Both LLMs failed: {e}. Using heuristic.")
        risk_score, risk_level, reasoning = _heuristic_score(entities)

    return {
        **state,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_reasoning": reasoning,
    }


def _heuristic_score(entities: dict) -> tuple[float, str, str]:
    HIGH = {
        "chest pain",
        "stroke",
        "sepsis",
        "cancer",
        "respiratory failure",
        "overdose",
        "myocardial infarction",
        "cardiac arrest",
    }
    MED = {
        "pneumonia",
        "diabetes",
        "fracture",
        "infection",
        "hypertension",
        "abdominal pain",
        "heart failure",
        "sleep apnea",
        "obesity",
    }
    all_terms = {t.lower() for v in entities.values() for t in v}
    if all_terms & HIGH:
        return 0.82, "HIGH", "High-risk condition detected (heuristic)."
    if all_terms & MED:
        return 0.55, "MEDIUM", "Moderate-risk condition detected (heuristic)."
    return 0.25, "LOW", "No high-risk conditions detected (heuristic)."


# --------------------------------------------------------------------------- #
#  NODE 4 — Recommend                                                           #
# --------------------------------------------------------------------------- #
def node_recommend(state: TriageState) -> TriageState:
    logger.info(f"[Recommend] note_id={state['note_id']}")
    entities = state.get("entities", {})
    risk_level = state.get("risk_level", "LOW")
    specialty = state.get("specialty", "GENERAL")

    query = (
        " ".join(
            entities.get("diagnoses", [])[:3]
            + entities.get("symptoms", [])[:2]
            + [specialty]
        )
        or state["raw_text"][:300]
    )

    retriever = get_retriever()
    retrieved = retriever.retrieve(query=query, n_results=3)
    context = "\n".join(
        [
            f"- [{r.get('risk_level','?')}] {r.get('condition','?')}: {r.get('pathway','')}"
            for r in retrieved
        ]
    )

    prompt = f"""You are a clinical care pathway expert.

PATIENT SUMMARY:
- Risk Level: {risk_level}
- Risk Score: {state.get('risk_score', 0):.2f}
- Specialty: {specialty}
- Diagnoses:   {', '.join(entities.get('diagnoses',   [])[:5]) or 'None'}
- Medications: {', '.join(entities.get('medications', [])[:5]) or 'None'}
- Symptoms:    {', '.join(entities.get('symptoms',    [])[:5]) or 'None'}
- Procedures:  {', '.join(entities.get('procedures',  [])[:3]) or 'None'}

SIMILAR CASE PATHWAYS:
{context or 'None found.'}

Write a concise care pathway recommendation (3-5 bullet points).
Be specific. Do not invent dosages. Plain text only, no JSON."""

    try:
        care_pathway = _llm_call(prompt, node="recommend")
    except Exception as e:
        logger.warning(
            f"[Recommend] Both LLMs failed: {e}. Using top retrieved pathway."
        )
        care_pathway = (
            retrieved[0]["pathway"] if retrieved else "Consult primary care physician."
        )

    return {**state, "care_pathway": care_pathway, "retrieved_pathways": retrieved}


# --------------------------------------------------------------------------- #
#  Build LangGraph                                                               #
# --------------------------------------------------------------------------- #
def build_triage_graph():
    graph = StateGraph(TriageState)
    graph.add_node("extract", node_extract)
    graph.add_node("validate", node_validate)
    graph.add_node("score", node_score)
    graph.add_node("recommend", node_recommend)
    graph.set_entry_point("extract")
    graph.add_edge("extract", "validate")
    graph.add_edge("validate", "score")
    graph.add_edge("score", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


_graph = None


def get_triage_graph():
    global _graph
    if _graph is None:
        _graph = build_triage_graph()
    return _graph


def run_triage(note_id: str, raw_text: str, specialty: str = "GENERAL") -> dict:
    graph = get_triage_graph()
    return graph.invoke(
        {
            "note_id": note_id,
            "raw_text": raw_text,
            "specialty": specialty,
            "entities": None,
            "validation_notes": None,
            "risk_score": None,
            "risk_level": None,
            "risk_reasoning": None,
            "care_pathway": None,
            "retrieved_pathways": None,
            "errors": [],
            "retry_count": 0,
        }
    )
