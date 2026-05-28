"""
Eval Gate Script
Runs triage on a fixed evaluation set and asserts minimum quality thresholds.
Blocks CI/CD if thresholds are not met.

Thresholds (mirrors Legal RAG pipeline approach):
  - avg risk score must be between 0.2 and 0.9 (sanity check — not all notes are high risk)
  - entity extraction must find at least 1 entity for 80% of notes
  - no more than 10% of notes should error out
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_gate")

# Fixed eval set (deterministic, no LLM needed for basic checks)
EVAL_NOTES = [
    {
        "note_id": "eval_001",
        "text": "Patient presents with substernal chest pain radiating to the left arm. History of hypertension. Takes aspirin and atorvastatin. EKG shows ST changes.",
        "specialty": "CARDIOLOGY",
        "expected_min_risk": 0.5,
    },
    {
        "note_id": "eval_002",
        "text": "52-year-old female with type 2 diabetes. HbA1c 8.2%. Current medications include metformin 1000mg twice daily. Follow-up for glucose management.",
        "specialty": "ENDOCRINOLOGY",
        "expected_min_risk": 0.3,
    },
    {
        "note_id": "eval_003",
        "text": "Annual wellness visit. No acute complaints. BP 120/78. BMI 23. Routine labs ordered. Flu vaccine given.",
        "specialty": "GENERAL",
        "expected_min_risk": 0.0,   # Low risk expected
    },
    {
        "note_id": "eval_004",
        "text": "Sudden onset left-sided weakness and facial droop. Last known well 1 hour ago. CT head: no hemorrhage. Right MCA territory involved.",
        "specialty": "NEUROLOGY",
        "expected_min_risk": 0.6,
    },
    {
        "note_id": "eval_005",
        "text": "67-year-old with productive cough, fever 38.9°C, and O2 sat 91%. CXR shows right lower lobe consolidation. WBC 14,200.",
        "specialty": "PULMONOLOGY",
        "expected_min_risk": 0.4,
    },
]

THRESHOLDS = {
    "min_entity_extraction_rate": 0.80,  # 80% of notes must have ≥1 entity
    "max_error_rate":             0.10,  # <10% errors
    "min_avg_risk_score":         0.20,  # Pipeline shouldn't default everything to zero
    "max_avg_risk_score":         0.95,  # Shouldn't flag everything as critical
}


def run_eval_gate() -> bool:
    """Run the eval gate. Returns True if all thresholds pass."""
    from src.nlp.clinical_ner import get_ner
    from src.agents.triage_graph import _heuristic_score  # Use heuristic for CI (no API key needed)

    logger.info("=" * 60)
    logger.info("EVAL GATE: Clinical Triage Pipeline")
    logger.info("=" * 60)

    ner = get_ner()
    results = []

    for note in EVAL_NOTES:
        try:
            entities = ner.extract(note["text"])
            entity_dict = entities.to_dict()
            entity_count = entities.entity_count()

            # Use heuristic scorer (no LLM needed for CI gate)
            risk_score, risk_level, reasoning = _heuristic_score(entity_dict)

            results.append({
                "note_id":      note["note_id"],
                "entity_count": entity_count,
                "risk_score":   risk_score,
                "risk_level":   risk_level,
                "error":        None,
            })
            logger.info(f"  [{note['note_id']}] entities={entity_count}, risk={risk_score:.2f} ({risk_level})")

        except Exception as e:
            logger.error(f"  [{note['note_id']}] ERROR: {e}")
            results.append({"note_id": note["note_id"], "error": str(e), "entity_count": 0, "risk_score": 0.0})

    # ── Compute metrics ─────────────────────────────────────────────────── #
    total      = len(results)
    errors     = sum(1 for r in results if r.get("error"))
    with_entities = sum(1 for r in results if r.get("entity_count", 0) >= 1)
    risk_scores = [r["risk_score"] for r in results if not r.get("error")]

    entity_rate = with_entities / total
    error_rate  = errors / total
    avg_risk    = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0

    logger.info("\n--- EVAL RESULTS ---")
    logger.info(f"  Total notes evaluated : {total}")
    logger.info(f"  Entity extraction rate: {entity_rate:.2%} (threshold: ≥{THRESHOLDS['min_entity_extraction_rate']:.0%})")
    logger.info(f"  Error rate            : {error_rate:.2%} (threshold: ≤{THRESHOLDS['max_error_rate']:.0%})")
    logger.info(f"  Avg risk score        : {avg_risk:.3f} (threshold: {THRESHOLDS['min_avg_risk_score']}–{THRESHOLDS['max_avg_risk_score']})")

    # ── Gate checks ─────────────────────────────────────────────────────── #
    failures = []

    if entity_rate < THRESHOLDS["min_entity_extraction_rate"]:
        failures.append(f"FAIL: entity extraction rate {entity_rate:.2%} < {THRESHOLDS['min_entity_extraction_rate']:.0%}")

    if error_rate > THRESHOLDS["max_error_rate"]:
        failures.append(f"FAIL: error rate {error_rate:.2%} > {THRESHOLDS['max_error_rate']:.0%}")

    if avg_risk < THRESHOLDS["min_avg_risk_score"]:
        failures.append(f"FAIL: avg risk score {avg_risk:.3f} < {THRESHOLDS['min_avg_risk_score']}")

    if avg_risk > THRESHOLDS["max_avg_risk_score"]:
        failures.append(f"FAIL: avg risk score {avg_risk:.3f} > {THRESHOLDS['max_avg_risk_score']}")

    if failures:
        logger.error("\n[EVAL GATE FAILED]")
        for f in failures:
            logger.error(f"  {f}")
        return False
    else:
        logger.info("\n[EVAL GATE PASSED] All thresholds met.")
        return True


if __name__ == "__main__":
    passed = run_eval_gate()
    sys.exit(0 if passed else 1)
