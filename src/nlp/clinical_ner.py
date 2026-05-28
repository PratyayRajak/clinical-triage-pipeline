"""
Clinical NER Module
Uses spaCy + scispaCy (en_core_sci_sm) for medical entity extraction.
Falls back to en_core_web_sm if scispaCy model not installed.

Extracts:
  - DISEASE / CONDITION
  - MEDICATION / DRUG
  - PROCEDURE
  - ANATOMY
  - SYMPTOM (heuristic)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-loaded spaCy model
_nlp = None


def _load_nlp():
    """Load spaCy model — prefer scispaCy, fall back to en_core_web_sm."""
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        import spacy

        try:
            _nlp = spacy.load("en_core_sci_sm")
            logger.info("Loaded scispaCy model: en_core_sci_sm")
        except OSError:
            logger.warning("en_core_sci_sm not found — falling back to en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    except ImportError:
        raise ImportError("spaCy not installed. Run: pip install spacy")

    return _nlp


# ---- Medication keyword list (subset of RxNorm common drugs) ---------------
MEDICATION_KEYWORDS = {
    "aspirin",
    "ibuprofen",
    "acetaminophen",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "amoxicillin",
    "metoprolol",
    "omeprazole",
    "losartan",
    "albuterol",
    "gabapentin",
    "sertraline",
    "amlodipine",
    "levothyroxine",
    "hydrochlorothiazide",
    "prednisone",
    "tramadol",
    "furosemide",
    "clopidogrel",
    "warfarin",
    "insulin",
    "morphine",
    "codeine",
    "oxycodone",
    "hydrocodone",
    "ciprofloxacin",
    "azithromycin",
    "doxycycline",
    "metronidazole",
    "diovan",
    "valsartan",
    "crestor",
    "rosuvastatin",
    "tricor",
    "fenofibrate",
}

# ---- Procedure keyword patterns --------------------------------------------
PROCEDURE_PATTERNS = [
    r"\b(surgery|operation|resection|biopsy|endoscopy|colonoscopy|angioplasty)\b",
    r"\b(mri|ct scan|x-ray|ultrasound|echocardiogram|ekg|ecg)\b",
    r"\b(catheterization|intubation|dialysis|chemotherapy|radiation)\b",
    r"\b(appendectomy|cholecystectomy|hysterectomy|mastectomy|bypass)\b",
]
PROCEDURE_RE = re.compile("|".join(PROCEDURE_PATTERNS), re.IGNORECASE)

# ---- Symptom keyword patterns -----------------------------------------------
SYMPTOM_PATTERNS = [
    r"\b(pain|ache|fever|nausea|vomiting|fatigue|dizziness|shortness of breath)\b",
    r"\b(headache|chest pain|back pain|abdominal pain|cough|dyspnea|edema)\b",
    r"\b(weakness|numbness|tingling|bleeding|swelling|rash|itching|insomnia)\b",
]
SYMPTOM_RE = re.compile("|".join(SYMPTOM_PATTERNS), re.IGNORECASE)


@dataclass
class ClinicalEntities:
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    anatomy: list[str] = field(default_factory=list)
    symptoms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "diagnoses": self.diagnoses,
            "medications": self.medications,
            "procedures": self.procedures,
            "anatomy": self.anatomy,
            "symptoms": self.symptoms,
        }

    def entity_count(self) -> int:
        return (
            len(self.diagnoses)
            + len(self.medications)
            + len(self.procedures)
            + len(self.anatomy)
            + len(self.symptoms)
        )


class ClinicalNER:
    """Extract structured medical entities from free-text clinical notes."""

    def __init__(self):
        self.nlp = _load_nlp()

    def extract(self, text: str) -> ClinicalEntities:
        """Run full NER pipeline on a clinical note."""
        if not text or not text.strip():
            return ClinicalEntities()

        # Truncate to 5000 chars for spaCy performance
        text = text[:5000]
        doc = self.nlp(text)

        entities = ClinicalEntities()

        # ── spaCy NER labels ──────────────────────────────────────────────── #
        for ent in doc.ents:
            label = ent.label_.upper()
            norm = ent.text.strip().lower()

            if label in ("DISEASE", "CONDITION", "DISORDER", "SYMPTOM_SIGN"):
                entities.diagnoses.append(ent.text.strip())

            elif label in ("CHEMICAL", "DRUG", "MEDICATION"):
                entities.medications.append(ent.text.strip())

            elif label in ("PROCEDURE", "TREATMENT"):
                entities.procedures.append(ent.text.strip())

            elif label in ("ANATOMY", "BODY_PART", "ORG"):
                entities.anatomy.append(ent.text.strip())

            # Generic ENTITY label from en_core_sci_sm
            elif label == "ENTITY":
                # Classify by keyword matching
                if norm in MEDICATION_KEYWORDS:
                    entities.medications.append(ent.text.strip())
                elif PROCEDURE_RE.search(norm):
                    entities.procedures.append(ent.text.strip())
                else:
                    entities.diagnoses.append(ent.text.strip())

        # ── Keyword / regex fallbacks ─────────────────────────────────────── #
        # Medications
        for token in doc:
            if token.text.lower() in MEDICATION_KEYWORDS:
                entities.medications.append(token.text)

        # Procedures
        for match in PROCEDURE_RE.finditer(text):
            entities.procedures.append(match.group().strip())

        # Symptoms
        for match in SYMPTOM_RE.finditer(text):
            entities.symptoms.append(match.group().strip())

        # ── Deduplicate (case-insensitive) ────────────────────────────────── #
        entities.diagnoses = _dedup(entities.diagnoses)
        entities.medications = _dedup(entities.medications)
        entities.procedures = _dedup(entities.procedures)
        entities.anatomy = _dedup(entities.anatomy)
        entities.symptoms = _dedup(entities.symptoms)

        return entities


def _dedup(items: list[str]) -> list[str]:
    seen, result = set(), []
    for item in items:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


# Singleton
_ner_instance: Optional[ClinicalNER] = None


def get_ner() -> ClinicalNER:
    global _ner_instance
    if _ner_instance is None:
        _ner_instance = ClinicalNER()
    return _ner_instance
