"""
ChromaDB RAG Retriever
Stores care pathway examples and similar case summaries.
Used by the Recommend node in the LangGraph pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import settings

logger = logging.getLogger(__name__)

# Care pathway knowledge base (seed data)
CARE_PATHWAYS = [
    {
        "id": "cp_001",
        "specialty": "CARDIOLOGY",
        "condition": "chest pain",
        "risk_level": "HIGH",
        "pathway": "Immediate ECG + troponin labs. Cardiology consult within 1hr. Consider cath lab activation if STEMI. Aspirin 325mg, heparin per protocol.",
        "tags": "chest pain, cardiac, STEMI, ACS, troponin",
    },
    {
        "id": "cp_002",
        "specialty": "CARDIOLOGY",
        "condition": "heart failure",
        "risk_level": "HIGH",
        "pathway": "IV furosemide diuresis. O2 supplementation. BNP + echo. Fluid restriction 1.5L/day. Cardiology follow-up within 7 days post-discharge.",
        "tags": "heart failure, CHF, edema, dyspnea, BNP, furosemide",
    },
    {
        "id": "cp_003",
        "specialty": "NEUROLOGY",
        "condition": "stroke",
        "risk_level": "HIGH",
        "pathway": "Activate stroke code. CT head stat. tPA eligibility assessment within 60 min. Neurology consult. BP management per protocol.",
        "tags": "stroke, CVA, tPA, CT, neurological deficit, aphasia",
    },
    {
        "id": "cp_004",
        "specialty": "ORTHOPEDIC",
        "condition": "fracture",
        "risk_level": "MEDIUM",
        "pathway": "X-ray confirmation. Ortho consult. Pain management with NSAIDs/opioids per severity. Immobilization. DVT prophylaxis if lower extremity.",
        "tags": "fracture, orthopedic, pain, immobilization, DVT",
    },
    {
        "id": "cp_005",
        "specialty": "PULMONOLOGY",
        "condition": "pneumonia",
        "risk_level": "MEDIUM",
        "pathway": "Chest X-ray. CBC + blood cultures before antibiotics. Start empiric antibiotics per CAP/HAP protocol. O2 therapy. Consider ICU for PORT score IV-V.",
        "tags": "pneumonia, respiratory, antibiotics, fever, cough, infiltrate",
    },
    {
        "id": "cp_006",
        "specialty": "ENDOCRINOLOGY",
        "condition": "diabetes",
        "risk_level": "MEDIUM",
        "pathway": "HbA1c + fasting glucose. Metformin first-line if eGFR ≥45. Annual eye exam, foot exam, nephrology screening. Lifestyle counseling.",
        "tags": "diabetes, glucose, metformin, insulin, HbA1c, endocrine",
    },
    {
        "id": "cp_007",
        "specialty": "GASTROENTEROLOGY",
        "condition": "abdominal pain",
        "risk_level": "MEDIUM",
        "pathway": "Abdominal exam + CBC, CMP, lipase. CT abdomen/pelvis if acute. NPO if surgical etiology suspected. GI consult for upper/lower GI bleed.",
        "tags": "abdominal pain, nausea, vomiting, GI, appendix, gallbladder",
    },
    {
        "id": "cp_008",
        "specialty": "GENERAL",
        "condition": "routine follow-up",
        "risk_level": "LOW",
        "pathway": "Routine vitals, medication reconciliation, preventive screening per age/sex guidelines. Follow-up labs if chronic conditions present.",
        "tags": "routine, follow-up, preventive, wellness, checkup",
    },
    {
        "id": "cp_009",
        "specialty": "ONCOLOGY",
        "condition": "cancer",
        "risk_level": "HIGH",
        "pathway": "Multidisciplinary tumor board. Staging workup (CT/PET). Tissue biopsy if not done. Oncology consult. Palliative care referral as appropriate.",
        "tags": "cancer, tumor, oncology, biopsy, chemotherapy, radiation",
    },
    {
        "id": "cp_010",
        "specialty": "PSYCHIATRY",
        "condition": "mental health",
        "risk_level": "MEDIUM",
        "pathway": "Psychiatric evaluation. Safety assessment. PHQ-9/GAD-7 scoring. Consider inpatient if SI/HI present. Outpatient CBT + pharmacotherapy.",
        "tags": "depression, anxiety, psychiatric, mental health, suicide, SI",
    },
    {
        "id": "cp_011",
        "specialty": "BARIATRICS",
        "condition": "morbid obesity",
        "risk_level": "MEDIUM",
        "pathway": "Pre-op bariatric risk review. Optimize diabetes, hypertension, and sleep apnea. Confirm DVT/PE prophylaxis plan. Review surgical leak, bleeding, and infection risks. Coordinate nutrition, anesthesia, and bariatric surgery follow-up.",
        "tags": "bariatrics, bariatric surgery, obesity, BMI, diabetes, sleep apnea, DVT, pulmonary embolism, leak",
    },
]


class CarePathwayRetriever:
    """ChromaDB-backed retriever for care pathways and similar cases."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        self.use_chroma = False
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        if not settings.use_vector_retrieval:
            logger.info(
                "Vector retrieval disabled. Using keyword care-pathway retrieval."
            )
            return

        try:
            self.ef = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="care_pathways",
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )

            # Seed if empty
            if self.collection.count() == 0:
                self._seed_pathways()
            self.use_chroma = True
        except Exception as e:
            logger.warning(
                f"ChromaDB vector retrieval unavailable ({e}). Using keyword fallback."
            )

    def _seed_pathways(self) -> None:
        """Load care pathway knowledge base into ChromaDB."""
        logger.info("Seeding care pathway knowledge base into ChromaDB...")
        self.collection.add(
            ids=[cp["id"] for cp in CARE_PATHWAYS],
            documents=[
                f"{cp['condition']} {cp['specialty']} {cp['tags']} {cp['pathway']}"
                for cp in CARE_PATHWAYS
            ],
            metadatas=[
                {k: v for k, v in cp.items() if k != "id"} for cp in CARE_PATHWAYS
            ],
        )
        logger.info(f"Seeded {len(CARE_PATHWAYS)} care pathways.")

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        risk_level_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve top-N care pathways matching the query.

        Args:
            query: Clinical note text or entity summary
            n_results: Number of results to return
            risk_level_filter: Optional — filter by 'HIGH', 'MEDIUM', 'LOW'

        Returns:
            List of care pathway dicts with relevance metadata
        """
        if not getattr(self, "use_chroma", True):
            return self._keyword_retrieve(
                query=query, n_results=n_results, risk_level_filter=risk_level_filter
            )

        where = None
        if risk_level_filter:
            where = {"risk_level": risk_level_filter.upper()}

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
            where=where,
        )

        pathways = []
        if results and results["metadatas"]:
            for i, meta in enumerate(results["metadatas"][0]):
                distance = (
                    results["distances"][0][i] if results.get("distances") else 1.0
                )
                pathways.append(
                    {
                        **meta,
                        "relevance_score": round(1 - distance, 3),
                    }
                )

        return pathways

    def _keyword_retrieve(
        self,
        query: str,
        n_results: int = 3,
        risk_level_filter: Optional[str] = None,
    ) -> list[dict]:
        """No-network fallback retriever for reliable local demos."""
        query_terms = {
            token for token in query.lower().replace(",", " ").split() if len(token) > 2
        }
        candidates = CARE_PATHWAYS
        if risk_level_filter:
            candidates = [
                cp for cp in candidates if cp["risk_level"] == risk_level_filter.upper()
            ]

        scored = []
        for cp in candidates:
            haystack = " ".join(
                [
                    cp["specialty"],
                    cp["condition"],
                    cp["risk_level"],
                    cp["pathway"],
                    cp["tags"],
                ]
            ).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if cp["condition"].lower() in query.lower():
                score += 5
            scored.append((score, cp))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: max(1, n_results)]
        max_score = max([score for score, _ in top] + [1])

        return [
            {
                **cp,
                "relevance_score": round(score / max_score, 3) if max_score else 0.0,
            }
            for score, cp in top
        ]

    def add_case(
        self,
        case_id: str,
        note_summary: str,
        entities: dict,
        risk_score: float,
        risk_level: str,
        care_pathway: str,
    ) -> None:
        """Add a processed case to ChromaDB for future RAG retrieval."""
        if not getattr(self, "use_chroma", True):
            logger.info("Skipping case add because vector retrieval is disabled.")
            return

        doc = f"{note_summary} entities: {json.dumps(entities)}"
        self.collection.add(
            ids=[f"case_{case_id}"],
            documents=[doc],
            metadatas=[
                {
                    "condition": note_summary[:200],
                    "specialty": "GENERAL",
                    "risk_level": risk_level,
                    "pathway": care_pathway,
                    "risk_score": str(round(risk_score, 3)),
                    "tags": " ".join(
                        entities.get("diagnoses", [])[:3]
                        + entities.get("symptoms", [])[:2]
                    ),
                }
            ],
        )


# Singleton
_retriever: Optional[CarePathwayRetriever] = None


def get_retriever(persist_dir: str = "data/chroma_db") -> CarePathwayRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CarePathwayRetriever(persist_dir=persist_dir)
    return _retriever
