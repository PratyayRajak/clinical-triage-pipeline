"""
ONNX Export Utility
Exports sentence-transformer embedding model to ONNX format
for optimized inference (faster ChromaDB embedding calls).

Usage:
    python src/monitoring/onnx_export.py
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ONNX_OUTPUT_DIR = Path("data/onnx_models")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def export_embedding_model_to_onnx(
    model_name: str = MODEL_NAME,
    output_dir: str = str(ONNX_OUTPUT_DIR),
) -> str:
    """
    Export a sentence-transformer model to ONNX via HuggingFace Optimum.
    Returns path to exported ONNX model directory.
    """
    output_path = Path(output_dir) / model_name.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_name} to ONNX → {output_path}")

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ORTModelForFeatureExtraction.from_pretrained(
            model_name,
            export=True,
        )

        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))

        logger.info(f"ONNX export complete: {output_path}")
        return str(output_path)

    except ImportError:
        logger.warning(
            "optimum[onnxruntime] not installed. Run: pip install optimum[onnxruntime]"
        )
        return ""
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return ""


class ONNXEmbedder:
    """
    Drop-in ONNX-optimized embedder.
    Falls back to sentence-transformers if ONNX model not available.
    """

    def __init__(self, onnx_model_dir: str = ""):
        self._onnx_model = None
        self._st_model = None

        if onnx_model_dir and Path(onnx_model_dir).exists():
            try:
                import torch
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(onnx_model_dir)
                self._onnx_model = ORTModelForFeatureExtraction.from_pretrained(
                    onnx_model_dir
                )
                logger.info("ONNX embedder loaded.")
            except Exception as e:
                logger.warning(f"ONNX load failed: {e}. Falling back to ST.")

        if self._onnx_model is None:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence-transformer embedder loaded (fallback).")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a list of texts."""
        if self._onnx_model is not None:
            import torch

            inputs = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._onnx_model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy().tolist()
        else:
            return self._st_model.encode(texts, show_progress_bar=False).tolist()


if __name__ == "__main__":
    path = export_embedding_model_to_onnx()
    if path:
        print(f"Model exported to: {path}")

        # Test inference
        embedder = ONNXEmbedder(onnx_model_dir=path)
        test_embeds = embedder.embed(
            ["Patient presents with chest pain and shortness of breath."]
        )
        print(f"Embedding dim: {len(test_embeds[0])}")
