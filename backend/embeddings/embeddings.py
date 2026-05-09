from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.category_memory.category_memory import (
    MEMORY_PATH,
    build_category_profile_text,
    build_semantic_profile_text,
    infer_general_category,
    load_category_memory,
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Umbrales iniciales. Deben calibrarse con el dataset de evaluación del TFM.
HIGH_SIMILARITY_THRESHOLD = 0.86
MEDIUM_SIMILARITY_THRESHOLD = 0.65
LOW_REVIEW_THRESHOLD = 0.45
MAX_DIRECT_TEXT_CHARS = 4000

_model = SentenceTransformer(MODEL_NAME)


def encode_text(text: str) -> np.ndarray:
    return _model.encode([text], normalize_embeddings=True)[0]


def _category_penalty(category: dict[str, Any]) -> float:
    """
    Las categorías provisionales no deben bloquear tan fácilmente la llamada al LLM.
    """
    if category.get("status") == "provisional" or category.get("requires_review"):
        return 0.03
    return 0.0


def _build_match(category: dict[str, Any], score: float, source: str) -> dict[str, Any]:
    adjusted = max(0.0, float(score) - _category_penalty(category))
    return {
        "categoria": category.get("name"),
        "category_id": category.get("id"),
        "similitud": round(adjusted, 4),
        "similitud_raw": round(float(score), 4),
        "document_count": category.get("document_count", 0),
        "status": category.get("status", "provisional"),
        "requires_review": category.get("requires_review", False),
        "descripcion": category.get("description", ""),
        "subcategories": category.get("subcategories", []),
        "palabras_clave": category.get("palabras_clave", []),
        "match_source": source
    }


def compare_text_with_memory(
    text: str,
    top_k: int = 3,
    memory_path: Path | str = MEMORY_PATH
) -> list[dict[str, Any]]:
    memory = load_category_memory(memory_path)
    categories = memory.get("categories", [])

    if not text or not text.strip() or not categories:
        return []

    query_text = text[:MAX_DIRECT_TEXT_CHARS]
    category_texts = [build_category_profile_text(category) for category in categories]

    query_embedding = _model.encode([query_text], normalize_embeddings=True)
    category_embeddings = _model.encode(category_texts, normalize_embeddings=True)

    scores = cosine_similarity(query_embedding, category_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [_build_match(categories[int(index)], float(scores[index]), "document_text") for index in top_indices]


def compare_semantic_profile_with_memory(
    semantic_profile: dict[str, Any],
    top_k: int = 3,
    memory_path: Path | str = MEMORY_PATH
) -> list[dict[str, Any]]:
    memory = load_category_memory(memory_path)
    categories = memory.get("categories", [])

    if not semantic_profile or not categories:
        return []

    query_text = build_semantic_profile_text(semantic_profile)
    category_texts = [build_category_profile_text(category) for category in categories]

    query_embedding = _model.encode([query_text], normalize_embeddings=True)
    category_embeddings = _model.encode(category_texts, normalize_embeddings=True)

    scores = cosine_similarity(query_embedding, category_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [_build_match(categories[int(index)], float(scores[index]), "semantic_profile") for index in top_indices]


def should_call_llm_from_similarity(
    memory_matches: list[dict[str, Any]],
    high_threshold: float = HIGH_SIMILARITY_THRESHOLD,
    medium_threshold: float = MEDIUM_SIMILARITY_THRESHOLD
) -> dict[str, Any]:
    if not memory_matches:
        return {
            "call_llm": True,
            "level": "sin_memoria",
            "reason": "No hay categorías previas en memoria. Se necesita el LLM para crear el primer perfil semántico.",
            "best_similarity": 0.0,
            "best_category": None,
        }

    best = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))

    if best.get("requires_review") or best.get("status") == "provisional":
        return {
            "call_llm": True,
            "level": "categoria_provisional",
            "reason": "La mejor categoría aún es provisional; se llama al LLM para evitar consolidar errores.",
            "best_similarity": round(similarity, 4),
            "best_category": best.get("categoria"),
        }

    if similarity >= high_threshold:
        return {
            "call_llm": False,
            "level": "alta",
            "reason": "La similitud con una categoría activa es alta. Se reutiliza sin llamar al LLM.",
            "best_similarity": round(similarity, 4),
            "best_category": best.get("categoria"),
        }

    if similarity >= medium_threshold:
        return {
            "call_llm": True,
            "level": "media",
            "reason": "La similitud es media. Se llama al LLM para validar o corregir.",
            "best_similarity": round(similarity, 4),
            "best_category": best.get("categoria"),
        }

    return {
        "call_llm": True,
        "level": "baja",
        "reason": "La similitud es baja. Se llama al LLM para generar una categoría nueva o más precisa.",
        "best_similarity": round(similarity, 4),
        "best_category": best.get("categoria"),
    }


def build_decision_without_llm(memory_matches: list[dict[str, Any]]) -> dict[str, Any]:
    best = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))
    return {
        "categoria_final": best.get("categoria"),
        "origen_categoria": "existente",
        "confianza": "alta",
        "similitud": round(similarity, 4),
        "requiere_revision": False,
        "llm_usado": False,
        "motivo": "Categoría activa reutilizada por similitud alta contra memoria antes de llamar al LLM."
    }


def get_confidence_from_similarity(similarity: float) -> str:
    if similarity >= HIGH_SIMILARITY_THRESHOLD:
        return "alta"
    if similarity >= MEDIUM_SIMILARITY_THRESHOLD:
        return "media"
    return "baja"


def decide_category_from_similarity(
    semantic_profile: dict[str, Any],
    memory_matches: list[dict[str, Any]],
    high_threshold: float = HIGH_SIMILARITY_THRESHOLD,
    medium_threshold: float = MEDIUM_SIMILARITY_THRESHOLD,
    low_review_threshold: float = LOW_REVIEW_THRESHOLD
) -> dict[str, Any]:
    suggested = infer_general_category(semantic_profile)

    if not memory_matches:
        return {
            "categoria_final": suggested,
            "origen_categoria": "nueva",
            "confianza": "baja",
            "similitud": 0.0,
            "requiere_revision": True,
            "llm_usado": True,
            "motivo": "No existen categorías previas. Se propone una nueva categoría provisional."
        }

    best = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))
    confidence = get_confidence_from_similarity(similarity)

    if (
        similarity >= high_threshold
        and not best.get("requires_review")
        and best.get("status") == "active"
    ):
        return {
            "categoria_final": best.get("categoria"),
            "origen_categoria": "existente",
            "confianza": confidence,
            "similitud": round(similarity, 4),
            "requiere_revision": False,
            "llm_usado": True,
            "motivo": "El perfil semántico coincide con una categoría activa existente."
        }

    if similarity >= medium_threshold:
        return {
            "categoria_final": best.get("categoria"),
            "origen_categoria": "revision",
            "confianza": confidence,
            "similitud": round(similarity, 4),
            "requiere_revision": True,
            "llm_usado": True,
            "motivo": "Existe similitud media o alta, pero la categoría requiere revisión antes de consolidarse."
        }

    if similarity < low_review_threshold:
        return {
            "categoria_final": suggested,
            "origen_categoria": "nueva",
            "confianza": confidence,
            "similitud": round(similarity, 4),
            "requiere_revision": True,
            "llm_usado": True,
            "motivo": "La similitud es baja. Se propone una nueva categoría provisional."
        }

    return {
        "categoria_final": suggested,
        "origen_categoria": "revision",
        "confianza": confidence,
        "similitud": round(similarity, 4),
        "requiere_revision": True,
        "llm_usado": True,
        "motivo": "La similitud es intermedia. Requiere revisión antes de decidir si reutilizar o crear categoría."
    }