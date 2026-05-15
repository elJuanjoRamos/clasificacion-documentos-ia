"""
embeddings.py — Lógica de comparación semántica y decisión del agente.

Usa ChromaDB (via vector_store.py) como índice vectorial persistente.
Los vectores ya NO se recomputan en cada llamada — ChromaDB los guarda en disco.

Flujo:
  1. compare_text_with_memory()         → busca en ChromaDB por texto raw
  2. should_call_llm_from_similarity()  → decide si necesita LLM
  3. compare_semantic_profile_with_memory() → busca en ChromaDB por perfil LLM
  4. decide_category_from_similarity()  → decisión final del agente
  5. sync_category_to_vector_store()    → se llama después de update_category_with_document()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.category_memory.category_memory import (
    MEMORY_PATH,
    build_category_profile_text,
    build_semantic_profile_text,
    find_category_by_name,
    infer_general_category,
    load_category_memory,
    slugify,
)
from backend.embeddings.vector_store import (
    query as chroma_query,
    upsert_category,
    sync_all_from_memory,
    count as chroma_count,
)

# ── Umbrales de decisión ─────────────────────────────────────────────────────
# Calibrar con el dataset de evaluación del TFM.

HIGH_SIMILARITY_THRESHOLD   = 0.86   # ≥ este valor → reutiliza sin LLM
MEDIUM_SIMILARITY_THRESHOLD = 0.65   # entre éste y HIGH → llama LLM para validar
LOW_REVIEW_THRESHOLD        = 0.45   # < este valor → categoría nueva provisional

MAX_DIRECT_TEXT_CHARS = 4000         # chars máx enviados como query a ChromaDB


# ── Helpers internos ─────────────────────────────────────────────────────────

def _category_penalty(category: dict[str, Any]) -> float:
    """
    Reduce el score de categorías provisionales para que no bloqueen
    innecesariamente la llamada al LLM.
    """
    if category.get("status") == "provisional" or category.get("requires_review"):
        return 0.03
    return 0.0


def _enrich_match(
    chroma_match: dict[str, Any],
    similarity: float,
    source: str,
    memory: dict[str, Any],
) -> dict[str, Any]:
    """
    Combina el resultado de ChromaDB (vector/score) con los metadatos ricos
    del category_memory.json (palabras_clave, subcategorías, ejemplos...).
    """
    cat_id   = chroma_match.get("category_id", "")
    cat_name = chroma_match.get("name", cat_id)

    # Buscar en JSON para campos ricos
    category = find_category_by_name(memory, cat_name) or {}

    adjusted = max(0.0, float(similarity) - _category_penalty(
        {"status": chroma_match.get("status"), "requires_review": chroma_match.get("requires_review")}
    ))

    return {
        "categoria":        category.get("name") or cat_name,
        "category_id":      cat_id,
        "similitud":        round(adjusted, 4),
        "similitud_raw":    round(float(similarity), 4),
        "document_count":   category.get("document_count", chroma_match.get("document_count", 0)),
        "status":           category.get("status", chroma_match.get("status", "provisional")),
        "requires_review":  category.get("requires_review", chroma_match.get("requires_review", True)),
        "descripcion":      category.get("description", chroma_match.get("description", "")),
        "subcategories":    category.get("subcategories", []),
        "palabras_clave":   category.get("palabras_clave", []),
        "match_source":     source,
    }


# ── Comparación principal ────────────────────────────────────────────────────

def compare_text_with_memory(
    text: str,
    top_k: int = 3,
    memory_path: Path | str = MEMORY_PATH,
) -> list[dict[str, Any]]:
    """
    Comparación PRE-LLM: busca en ChromaDB usando el texto crudo del documento.
    Mucho más rápido que la versión anterior (vectores ya persistidos en disco).
    """
    if not text or not text.strip():
        return []

    if chroma_count() == 0:
        return []

    query_text = text[:MAX_DIRECT_TEXT_CHARS]
    raw_matches = chroma_query(query_text, top_k=top_k)

    if not raw_matches:
        return []

    memory = load_category_memory(memory_path)
    return [
        _enrich_match(m, m["similitud_raw"], "document_text", memory)
        for m in raw_matches
    ]


def compare_semantic_profile_with_memory(
    semantic_profile: dict[str, Any],
    top_k: int = 3,
    memory_path: Path | str = MEMORY_PATH,
) -> list[dict[str, Any]]:
    """
    Comparación POST-LLM: busca en ChromaDB usando el perfil semántico generado
    por el LLM. Más preciso que la comparación por texto crudo.
    """
    if not semantic_profile:
        return []

    if chroma_count() == 0:
        return []

    query_text  = build_semantic_profile_text(semantic_profile)
    raw_matches = chroma_query(query_text, top_k=top_k)

    if not raw_matches:
        return []

    memory = load_category_memory(memory_path)
    return [
        _enrich_match(m, m["similitud_raw"], "semantic_profile", memory)
        for m in raw_matches
    ]


def sync_category_to_vector_store(
    memory: dict[str, Any],
    category_name: str,
) -> None:
    """
    Sincroniza una sola categoría con ChromaDB después de actualizar el JSON.
    Se llama desde apply_optimized_semantic_flow en procesador.py.
    """
    category = find_category_by_name(memory, category_name)
    if category is None:
        return

    profile_text = build_category_profile_text(category)
    upsert_category(category, profile_text)


def rebuild_vector_store_from_memory(memory_path: Path | str = MEMORY_PATH) -> int:
    """
    Reconstruye ChromaDB completo desde category_memory.json.
    Útil para:
      - Migración inicial desde el sistema anterior (sin ChromaDB)
      - Reparar el índice si se corrompe
      - Forzar re-embedding tras cambio de modelo
    """
    memory = load_category_memory(memory_path)
    return sync_all_from_memory(memory, build_category_profile_text)


# ── Lógica de decisión del agente ────────────────────────────────────────────
# Estas funciones NO cambian — trabajan sobre los dicts de matches,
# independientemente de si vinieron de ChromaDB o del sistema anterior.

def should_call_llm_from_similarity(
    memory_matches: list[dict[str, Any]],
    high_threshold:   float = HIGH_SIMILARITY_THRESHOLD,
    medium_threshold: float = MEDIUM_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    """
    Decide si invocar el LLM basándose en los matches pre-LLM.
    """
    if not memory_matches:
        return {
            "call_llm":       True,
            "level":          "sin_memoria",
            "reason":         "No hay categorías en ChromaDB. Se necesita el LLM para crear el primer perfil.",
            "best_similarity": 0.0,
            "best_category":  None,
        }

    best       = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))

    if best.get("requires_review") or best.get("status") == "provisional":
        return {
            "call_llm":        True,
            "level":           "categoria_provisional",
            "reason":          "La mejor categoría es provisional. Se llama al LLM para evitar consolidar errores.",
            "best_similarity": round(similarity, 4),
            "best_category":   best.get("categoria"),
        }

    if similarity >= high_threshold:
        return {
            "call_llm":        False,
            "level":           "alta",
            "reason":          "Similitud alta contra categoría activa. Se reutiliza sin LLM.",
            "best_similarity": round(similarity, 4),
            "best_category":   best.get("categoria"),
        }

    if similarity >= medium_threshold:
        return {
            "call_llm":        True,
            "level":           "media",
            "reason":          "Similitud media. Se llama al LLM para validar o corregir.",
            "best_similarity": round(similarity, 4),
            "best_category":   best.get("categoria"),
        }

    return {
        "call_llm":        True,
        "level":           "baja",
        "reason":          "Similitud baja. Se llama al LLM para generar una categoría nueva o más precisa.",
        "best_similarity": round(similarity, 4),
        "best_category":   best.get("categoria"),
    }


def build_decision_without_llm(memory_matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Ruta rápida: categoría tomada directamente de ChromaDB por similitud alta."""
    best       = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))
    return {
        "categoria_final":  best.get("categoria"),
        "origen_categoria": "existente",
        "confianza":        "alta",
        "similitud":        round(similarity, 4),
        "requiere_revision": False,
        "llm_usado":        False,
        "motivo":           "Categoría activa reutilizada por similitud alta en ChromaDB (sin LLM).",
    }


def get_confidence_from_similarity(similarity: float) -> str:
    if similarity >= HIGH_SIMILARITY_THRESHOLD:
        return "alta"
    if similarity >= MEDIUM_SIMILARITY_THRESHOLD:
        return "media"
    return "baja"


def decide_category_from_similarity(
    semantic_profile: dict[str, Any],
    memory_matches:   list[dict[str, Any]],
    high_threshold:      float = HIGH_SIMILARITY_THRESHOLD,
    medium_threshold:    float = MEDIUM_SIMILARITY_THRESHOLD,
    low_review_threshold: float = LOW_REVIEW_THRESHOLD,
) -> dict[str, Any]:
    """
    Decisión final del agente después de obtener el perfil LLM y compararlo
    con las categorías en ChromaDB.
    """
    suggested  = infer_general_category(semantic_profile)

    if not memory_matches:
        return {
            "categoria_final":   suggested,
            "origen_categoria":  "nueva",
            "confianza":         "baja",
            "similitud":         0.0,
            "requiere_revision": True,
            "llm_usado":         True,
            "motivo":            "Sin categorías en ChromaDB. Se propone categoría nueva provisional.",
        }

    best       = memory_matches[0]
    similarity = float(best.get("similitud", 0.0))
    confidence = get_confidence_from_similarity(similarity)

    if (
        similarity >= high_threshold
        and not best.get("requires_review")
        and best.get("status") == "active"
    ):
        return {
            "categoria_final":   best.get("categoria"),
            "origen_categoria":  "existente",
            "confianza":         confidence,
            "similitud":         round(similarity, 4),
            "requiere_revision": False,
            "llm_usado":         True,
            "motivo":            "Perfil LLM coincide con categoría activa en ChromaDB.",
        }

    if similarity >= medium_threshold:
        return {
            "categoria_final":   best.get("categoria"),
            "origen_categoria":  "revision",
            "confianza":         confidence,
            "similitud":         round(similarity, 4),
            "requiere_revision": True,
            "llm_usado":         True,
            "motivo":            "Similitud media/alta pero requiere revisión antes de consolidar.",
        }

    if similarity < low_review_threshold:
        return {
            "categoria_final":   suggested,
            "origen_categoria":  "nueva",
            "confianza":         confidence,
            "similitud":         round(similarity, 4),
            "requiere_revision": True,
            "llm_usado":         True,
            "motivo":            "Similitud baja. Se propone categoría nueva provisional.",
        }

    return {
        "categoria_final":   suggested,
        "origen_categoria":  "revision",
        "confianza":         confidence,
        "similitud":         round(similarity, 4),
        "requiere_revision": True,
        "llm_usado":         True,
        "motivo":            "Similitud intermedia. Requiere revisión antes de decidir.",
    }


# ── Compatibilidad hacia atrás ───────────────────────────────────────────────
# Alias para no romper imports existentes en tests o código externo

def encode_text(text: str):
    """Genera embedding de texto. Delega al modelo interno de vector_store."""
    from backend.embeddings.vector_store import _LocalEmbeddingFn
    fn = _LocalEmbeddingFn()
    return fn([text])[0]