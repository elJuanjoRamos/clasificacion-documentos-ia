"""
vector_store.py — Capa de persistencia vectorial con ChromaDB (100% local).

Arquitectura:
  - ChromaDB persiste los vectores en disco (backend/embeddings/chroma_db/)
  - category_memory.json sigue siendo la fuente de verdad de metadatos ricos
    (subcategorías, ejemplos, historial, palabras clave)
  - ChromaDB es el índice vectorial: almacena el texto de perfil de cada
    categoría ya embebido, listo para búsquedas rápidas sin recomputar.

Flujo de datos:
  Clasificar documento
    → query_by_text() / query_by_profile()   ← búsqueda rápida en ChromaDB
    → update_category_with_document()         ← guarda JSON
    → upsert_category()                       ← sincroniza vector en ChromaDB
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

# ── Configuración ───────────────────────────────────────────────────────────

MODEL_NAME      = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHROMA_DIR      = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "category_profiles"

# ── Función de embedding personalizada para ChromaDB ────────────────────────

class _LocalEmbeddingFn(EmbeddingFunction):
    """
    Usa el mismo modelo sentence-transformers que el resto del sistema.
    ChromaDB la llama internamente cuando hace upsert/query con texto.
    """
    _model: SentenceTransformer | None = None
    _lock  = threading.Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def __call__(self, input: Documents) -> Embeddings:
        model = self._get_model()
        vectors = model.encode(list(input), normalize_embeddings=True)
        return vectors.tolist()


# ── Singleton: cliente y colección ChromaDB ─────────────────────────────────

_client:     chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_write_lock  = threading.Lock()


def _get_collection() -> chromadb.Collection:
    """
    Retorna (o crea) la colección ChromaDB persistente.
    Thread-safe: el singleton se inicializa una sola vez.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    with _write_lock:
        if _collection is not None:      # doble check post-lock
            return _collection

        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_LocalEmbeddingFn(),
            metadata={"hnsw:space": "cosine"},   # distancia coseno nativa
        )

    return _collection


# ── API pública ─────────────────────────────────────────────────────────────

def count() -> int:
    """Número de categorías indexadas en ChromaDB."""
    return _get_collection().count()


def upsert_category(category: dict[str, Any], profile_text: str) -> None:
    """
    Inserta o actualiza el vector de una categoría en ChromaDB.
    Se llama después de cada update_category_with_document().

    Args:
        category:     dict de la categoría (de category_memory.json)
        profile_text: texto generado por build_category_profile_text()
    """
    cat_id = str(category.get("id") or category.get("name", "unknown"))

    metadata = {
        "name":             str(category.get("name", "")),
        "status":           str(category.get("status", "provisional")),
        "requires_review":  str(category.get("requires_review", True)),
        "document_count":   int(category.get("document_count", 0)),
        "description":      str(category.get("description", ""))[:500],
    }

    with _write_lock:
        _get_collection().upsert(
            ids=[cat_id],
            documents=[profile_text],
            metadatas=[metadata],
        )


def query(query_text: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Busca las top_k categorías más similares al query_text.

    Retorna lista de dicts con:
        category_id, similitud_raw, status, requires_review,
        document_count, description
    (Los campos ricos —palabras_clave, subcategorías— se enriquecen
    luego en embeddings.py cruzando con category_memory.json.)
    """
    col = _get_collection()

    if col.count() == 0:
        return []

    n = min(top_k, col.count())

    results = col.query(
        query_texts=[query_text],
        n_results=n,
        include=["metadatas", "distances"],
    )

    matches: list[dict[str, Any]] = []

    for i, cat_id in enumerate(results["ids"][0]):
        # ChromaDB cosine distance ∈ [0, 2]
        # similarity = 1 - distance   (para vectores normalizados)
        distance   = float(results["distances"][0][i])
        similarity = round(1.0 - distance, 4)
        meta       = results["metadatas"][0][i]

        matches.append({
            "category_id":    cat_id,
            "similitud_raw":  similarity,
            "status":         meta.get("status", "provisional"),
            "requires_review": meta.get("requires_review", "True") == "True",
            "document_count": int(meta.get("document_count", 0)),
            "description":    meta.get("description", ""),
            "name":           meta.get("name", cat_id),
        })

    return matches


def sync_all_from_memory(memory: dict[str, Any], build_profile_fn) -> int:
    """
    Reconstruye ChromaDB completo desde category_memory.json.
    Útil para migrar una memoria JSON existente o reparar el índice.

    Args:
        memory:           dict cargado con load_category_memory()
        build_profile_fn: función build_category_profile_text de category_memory.py

    Returns:
        Número de categorías sincronizadas.
    """
    categories = memory.get("categories", [])
    count_synced = 0

    for cat in categories:
        profile_text = build_profile_fn(cat)
        upsert_category(cat, profile_text)
        count_synced += 1

    return count_synced


def delete_category(category_id: str) -> None:
    """Elimina una categoría del índice vectorial."""
    with _write_lock:
        _get_collection().delete(ids=[category_id])


def reset_collection() -> None:
    """
    Borra y recrea la colección. DESTRUCTIVO — solo para desarrollo/tests.
    """
    global _collection
    with _write_lock:
        if _client is not None:
            _client.delete_collection(COLLECTION_NAME)
            _collection = None
