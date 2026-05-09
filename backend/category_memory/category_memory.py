from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import re
import uuid


BASE_DIR = Path(__file__).resolve().parent
MEMORY_PATH = BASE_DIR / "category_memory.json"
MAX_EXAMPLES_PER_CATEGORY = 10
MAX_HISTORY_PER_CATEGORY = 50
MAX_KEYWORDS_PER_CATEGORY = 20


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-záéíóúñü0-9]+", "_", value, flags=re.IGNORECASE)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or f"categoria_{uuid.uuid4().hex[:8]}"


def normalize_text(value: Any, default: str = "") -> str:
    value = str(value or default).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_category_name(value: Any) -> str:
    value = normalize_text(value, "Otros")
    return value[:80] if value else "Otros"


def normalize_confidence(value: Any) -> str:
    value = normalize_text(value, "baja").lower()
    return value if value in {"alta", "media", "baja"} else "baja"


def empty_memory() -> dict[str, Any]:
    now = now_iso()
    return {
        "version": 2,
        "created_at": now,
        "updated_at": now,
        "categories": [],
        "pending_reviews": []
    }


def load_category_memory(memory_path: Path | str = MEMORY_PATH) -> dict[str, Any]:
    path = Path(memory_path)
    if not path.exists():
        memory = empty_memory()
        save_category_memory(memory, path)
        return memory

    try:
        with path.open("r", encoding="utf-8") as file:
            memory = json.load(file)
    except json.JSONDecodeError:
        memory = empty_memory()

    memory.setdefault("version", 2)
    memory.setdefault("created_at", now_iso())
    memory.setdefault("updated_at", now_iso())
    memory.setdefault("categories", [])
    memory.setdefault("pending_reviews", [])
    return memory


def save_category_memory(memory: dict[str, Any], memory_path: Path | str = MEMORY_PATH) -> None:
    path = Path(memory_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    memory["updated_at"] = now_iso()
    with path.open("w", encoding="utf-8") as file:
        json.dump(memory, file, ensure_ascii=False, indent=2)


def normalize_keywords(keywords: Any, limit: int = 8) -> list[str]:
    if isinstance(keywords, str):
        keywords = [item.strip() for item in keywords.split(",")]
    if not isinstance(keywords, list):
        return []

    result: list[str] = []
    for item in keywords:
        keyword = normalize_text(item)
        if keyword and keyword.lower() not in {k.lower() for k in result}:
            result.append(keyword)
        if len(result) >= limit:
            break
    return result


def infer_general_category(profile: dict[str, Any]) -> str:
    """
    Evita que el tema puntual del documento se convierta en categoría.
    Prioriza categoria_general si el LLM la devuelve; si no, aplica reglas simples.
    """
    explicit = normalize_text(profile.get("categoria_general"))
    if explicit:
        return normalize_category_name(explicit)

    category = normalize_text(profile.get("categoria_sugerida"))
    doc_type = normalize_text(profile.get("tipo_documento")).lower()
    area = normalize_text(profile.get("area_funcional")).lower()
    topic = normalize_text(profile.get("tema")).lower()

    joined = " ".join([category.lower(), doc_type, area, topic])

    if any(term in joined for term in ["tfm", "trabajo de fin", "tesis", "académic", "universidad", "máster", "master"]):
        return "Documentos académicos"
    if "financier" in joined or any(term in joined for term in ["ingresos", "gastos", "beneficio", "roi"]):
        return "Informes financieros"
    if any(term in joined for term in ["diligencia", "juzgado", "denuncia", "policial", "judicial"]):
        return "Comunicaciones judiciales"
    if "factura" in joined:
        return "Facturas"
    if "contrato" in joined:
        return "Contratos"
    if "acta" in joined:
        return "Actas"
    if "certificado" in joined or "constancia" in joined:
        return "Certificados"
    if category:
        return normalize_category_name(category)
    return "Otros"


def infer_subcategory(profile: dict[str, Any]) -> str:
    explicit = normalize_text(profile.get("subcategoria"))
    if explicit:
        return explicit[:80]

    doc_type = normalize_text(profile.get("tipo_documento"))
    topic = normalize_text(profile.get("tema"))
    category = infer_general_category(profile)

    joined = f"{doc_type} {topic}".lower()
    if category == "Documentos académicos":
        if "tfm" in joined or "trabajo de fin" in joined:
            return "Memoria TFM"
        return "Documento académico"
    if category == "Informes financieros":
        return "Informe financiero"
    if category == "Comunicaciones judiciales":
        return doc_type or "Diligencia judicial"
    return doc_type or "General"


def build_category_description(profile: dict[str, Any]) -> str:
    category = infer_general_category(profile)
    subcategory = infer_subcategory(profile)

    return (
        f"Categoría para documentos de tipo {category}. "
        f"Subtipo frecuente: {subcategory}. "
        "Esta categoría agrupa documentos por tipo documental, no por tema específico."
    )

def build_semantic_profile_text(profile: dict[str, Any]) -> str:
    keywords = "; ".join(normalize_keywords(profile.get("palabras_clave"), limit=8))

    return "\n".join([
        f"Tipo documental: {profile.get('categoria_general', '')}",
        f"Subtipo documental: {profile.get('subcategoria', '')}",
        f"Tema: {profile.get('tema', '')}",
        f"Conceptos clave: {keywords}",
        f"Resumen: {profile.get('resumen', '')}"
    ]).strip()



def build_category_profile_text(category: dict[str, Any]) -> str:
    keywords = "; ".join(normalize_keywords(category.get("palabras_clave"), limit=12))
    subcategories = "; ".join(category.get("subcategories", [])[:8])

    return "\n".join([
        f"Tipo documental: {category.get('name', '')}",
        f"Subtipos documentales frecuentes: {subcategories}",
        f"Conceptos clave frecuentes: {keywords}",
        f"Descripción general: {category.get('description', '')}"
    ]).strip()


def find_category_by_name(memory: dict[str, Any], category_name: str) -> dict[str, Any] | None:
    normalized = normalize_category_name(category_name).lower()
    for category in memory.get("categories", []):
        if normalize_category_name(category.get("name", "")).lower() == normalized:
            return category
    return None


def append_unique(target: list[str], values: list[str], max_items: int) -> list[str]:
    existing = {normalize_text(item).lower() for item in target}
    for value in values:
        item = normalize_text(value)
        if item and item.lower() not in existing:
            target.append(item)
            existing.add(item.lower())
        if len(target) >= max_items:
            break
    return target


def create_category_from_profile(profile: dict[str, Any], final_decision: dict[str, Any]) -> dict[str, Any]:
    name = normalize_category_name(final_decision.get("categoria_final") or infer_general_category(profile))
    now = now_iso()
    return {
        "id": slugify(name),
        "name": name,
        "description": build_category_description(profile),
        "status": "provisional" if final_decision.get("requires_review") or final_decision.get("requiere_revision") else "active",
        "requires_review": bool(final_decision.get("requiere_revision", False)),
        "subcategories": [infer_subcategory(profile)],
        "tipo_documento": [normalize_text(profile.get("tipo_documento"))] if normalize_text(profile.get("tipo_documento")) else [],
        "area_funcional": [normalize_text(profile.get("area_funcional"))] if normalize_text(profile.get("area_funcional")) else [],
        "palabras_clave": normalize_keywords(profile.get("palabras_clave"), limit=MAX_KEYWORDS_PER_CATEGORY),
        "created_at": now,
        "updated_at": now,
        "document_count": 0,
        "examples": [],
        "history": []
    }


def register_pending_review(memory: dict[str, Any], semantic_profile: dict[str, Any], final_decision: dict[str, Any], document_result: dict[str, Any]) -> None:
    item = {
        "file_name": document_result.get("file_name"),
        "full_path": document_result.get("full_path"),
        "categoria_propuesta": final_decision.get("categoria_final"),
        "origen_categoria": final_decision.get("origen_categoria"),
        "similitud": final_decision.get("similitud"),
        "motivo": final_decision.get("motivo"),
        "semantic_profile": semantic_profile,
        "created_at": now_iso()
    }
    pending = memory.setdefault("pending_reviews", [])
    pending.insert(0, item)
    memory["pending_reviews"] = pending[:100]


def update_category_with_document(
    memory: dict[str, Any],
    category_name: str,
    semantic_profile: dict[str, Any],
    final_decision: dict[str, Any],
    document_result: dict[str, Any],
    memory_path: Path | str = MEMORY_PATH,
    save_provisional: bool = True
) -> dict[str, Any] | None:
    """
    Guarda memoria sin convertir automáticamente una categoría nueva en definitiva.

    - Categorías nuevas: se guardan como provisional + requires_review=True.
    - Categorías existentes: se actualizan, pero si la decisión requiere revisión también se registra en pending_reviews.
    - Errores/no determinado: no se guardan como categoría útil.
    """
    category_name = normalize_category_name(category_name)
    if category_name.lower() in {"no determinado", "otros"} and final_decision.get("requiere_revision"):
        register_pending_review(memory, semantic_profile, final_decision, document_result)
        save_category_memory(memory, memory_path)
        return None

    category = find_category_by_name(memory, category_name)
    if category is None:
        if not save_provisional and final_decision.get("requiere_revision"):
            register_pending_review(memory, semantic_profile, final_decision, document_result)
            save_category_memory(memory, memory_path)
            return None
        category = create_category_from_profile(semantic_profile, final_decision)
        category["name"] = category_name
        category["id"] = slugify(category_name)
        memory.setdefault("categories", []).append(category)

    category["updated_at"] = now_iso()
    category["document_count"] = int(category.get("document_count", 0)) + 1

    if final_decision.get("requiere_revision"):
        category["requires_review"] = True
        if category.get("status") != "active":
            category["status"] = "provisional"
        register_pending_review(memory, semantic_profile, final_decision, document_result)

    category["subcategories"] = append_unique(
        category.setdefault("subcategories", []),
        [infer_subcategory(semantic_profile)],
        20
    )
    category["tipo_documento"] = append_unique(
        category.setdefault("tipo_documento", []),
        [normalize_text(semantic_profile.get("tipo_documento"))],
        20
    )
    category["area_funcional"] = append_unique(
        category.setdefault("area_funcional", []),
        [normalize_text(semantic_profile.get("area_funcional"))],
        10
    )
    category["palabras_clave"] = append_unique(
        category.setdefault("palabras_clave", []),
        normalize_keywords(semantic_profile.get("palabras_clave"), limit=8),
        MAX_KEYWORDS_PER_CATEGORY
    )

    example = {
        "file_name": document_result.get("file_name"),
        "full_path": document_result.get("full_path"),
        "subcategoria": infer_subcategory(semantic_profile),
        "tipo_documento": semantic_profile.get("tipo_documento"),
        "area_funcional": semantic_profile.get("area_funcional"),
        "tema": semantic_profile.get("tema"),
        "palabras_clave": normalize_keywords(semantic_profile.get("palabras_clave"), limit=8),
        "decision": final_decision,
        "created_at": now_iso()
    }

    examples = category.setdefault("examples", [])
    examples.insert(0, example)
    category["examples"] = examples[:MAX_EXAMPLES_PER_CATEGORY]

    history = category.setdefault("history", [])
    history.insert(0, {
        "event": "document_classified",
        "file_name": document_result.get("file_name"),
        "origen_categoria": final_decision.get("origen_categoria"),
        "confianza": final_decision.get("confianza"),
        "similitud": final_decision.get("similitud"),
        "requiere_revision": final_decision.get("requiere_revision"),
        "created_at": now_iso()
    })
    category["history"] = history[:MAX_HISTORY_PER_CATEGORY]

    save_category_memory(memory, memory_path)
    return category


def build_profile_from_existing_category(
    memory: dict[str, Any],
    category_name: str,
    document_text: str = "",
    max_summary_chars: int = 350
) -> dict[str, Any]:
    category = find_category_by_name(memory, category_name) or {}
    subcategories = category.get("subcategories", []) or []
    tipos = category.get("tipo_documento", []) or []
    areas = category.get("area_funcional", []) or []

    return {
        "categoria_general": normalize_category_name(category_name),
        "categoria_sugerida": normalize_category_name(category_name),
        "subcategoria": subcategories[0] if subcategories else "General",
        "tipo_documento": tipos[0] if tipos else "No determinado",
        "area_funcional": areas[0] if areas else "otra",
        "tema": "No determinado sin LLM",
        "palabras_clave": category.get("palabras_clave", [])[:8],
        "resumen": (document_text or category.get("description", ""))[:max_summary_chars],
        "confianza_llm": "no_usado",
        "justificacion": "Perfil mínimo generado desde una categoría existente por similitud alta. No se llamó al LLM."
    }
