import re
import unicodedata

try:
    import spacy
except ImportError:
    spacy = None


TEXT_BLOCK_TYPES = {
    "paragraph",
    "title",
    "list_item",
    "table",
    "image_ocr",
    "ocr_block"
}

MAX_LLM_BLOCKS = 15
MAX_LLM_CHARS = 12000
MAX_BLOCK_TEXT_CHARS = 1200
SPACY_MODEL = "es_core_news_md"

try:
    NLP = spacy.load(SPACY_MODEL) if spacy else None
except OSError:
    NLP = None


# Etiquetas traducidas a nombres claros para el LLM.
SPACY_LABEL_MAP = {
    # spaCy español
    "PER": "personas",
    "PERSON": "personas",
    "ORG": "organizaciones",
    "LOC": "ubicaciones",
    "GPE": "ubicaciones",
    "FAC": "instalaciones",
    "MISC": "conceptos",

    # Fechas, valores y cantidades
    "DATE": "fechas",
    "TIME": "horas",
    "PERCENT": "porcentajes",
    "MONEY": "dinero",
    "QUANTITY": "cantidades",

    # Objetos y conceptos nombrados
    "PRODUCT": "productos",
    "EVENT": "eventos",
    "WORK_OF_ART": "obras",
    "LAW": "leyes",
    "LANGUAGE": "idiomas",

    # Otros
    "NORP": "grupos",
    "ORDINAL": "ordinales",
    "CARDINAL": "numeros",
}

GLOBAL_FEATURE_KEYS = {
    "personas",
    "organizaciones",
    "ubicaciones",
    "instalaciones",
    "conceptos",
    "productos",
    "eventos",
    "obras",
    "leyes",
    "idiomas",
    "grupos",
}

LOCAL_FEATURE_KEYS = {
    "fechas",
    "horas",
    "porcentajes",
    "dinero",
    "cantidades",
    "numeros",
    "ordinales",
}

INTERNAL_FEATURE_KEYS = {
    "text_length",
    "score",
    "nlp_enabled",
}

ACCENT_TRANSLATION = str.maketrans({
    "À": "Á",
    "È": "É",
    "Ì": "Í",
    "Ò": "Ó",
    "Ù": "Ú",
    "à": "á",
    "è": "é",
    "ì": "í",
    "ò": "ó",
    "ù": "ú",
})


DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
    r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b",
]

TIME_PATTERNS = [
    r"\b\d{1,2}:\d{2}(?:\s?(?:a\.m\.|p\.m\.|am|pm))?\b",
]

MONEY_PATTERNS = [
    r"\bQ\.?\s?\d[\d,]*(?:\.\d{2})?\b",
    r"\$\s?\d[\d,]*(?:\.\d{2})?\b",
    r"\b\d[\d,]*(?:\.\d{2})?\s?(?:USD|GTQ|EUR)\b",
]

PERCENT_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?%\b",
]

QUANTITY_PATTERNS = [
    r"\b\d+(?:\.\d+)?\s?(?:kg|g|mg|km|m|cm|mm|l|ml|horas?|días?|meses|años)\b",
]

STRUCTURED_EXTRACTION_STEPS = [
    ("fechas", DATE_PATTERNS),
    ("horas", TIME_PATTERNS),
    ("dinero", MONEY_PATTERNS),
    ("porcentajes", PERCENT_PATTERNS),
    ("cantidades", QUANTITY_PATTERNS),
]


def normalize_extraction_result(extraction_result: dict) -> dict:
    original_blocks = extraction_result.get("blocks", [])
    normalized_blocks = []
    normalized_text_parts = []

    for block in original_blocks:
        normalized_block = normalize_block(block)

        if not normalized_block:
            continue

        enriched_block = get_nlp_data(normalized_block)
        normalized_blocks.append(enriched_block)
        normalized_text_parts.append(enriched_block["text"])

    normalized_text = "\n\n".join(normalized_text_parts).strip()

    selected_blocks, llm_text = build_selected_llm_text(normalized_blocks)

    return {
        "file_type": extraction_result.get("file_type"),
        "extension": extraction_result.get("extension"),
        "supported": extraction_result.get("supported", False),
        "extraction_method": extraction_result.get("extraction_method"),
        "normalized_blocks": normalized_blocks,
        "selected_blocks": selected_blocks,
        "normalized_text": normalized_text,
        "llm_text": llm_text,
        "normalization_quality": build_normalization_quality(
            original_blocks=original_blocks,
            normalized_blocks=normalized_blocks,
            normalized_text=normalized_text,
            llm_text=llm_text,
        ),
        "error": extraction_result.get("error"),
    }


# =========================
# NORMALIZACION
# =========================

def normalize_block(block: dict) -> dict | None:
    block_type = block.get("type")
    accepted = block.get("accepted", True)
    quality = block.get("quality")
    text = clean_text(str(block.get("text", ""))).translate(ACCENT_TRANSLATION)


    if (not text) or (block_type not in TEXT_BLOCK_TYPES) or (not accepted and quality != "medium"):
        return None

    return {
        "order": block.get("order"),
        "type": block_type,
        "text": text,
        "source": block.get("source"),
        "metadata": block.get("metadata", {}),
    }


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(
        char for char in text
        if unicodedata.category(char)[0] != "C" or char == "\n"
    )

    lines = []
    previous = None

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"[_\-–—]{4,}", " ", line).strip()

        if line and line != previous and not is_noise_line(line):
            lines.append(line)

        previous = line

    return "\n".join(lines).strip()


def is_noise_line(line: str) -> bool:
    if len(line) < 2:
        return True

    if re.fullmatch(r"[\d\s.,\-/:|]+", line):
        return True

    symbols = re.findall(r"[^\w\sáéíóúÁÉÍÓÚñÑüÜ.,:;()\-/%@$€Q]", line)
    return len(symbols) / max(len(line), 1) > 0.35


# =========================
# ENRIQUECIMIENTO PLN
# =========================

def get_nlp_data(block: dict) -> dict:
    features = {}
    text = block["text"]

    if block["type"] != "table" and NLP is not None:
        nlp_text = re.sub(r"\s+", " ", text).strip()
        nlp_text = re.sub(
            r"\b[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,})*\b",
            lambda match: match.group(0).title(),
            nlp_text
        )

        doc = NLP(nlp_text)

        for ent in doc.ents:
            key = SPACY_LABEL_MAP.get(ent.label_)

            if key:
                add_unique(features, key, ent.text.strip())

    feature_text = text

    for key, patterns in STRUCTURED_EXTRACTION_STEPS:
        feature_text = add_matches(features, key, feature_text, patterns)

    score = calculate_block_score(
        block_type=block["type"],
        text=text,
        features=features
    )

    return {
        **block,
        "features": {
            **features,
            "text_length": len(text),
            "score": score,
            "nlp_enabled": NLP is not None,
        }
    }


def add_matches(features: dict, key: str, text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)

        for match in matches:
            value = (match if isinstance(match, str) else " ".join(match)).strip()

            if not value:
                continue

            add_unique(features, key, value)
            text = re.sub(re.escape(value), " ", text, flags=re.IGNORECASE)

    return text


def add_unique(features: dict, key: str, value: str) -> None:
    normalized_value = re.sub(r"\s+", " ", str(value)).strip(" .,;:\n\t").lower()

    if not normalized_value:
        return

    values = features.setdefault(key, [])

    if normalized_value not in {item.lower() for item in values}:
        values.append(value)


# =========================
# SCORE
# =========================

def calculate_block_score(block_type: str, text: str, features: dict) -> float:
    score = min(len(text) / 400, 3)

    semantic_count = sum(
        len(value)
        for key, value in features.items()
        if isinstance(value, list)
    )

    score += min(semantic_count * 0.35, 3)

    type_weights = {
        "title": 2.5,
        "table": 2.0,
        "image_ocr": 1.5,
        "ocr_block": 1.5,
        "paragraph": 1.0,
        "list_item": 0.8,
    }

    score += type_weights.get(block_type, 0)

    return round(score, 3)


# =========================
# TEXTO FINAL / SELECCION
# =========================

def build_selected_llm_text(blocks: list[dict]) -> tuple[list[dict], str]:
    sorted_blocks = sorted(
        blocks,
        key=lambda item: (
            item.get("features", {}).get("score", 0),
            -(item.get("order") or 0)
        ),
        reverse=True
    )

    selected_blocks = []
    current_chars = 0

    for block in sorted_blocks:
        text_length = len(block.get("text", ""))

        if len(selected_blocks) >= MAX_LLM_BLOCKS:
            break

        if current_chars + text_length > MAX_LLM_CHARS:
            continue

        selected_blocks.append(block)
        current_chars += text_length

    selected_blocks.sort(key=lambda item: item.get("order") or 0)

    parts = []
    global_features = build_global_features(selected_blocks)
    global_lines = format_features_for_llm(global_features)

    if global_lines:
        parts.append(
            "FEATURES GENERALES\n" + "\n".join(global_lines)
        )

    for block in selected_blocks:
        section = [f"TIPO: {block['type']}"]
        local_features = {
            key: value
            for key, value in block.get("features", {}).items()
            if key in LOCAL_FEATURE_KEYS
        }
        feature_lines = format_features_for_llm(local_features)

        if feature_lines:
            section.extend(feature_lines)

        section.append("TEXTO:\n" + truncate_text(block["text"]))
        parts.append("\n".join(section))

    return selected_blocks, "\n\n====================\n\n".join(parts).strip()


def build_global_features(blocks: list[dict]) -> dict:
    global_features = {}

    for block in blocks:
        for key, value in block.get("features", {}).items():
            if key in GLOBAL_FEATURE_KEYS and isinstance(value, list):
                for item in value:
                    add_unique(global_features, key, item)

    return global_features


def format_features_for_llm(features: dict) -> list[str]:
    lines = []

    for key, value in features.items():
        if key in INTERNAL_FEATURE_KEYS:
            continue

        if isinstance(value, list) and value:
            label = key.replace("_", " ").upper()
            lines.append(f"{label}: {', '.join(value)}")

    return lines


def truncate_text(text: str, max_chars: int = MAX_BLOCK_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


# =========================
# CALIDAD
# =========================

def build_normalization_quality(
    original_blocks: list[dict],
    normalized_blocks: list[dict],
    normalized_text: str,
    llm_text: str
) -> dict:
    original_text_length = sum(
        len(block.get("text", "").strip())
        for block in original_blocks
    )

    normalized_text_length = len(normalized_text)

    return {
        "empty": normalized_text_length == 0,
        "original_block_count": len(original_blocks),
        "normalized_block_count": len(normalized_blocks),
        "removed_block_count": len(original_blocks) - len(normalized_blocks),
        "selected_block_count": len(normalized_blocks),
        "original_text_length": original_text_length,
        "normalized_text_length": normalized_text_length,
        "llm_text_length": len(llm_text),
        "reduction_ratio": calculate_reduction_ratio(
            original_text_length,
            normalized_text_length
        ),
    }


def calculate_reduction_ratio(original_length: int, normalized_length: int) -> float:
    if original_length == 0:
        return 0.0

    return round(1 - (normalized_length / original_length), 4)
