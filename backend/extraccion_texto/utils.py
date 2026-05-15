import os
import re
import shutil
import json
from pathlib import Path

MIN_BLOCK_TEXT_LENGTH = 5
MIN_BLOCK_SCORE = 0.30


def clean_temp_files(file_paths: list[str]) -> None:
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass


def clean_temp_dir(temp_path):
    try:
        workdir = os.getcwd()
        full_path = os.path.abspath(os.path.join(workdir, temp_path))

        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        elif os.path.exists(full_path):
            os.remove(full_path)

    except Exception:
        pass


def clean_text(value) -> str:
    if value is None:
        return ""

    return "\n".join(
        " ".join(line.split())
        for line in str(value).splitlines()
        if line.strip()
    ).strip()


def get_total_text_length(blocks: list[dict]) -> int:
    return sum(
        len(block.get("text", "").strip())
        for block in blocks
    )


def normalize_score(score, max_value: float = 1.0) -> float:
    try:
        score = float(score)
    except (TypeError, ValueError):
        return 0.0

    if max_value != 1.0:
        score = score / max_value

    return round(max(0.0, min(score, 1.0)), 3)


def score_to_confidence(score: float) -> str:
    score = normalize_score(score)

    if score >= 0.75:
        return "high"

    if score >= 0.50:
        return "medium"

    if score > 0:
        return "low"

    return "none"


def score_to_quality(score: float) -> str:
    score = normalize_score(score)

    if score >= 0.75:
        return "good"

    if score >= 0.50:
        return "usable"

    if score >= 0.35:
        return "weak"

    if score > 0:
        return "poor"

    return "empty"


def normalize_quality_result(
    text: str,
    score=None,
    confidence=None,
    quality=None,
    accepted=None,
    rejection_reason=None,
    metrics: dict | None = None
) -> dict:
    text = clean_text(text)
    metrics = metrics or {}

    normalized_score = normalize_score(score)

    if confidence not in {"none", "low", "medium", "high"}:
        confidence = score_to_confidence(normalized_score)

    if quality not in {"empty", "poor", "weak", "usable", "good"}:
        quality = score_to_quality(normalized_score)

    if accepted is None:
        accepted = (
            bool(text)
            and len(text) >= MIN_BLOCK_TEXT_LENGTH
            and normalized_score >= MIN_BLOCK_SCORE
            and not is_probably_noise(text)
        )

    if rejection_reason is None:
        if not text:
            rejection_reason = "empty_text"
        elif len(text) < MIN_BLOCK_TEXT_LENGTH:
            rejection_reason = "text_too_short"
        elif is_probably_noise(text):
            rejection_reason = "probable_noise"
        elif normalized_score < MIN_BLOCK_SCORE:
            rejection_reason = "low_score"

    return {
        "score": normalized_score,
        "confidence": confidence,
        "quality": quality,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "metrics": metrics
    }


def score_text_block(
    text: str,
    block_type: str = "paragraph",
    source: str = "",
    metadata: dict | None = None
) -> dict:
    text = clean_text(text)
    metadata = metadata or {}

    if not text:
        return normalize_quality_result(
            text="",
            score=0.0,
            accepted=False,
            rejection_reason="empty_text",
            metrics={
                "text_length": 0,
                "word_count": 0,
                "line_count": 0,
                "alnum_ratio": 0.0,
                "symbol_ratio": 0.0,
                "avg_line_length": 0.0,
                "digit_ratio": 0.0,
            }
        )

    text_length = len(text)
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    alnum_count = sum(char.isalnum() for char in text)
    digit_count = sum(char.isdigit() for char in text)
    symbol_count = sum(
        not char.isalnum() and not char.isspace()
        for char in text
    )

    word_count = len(words)
    line_count = len(lines)

    alnum_ratio = alnum_count / max(text_length, 1)
    digit_ratio = digit_count / max(text_length, 1)
    symbol_ratio = symbol_count / max(text_length, 1)

    avg_line_length = (
        sum(len(line) for line in lines) / max(line_count, 1)
    )

    score = 0.0
    score += min(text_length / 500, 1.0) * 0.25
    score += min(word_count / 80, 1.0) * 0.25
    score += alnum_ratio * 0.25
    score += min(avg_line_length / 80, 1.0) * 0.15

    if symbol_ratio > 0.35:
        score -= 0.15

    if text_length < MIN_BLOCK_TEXT_LENGTH:
        score -= 0.25

    if is_probably_noise(text):
        score -= 0.30

    score += get_block_type_bonus(
        block_type=block_type,
        text=text,
        metadata=metadata
    )
    # BONUS DE TEXTO LIMPIO
    if text_length >= 15 and word_count >= 3 and alnum_ratio >= 0.65 and symbol_ratio <= 0.20:
        score += 0.12

    score = normalize_score(score)

    metrics = {
        "text_length": text_length,
        "word_count": word_count,
        "line_count": line_count,
        "alnum_ratio": round(alnum_ratio, 3),
        "symbol_ratio": round(symbol_ratio, 3),
        "avg_line_length": round(avg_line_length, 3),
        "digit_ratio": round(digit_ratio, 3),
    }
    return normalize_quality_result(
        text=text,
        score=score,
        metrics=metrics
    )


def get_block_type_bonus(
    block_type: str,
    text: str,
    metadata: dict
) -> float:
    block_type = block_type or ""

    if block_type == "title":
        return 0.08 if 3 <= len(text.split()) <= 20 else 0.02

    if block_type == "paragraph":
        return 0.05

    if block_type == "list_item":
        return 0.04

    if block_type == "table":
        row_count = metadata.get("row_count", 0) or 0
        column_count = metadata.get("column_count", 0) or 0

        if row_count >= 2 and column_count >= 2:
            return 0.10

        return 0.03

    if block_type == "xml_element":
        return 0.06

    if block_type == "image_ocr":
        return 0.06

    return 0.0


def is_probably_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text.lower())

    if not compact:
        return True

    if len(compact) <= 2:
        return True

    unique_chars = len(set(compact))
    alpha_count = sum(char.isalpha() for char in compact)
    digit_count = sum(char.isdigit() for char in compact)

    if len(compact) > 20 and unique_chars <= 5:
        return True

    if alpha_count == 0 and digit_count == 0:
        return True

    symbol_count = sum(
        not char.isalnum()
        for char in compact
    )

    if symbol_count / max(len(compact), 1) > 0.60:
        return True

    return False


def create_block(
    order: int,
    block_type: str,
    text: str,
    source: str,
    metadata: dict | None = None,
    precomputed_quality: dict | None = None
) -> dict:
    text = clean_text(text)
    metadata = metadata or {}

    if precomputed_quality:
        quality = normalize_quality_result(
            text=text,
            score=precomputed_quality.get("score"),
            confidence=precomputed_quality.get("confidence"),
            quality=precomputed_quality.get("quality"),
            accepted=precomputed_quality.get("accepted"),
            rejection_reason=precomputed_quality.get("rejection_reason"),
            metrics=precomputed_quality.get("metrics", {})
        )
    else:
        quality = score_text_block(
            text=text,
            block_type=block_type,
            source=source,
            metadata=metadata
        )

    return {
        "order": order,
        "type": block_type,
        "text": text,
        "source": source,
        "score": quality["score"],
        "confidence": quality["confidence"],
        "quality": quality["quality"],
        "accepted": quality["accepted"],
        "rejection_reason": quality["rejection_reason"],
        "metrics": quality["metrics"],
        "metadata": metadata
    }


def build_document_output(
    file_type: str,
    extension: str,
    supported: bool,
    extraction_method: str | None,
    blocks: list[dict],
    error: str | None = None,
    file_path: str | None = None,
    metadata: dict | None = None,
    type_metadata_key: str | None = None
) -> dict:
    metadata = metadata or {}

    accepted_blocks = [
        block for block in blocks
        if block.get("accepted", True)
    ]

    combined_text = "\n\n".join(
        block.get("text", "").strip()
        for block in accepted_blocks
        if block.get("text", "").strip()
    )

    document_score = calculate_document_score(accepted_blocks)

    result = {
        "file_type": file_type,
        "extension": extension,
        "supported": supported,
        "file_path": str(file_path),
        "extraction_method": extraction_method,
        "combined_text": combined_text,
        "blocks": blocks,
        "accepted_blocks_count": len(accepted_blocks),
        "total_blocks_count": len(blocks),
        "text_length": len(combined_text),
        "document_score": document_score,
        "document_confidence": score_to_confidence(document_score),
        "document_quality": score_to_quality(document_score),
        "error": error
    }

    if type_metadata_key:
        result[type_metadata_key] = metadata
    else:
        result["metadata"] = metadata

    return result


def calculate_document_score(blocks: list[dict]) -> float:
    if not blocks:
        return 0.0

    weighted_scores = []
    weights = []

    for block in blocks:
        text_length = len(block.get("text", "").strip())
        weight = max(1, min(text_length, 1000))

        weights.append(weight)
        weighted_scores.append(block.get("score", 0.0) * weight)

    return round(
        sum(weighted_scores) / max(sum(weights), 1),
        3
    )


##################################################
# Configuraciones
##################################################


CONFIG_PATH = Path("configuracion.json")

DEFAULT_EXTRACTION_CONFIG = {
    "max_caracteres_pdf": 15000,
    "max_caracteres_word": 8000,
    "max_caracteres_excel": 6000,
    "max_caracteres_txt": 4000,
    "max_caracteres_xml": 5000,
    "aplicar_ocr_imagenes_embebidas": True,
    "extraer_tablas_word": True,
    "extraer_tablas_pdf": False,
    "modo_simulacion": True
}

def load_extraction_config() -> dict:
    if not CONFIG_PATH.exists():
        return DEFAULT_EXTRACTION_CONFIG.copy()

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            config = json.load(file)

        return {**DEFAULT_EXTRACTION_CONFIG, **config}

    except Exception:
        return DEFAULT_EXTRACTION_CONFIG.copy()


def limit_blocks_by_chars(
    blocks: list[dict],
    max_chars: int
) -> list[dict]:
    if not max_chars or max_chars <= 0:
        return blocks

    limited_blocks = []
    total_chars = 0

    for block in blocks:
        text = block.get("text", "")

        if total_chars >= max_chars:
            break

        remaining = max_chars - total_chars

        if len(text) > remaining:
            block = block.copy()
            block["text"] = text[:remaining].rstrip()
            limited_blocks.append(block)
            break

        limited_blocks.append(block)
        total_chars += len(text)

    return limited_blocks

def save_extraction_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4
        )