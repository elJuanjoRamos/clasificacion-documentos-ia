import os
import re
import cv2
import numpy as np
import pytesseract


DEFAULT_LANG = "eng"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =========================
# Configuración OCR
# =========================

MAIN_PSMS = [
    {"psm": 6, "mode": "single_uniform_block"},
    {"psm": 4, "mode": "single_column"},
    {"psm": 11, "mode": "sparse_text"},
]

DEFAULT_ANGLES = [0, 90, 180, 270]

UPSCALE_MIN_WIDTH = 1600
UPSCALE_FACTOR = 1.5

MIN_ACCEPTED_OCR_SCORE = 1.20
EARLY_STOP_OCR_SCORE = 1.50

MIN_ACCEPTED_TEXT_LENGTH = 20
MIN_ACCEPTED_BLOCK_TEXT_LENGTH = 4
MIN_ACCEPTED_BLOCK_SCORE = 1.00


# =========================
# Lectura
# =========================

def read_image(image_path: str):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    return img


# =========================
# Análisis y preprocesado
# =========================

def analyze_image_quality(img) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sample = gray.flatten()[::100]
    unique_values = len(set(sample))

    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    return {
        "contrast": round(contrast, 3),
        "sharpness": round(sharpness, 3),
        "is_binary": unique_values < 20,
        "is_color": saturation > 35,
        "is_blurry": sharpness < 80,
        "low_contrast": contrast < 45,
    }


def upscale_image(img, scale: float = UPSCALE_FACTOR):
    h, w = img.shape[:2]

    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC
    )


def sharpen_image(gray):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(gray, -1, kernel)


def apply_preprocessing(img):
    def info(selected: str, upscaled, quality) -> dict:
        return {
            "selected": selected,
            "upscaled": upscaled,
            "quality": quality,
        }

    # REESCALADO
    h, w = img.shape[:2]
    upscaled = False

    if w < UPSCALE_MIN_WIDTH:
        img = upscale_image(img)
        upscaled = True

    quality = analyze_image_quality(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if quality["is_binary"]:
        return gray, info("already_binary", upscaled, quality)

    if quality["is_color"]:
        return cv2.equalizeHist(gray), info("color_to_gray_equalized", upscaled, quality)

    if quality["is_blurry"]:
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return sharpen_image(denoised), info("denoise_sharpen", upscaled, quality)

    if quality["low_contrast"]:
        processed = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11
        )
        return processed, info("adaptive_threshold",upscaled, quality)

    return sharpen_image(gray), info("gray_sharpen", upscaled, quality)


# =========================
# Rotación
# =========================

def rotate_image(img, angle: int):
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)

    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


# =========================
# Limpieza OCR
# =========================

def is_repetitive_noise(line: str) -> bool:
    compact = re.sub(r"\s+", "", line.lower())

    if len(compact) < 15:
        return False

    alpha_count = sum(c.isalpha() for c in compact)
    unique_count = len(set(compact))

    if alpha_count / max(len(compact), 1) > 0.85 and unique_count <= 6:
        return True

    if len(compact) > 35 and " " not in line.strip() and unique_count <= 10:
        return True

    return False


def is_noise_line(line: str) -> bool:
    line = line.strip()

    if not line:
        return True

    if len(line) <= 2:
        return True

    if is_repetitive_noise(line):
        return True

    total = len(line)
    alnum = sum(c.isalnum() for c in line)
    symbol_ratio = 1 - (alnum / max(total, 1))

    if symbol_ratio > 0.70:
        return True

    letters = [c.lower() for c in line if c.isalpha()]

    if len(letters) > 12:
        vowels = sum(c in "aeiou" for c in letters)

        if vowels / max(len(letters), 1) < 0.12:
            return True

    return False


def clean_ocr_text(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []

    for line in str(text).splitlines():
        line = line.strip()
        line = re.sub(r"\s+", " ", line)

        if is_noise_line(line):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# =========================
# OCR estructurado
# =========================

def run_ocr_data(img, lang: str = DEFAULT_LANG, psm: int = 6) -> list[dict]:
    config = f"--psm {psm}"

    data = pytesseract.image_to_data(
        img,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    lines = {}

    for index, word in enumerate(data.get("text", [])):
        text = clean_ocr_text(word)

        if not text:
            continue

        confidence = safe_float(
            data.get("conf", [])[index],
            default=-1
        )

        if confidence < 0:
            continue

        line_key = (
            data["block_num"][index],
            data["par_num"][index],
            data["line_num"][index]
        )

        left = data["left"][index]
        top = data["top"][index]
        right = left + data["width"][index]
        bottom = top + data["height"][index]

        if line_key not in lines:
            lines[line_key] = {
                "texts": [],
                "confidences": [],
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
            }

        lines[line_key]["texts"].append(text)
        lines[line_key]["confidences"].append(confidence)

        lines[line_key]["left"] = min(lines[line_key]["left"], left)
        lines[line_key]["top"] = min(lines[line_key]["top"], top)
        lines[line_key]["right"] = max(lines[line_key]["right"], right)
        lines[line_key]["bottom"] = max(lines[line_key]["bottom"], bottom)

    results = []

    for line in lines.values():
        line_text = clean_ocr_text(" ".join(line["texts"]))

        if not line_text:
            continue

        confidence = sum(line["confidences"]) / max(len(line["confidences"]), 1)

        results.append({
            "text": line_text,
            "confidence": round(confidence, 3),
            "bbox": {
                "left": line["left"],
                "top": line["top"],
                "right": line["right"],
                "bottom": line["bottom"],
            }
        })

    return results


# =========================
# Bloques OCR
# =========================

def build_ocr_blocks(lines: list[dict]) -> list[dict]:
    raw_blocks = []

    for line in lines:
        text = clean_ocr_text(line.get("text", ""))

        if not text:
            continue

        raw_blocks.append({
            "order": len(raw_blocks) + 1,
            "type": "ocr_line",
            "text": text,
            "source": "ocr_line",
            "metadata": {
                "confidence": line.get("confidence"),
                "bbox": line.get("bbox")
            }
        })

    merged_blocks = merge_related_ocr_blocks(raw_blocks)

    return [
        evaluate_ocr(
            blocks=[block],
            min_length=MIN_ACCEPTED_BLOCK_TEXT_LENGTH,
            min_score=MIN_ACCEPTED_BLOCK_SCORE,
            global_result=False
        )
        for block in merged_blocks
    ]


def merge_related_ocr_blocks(blocks: list[dict]) -> list[dict]:
    merged = []
    current = None

    for block in blocks:
        if current is None:
            current = block.copy()
            continue

        if should_merge_ocr_blocks(current, block):
            current["text"] = f'{current["text"].rstrip()} {block["text"].lstrip()}'

            current_metadata = current.get("metadata", {})
            next_metadata = block.get("metadata", {})

            current_metadata["confidence"] = average_values([
                current_metadata.get("confidence"),
                next_metadata.get("confidence")
            ])

            current["metadata"] = current_metadata

        else:
            merged.append(current)
            current = block.copy()

    if current:
        merged.append(current)

    for index, block in enumerate(merged, start=1):
        block["order"] = index
        block["type"] = "ocr_block"
        block["source"] = "ocr_block"

    return merged


def should_merge_ocr_blocks(current: dict, next_block: dict) -> bool:
    current_text = current.get("text", "").strip()
    next_text = next_block.get("text", "").strip()

    if not current_text or not next_text:
        return False

    strong_endings = (".", ":", ";", "?", "!", "|")

    continuation_words = {
        "y", "e", "o", "u",
        "de", "del", "la", "el", "los", "las",
        "en", "para", "con", "por", "a",
        "and", "or", "not"
    }

    last_word = current_text.split()[-1].lower()

    if last_word in continuation_words:
        return True

    return (
        len(current_text) <= 90
        and len(next_text) <= 90
        and not current_text.endswith(strong_endings)
    )



# =========================
# Métricas globales OCR
# =========================

def build_combined_text(blocks: list[dict], only_accepted: bool = True) -> str:
    selected_blocks = []

    for block in blocks:
        metadata = block.get("metadata", {}) or {}

        if only_accepted and not metadata.get("accepted"):
            continue

        text = block.get("text", "").strip()

        if text:
            selected_blocks.append(text)

    return "\n".join(selected_blocks).strip()


def evaluate_ocr(
    blocks: list[dict],
    min_length: int,
    min_score: float,
    global_result: bool = False
):
    target_blocks = blocks

    if global_result:
        target_blocks = [
            block for block in blocks
            if block.get("metadata", {}).get("accepted")
        ]

    combined_text = build_combined_text(
        target_blocks,
        only_accepted=False
    )

    avg_confidence = calculate_average_confidence(target_blocks)

    score = calculate_score(global_result, target_blocks, combined_text, avg_confidence)

    accepted, rejection_reason = evaluate_acceptance(
        text=combined_text,
        score=score,
        min_length=min_length,
        min_score=min_score
    )

    if global_result and not target_blocks:
        accepted = False
        rejection_reason = "no_accepted_blocks"

    metrics = {
        "score": score,
        "quality": classify_score(score),
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "text_length": len(combined_text),
        "avg_confidence": avg_confidence,
        "block_count": len(blocks)
    }

    if global_result:
        metrics.update({
            "accepted_block_count": len(target_blocks),
            "rejected_block_count": len(blocks) - len(target_blocks)
        })

        return metrics

    block = blocks[0]
    metadata = block.get("metadata", {}) or {}

    metadata.update({
        "score": metrics["score"],
        "quality": metrics["quality"],
        "accepted": metrics["accepted"],
        "rejection_reason": metrics["rejection_reason"],
        "text_length": metrics["text_length"],
        "confidence": metrics["avg_confidence"],
    })

    block["metadata"] = metadata

    return block



def evaluate_acceptance(
    text: str,
    score: float,
    min_length: int,
    min_score: float
) -> tuple[bool, str | None]:

    if not text.strip():
        return False, "empty_text"

    if len(text.strip()) < min_length:
        return False, "text_too_short"

    if score < min_score:
        return False, "low_score"

    return True, None



def calculate_score(global_result: bool, blocks: list[dict], combined_text, avg_confidence) -> float:

    if global_result:
        if not blocks:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for block in blocks:
            metadata = block.get("metadata", {}) or {}

            score = safe_float(metadata.get("score"), default=0.0)
            weight = max(len(block.get("text", "").strip()), 1)

            weighted_sum += score * weight
            total_weight += weight

        return round(weighted_sum / max(total_weight, 1), 3)
    else:
        text_score = score_ocr_text(combined_text)
        confidence_score = round((max(min(avg_confidence, 100), 0) / 100) * 2, 3)

        return round((text_score * 0.6) + (confidence_score * 0.4),3)

def calculate_average_confidence(blocks: list[dict]) -> float:
    if not blocks:
        return 0.0

    values = [
        safe_float(block.get("metadata", {}).get("confidence"), default=0.0)
        for block in blocks
    ]

    return round(sum(values) / max(len(values), 1), 3)


def classify_score(score: float) -> str:
    if score >= 1.60:
        return "good"

    if score >= 1.20:
        return "medium"

    if score > 0:
        return "poor"

    return "empty"


def score_ocr_text(text: str) -> float:
    text = text.strip()

    if not text:
        return 0.0

    total = len(text)
    alnum = sum(c.isalnum() for c in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    avg_line_len = sum(len(line) for line in lines) / max(len(lines), 1)
    alnum_ratio = alnum / max(total, 1)

    score = 0.0
    score += alnum_ratio
    score += min(avg_line_len / 12, 1)

    return round(score, 3)


# =========================
# Estrategias OCR
# =========================

def extract_best_text(
    img,
    lang: str = DEFAULT_LANG,
    angles: list[int] | None = None,
    psms: list[dict] | None = None,
    keep_candidates: bool = False,
) -> dict:

    angles = angles or DEFAULT_ANGLES
    psms = psms or MAIN_PSMS

    candidates = []
    best = None

    for angle in angles:
        rotated = rotate_image(img, angle)

        for cfg in psms:
            lines = run_ocr_data(
                rotated,
                lang=lang,
                psm=cfg["psm"]
            )

            blocks = build_ocr_blocks(lines)
            metrics = evaluate_ocr(
                blocks=blocks,
                min_length=MIN_ACCEPTED_TEXT_LENGTH,
                min_score=MIN_ACCEPTED_OCR_SCORE,
                global_result=True
            )
            combined_text = build_combined_text(
                blocks,
                only_accepted=True
            )

            candidate = {
                "angle": angle,
                "psm": cfg["psm"],
                "mode": cfg["mode"],
                "combined_text": combined_text,
                "blocks": blocks,
                "ocr": metrics,
            }

            candidates.append(candidate)

            if best is None or metrics["score"] > best["ocr"]["score"]:
                best = candidate

            if metrics["score"] >= EARLY_STOP_OCR_SCORE:
                result = {"best": candidate}

                if keep_candidates:
                    result["candidates"] = candidates

                return result

    result = {"best": best}

    if keep_candidates:
        result["candidates"] = candidates

    return result


# =========================
# OCR imagen
# =========================

def extract_ocr_image(
    image_path: str,
    lang: str = DEFAULT_LANG,
    keep_candidates: bool = False,
) -> dict:

    original = read_image(image_path)

    processed, preprocessing_info = apply_preprocessing(original)

    main_result = extract_best_text(
        processed,
        lang=lang,
        angles=DEFAULT_ANGLES,
        psms=MAIN_PSMS,
        keep_candidates=keep_candidates,
    )

    best = main_result["best"]

    ocr = {
        **best["ocr"],
        "psm": best["psm"],
        "mode": best["mode"],
        "angle": best["angle"],
    }

    result = {
        "file_path": str(image_path),
        "type": "image",
        "combined_text": best["combined_text"] if ocr["accepted"] else "",
        "blocks": best["blocks"],
        "ocr": ocr,
        "preprocessing": {
            "selected": preprocessing_info.get("selected"),
            "upscaled": preprocessing_info.get("upscaled", False),
            "quality": preprocessing_info.get("quality"),
        },
    }

    if keep_candidates:
        result["candidates"] = main_result.get("candidates", [])

    return result


# =========================
# Utilidades
# =========================

def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def average_values(values: list) -> float:
    numeric_values = [
        safe_float(value, default=None)
        for value in values
        if value is not None
    ]

    numeric_values = [
        value for value in numeric_values
        if value is not None
    ]

    if not numeric_values:
        return 0.0

    return round(
        sum(numeric_values) / len(numeric_values),
        3
    )