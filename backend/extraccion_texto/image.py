from ..ocr.ocr import extract_ocr_image
from .utils import (
    clean_text,
    create_block,
    build_document_output,
    normalize_score,
    score_to_confidence,
    score_to_quality
)


def main_image(ext, path):
    result = {}
    blocks = []
    error = None
    supported = False
    extraction_method = None

    try:
        result = extract_ocr_image(str(path))

        supported = True
        extraction_method = "image_ocr"

        ocr_blocks = result.get("blocks", [])

        for block in ocr_blocks:
            text = clean_text(block.get("text", ""))

            if not text:
                continue

            original_metadata = block.get("metadata", {}) or {}

            raw_ocr_score = original_metadata.get("score", 0.0)

            # El OCR usa un score aproximado de 0 a 2.
            # El estándar general del sistema usa 0 a 1.
            normalized_score = normalize_score(raw_ocr_score, max_value=2.0)

            ocr_confidence = original_metadata.get("confidence", 0.0)

            blocks.append(create_block(
                order=len(blocks) + 1,
                block_type="image_ocr",
                text=text,
                source="image_file",
                metadata={
                    "image_type": "standalone_image",
                    "file_path": str(path),
                    "bbox": original_metadata.get("bbox"),
                    "ocr": {
                        "raw_score": raw_ocr_score,
                        "raw_quality": original_metadata.get("quality"),
                        "ocr_confidence": ocr_confidence,
                    }
                },
                precomputed_quality={
                    "score": normalized_score,
                    "confidence": score_to_confidence(normalized_score),
                    "quality": score_to_quality(normalized_score),
                    "accepted": original_metadata.get("accepted", True),
                    "rejection_reason": original_metadata.get("rejection_reason"),
                    "metrics": {
                        "text_length": original_metadata.get("text_length", len(text)),
                        "ocr_confidence": ocr_confidence,
                        "raw_ocr_score": raw_ocr_score,
                    }
                }
            ))

        if not blocks:
            text = clean_text(result.get("combined_text", ""))

            if text:
                global_ocr = result.get("ocr", {}) or {}

                raw_ocr_score = global_ocr.get("score", 0.0)
                normalized_score = normalize_score(raw_ocr_score, max_value=2.0)

                blocks.append(create_block(
                    order=1,
                    block_type="image_ocr",
                    text=text,
                    source="image_file",
                    metadata={
                        "image_type": "standalone_image",
                        "file_path": str(path),
                        "bbox": None,
                        "ocr": {
                            "raw_score": raw_ocr_score,
                            "raw_quality": global_ocr.get("quality"),
                            "ocr_confidence": global_ocr.get("avg_confidence", 0.0),
                        }
                    },
                    precomputed_quality={
                        "score": normalized_score,
                        "confidence": score_to_confidence(normalized_score),
                        "quality": score_to_quality(normalized_score),
                        "accepted": global_ocr.get("accepted", True),
                        "rejection_reason": global_ocr.get("rejection_reason"),
                        "metrics": {
                            "text_length": len(text),
                            "ocr_confidence": global_ocr.get("avg_confidence", 0.0),
                            "raw_ocr_score": raw_ocr_score,
                        }
                    }
                ))

    except Exception as e:
        error = str(e)

    return build_document_output(
        file_type="image",
        extension=ext,
        supported=supported,
        extraction_method=extraction_method,
        blocks=blocks,
        error=error,
        file_path=str(path),
        metadata={
            "image_type": "standalone_image",
            "file_path": str(path),
            "ocr": result.get("ocr", {}),
            "preprocessing": result.get("preprocessing", {})
        },
        type_metadata_key="image_metadata"
    )