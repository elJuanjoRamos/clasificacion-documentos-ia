import os
import uuid
import fitz
import pdfplumber
import numpy as np

from pdf2image import convert_from_path

from ..ocr.ocr import extract_ocr_image

from .utils import (
    clean_text,
    create_block,
    clean_temp_dir,
    build_document_output,
    normalize_score,
    score_to_confidence,
    score_to_quality
)

TEMP_PDF_IMAGES_DIR = "temp_pdf_images"

BLANK_PAGE_WHITE_THRESHOLD = 245
BLANK_PAGE_MIN_WHITE_PERCENTAGE = 0.98

FALLBACK_RENDER_DPI = 40
FALLBACK_OCR_RENDER_DPI = 300


def main_pdf(ext, path):
    blocks = []
    error = None
    supported = False
    analysis = {}

    try:
        blocks, analysis = extract_pdf_blocks(path)

        supported = True

    except Exception as e:
        error = str(e)

    finally:
        clean_temp_dir(TEMP_PDF_IMAGES_DIR)

    return build_document_output(
        file_type="pdf",
        extension=ext,
        supported=supported,
        extraction_method="pdf_structured",
        blocks=blocks,
        error=error,
        file_path=path,
        metadata={
            "analysis": analysis
        },
        type_metadata_key="pdf_metadata"
    )


def extract_pdf_blocks(file_path: str):
    os.makedirs(TEMP_PDF_IMAGES_DIR, exist_ok=True)

    blocks = []

    analysis = {
        "page_count": 0,
        "native_text_pages": 0,
        "table_pages": 0,
        "image_pages": 0,
        "ocr_image_pages": 0,
        "ocr_fallback_pages": 0,
        "blank_pages": 0,
        "total_tables": 0,
        "total_images": 0,
        "total_native_text_length": 0
    }

    pdf_document = fitz.open(file_path)

    with pdfplumber.open(file_path) as plumber_pdf:
        analysis["page_count"] = len(plumber_pdf.pages)

        for page_number, plumber_page in enumerate(
            plumber_pdf.pages,
            start=1
        ):

            fitz_page = pdf_document[page_number - 1]

            page_blocks = []

            #
            # TEXTO NATIVO
            #
            native_text = clean_text(plumber_page.extract_text() or "")

            if native_text:
                analysis["native_text_pages"] += 1
                analysis["total_native_text_length"] += len(native_text)

                page_blocks.append(create_block(
                    order=0,
                    block_type="paragraph",
                    text=native_text,
                    source="pdf_native_text",
                    metadata={
                        "page_number": page_number
                    }
                ))

            #
            # TABLAS
            #

            table_blocks = extract_page_tables(
                page=plumber_page,
                page_number=page_number
            )

            if table_blocks:
                analysis["table_pages"] += 1
                analysis["total_tables"] += len(table_blocks)
                page_blocks.extend(table_blocks)

            #
            # IMAGENES EMBEBIDAS
            #

            image_blocks = extract_page_image_ocr_blocks(
                page=fitz_page,
                page_number=page_number
            )

            if image_blocks:
                analysis["image_pages"] += 1
                analysis["ocr_image_pages"] += 1
                analysis["total_images"] += len(image_blocks)
                page_blocks.extend(image_blocks)

            #
            # FALLBACK OCR PAGINA COMPLETA
            #

            has_content = bool(
                native_text
                or table_blocks
                or image_blocks
            )

            if not has_content:
                if is_blank_pdf_page(
                    file_path=file_path,
                    page_number=page_number
                ):
                    analysis["blank_pages"] += 1

                else:
                    fallback_blocks = extract_page_fallback_ocr(
                        page=fitz_page,
                        page_number=page_number
                    )

                    if fallback_blocks:
                        analysis["ocr_fallback_pages"] += 1
                        page_blocks.extend(fallback_blocks)

            blocks.extend(page_blocks)

    blocks.sort(key=get_pdf_block_sort_key)

    for index, block in enumerate(blocks, start=1):
        block["order"] = index

    pdf_document.close()

    return blocks, analysis


#
# TABLAS
#

def extract_page_tables(
    page,
    page_number: int
) -> list[dict]:

    blocks = []

    tables = page.extract_tables() or []

    for table_index, table in enumerate(
        tables,
        start=1
    ):

        rows = clean_table_rows(table)

        if not rows:
            continue

        blocks.append(create_block(
            order=0,
            block_type="table",
            text=table_to_text(rows),
            source="pdf_table",
            metadata={
                "page_number": page_number,
                "table_index": table_index,
                "row_count": len(rows),
                "column_count": max(
                    len(row)
                    for row in rows
                )
            }
        ))

    return blocks


#
# IMAGENES EMBEBIDAS
#
def extract_page_image_ocr_blocks(
    page,
    page_number: int
) -> list[dict]:

    blocks = []

    for image_index, image_info in enumerate(page.get_images(full=True), start=1):
        xref = image_info[0]

        try:
            image_data = page.parent.extract_image(xref)
            image_bytes = image_data["image"]
            image_extension = image_data["ext"]

            image_path = os.path.join(
                TEMP_PDF_IMAGES_DIR,
                f"pdf_p{page_number}_img_{image_index}_{uuid.uuid4().hex}.{image_extension}"
            )

            with open(image_path, "wb") as file:
                file.write(image_bytes)

            result = extract_ocr_image(image_path)

            blocks.extend(build_pdf_ocr_blocks(
                result=result,
                page_number=page_number,
                image_path=image_path,
                source="pdf_embedded_image",
                base_metadata={
                    "image_index": image_index
                }
            ))

        except Exception:
            continue

    return blocks
#
# FALLBACK OCR PAGINA COMPLETA
#
def extract_page_fallback_ocr(
    page,
    page_number: int
) -> list[dict]:

    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

        image_path = os.path.join(
            TEMP_PDF_IMAGES_DIR,
            f"pdf_page_{page_number}_{uuid.uuid4().hex}.png"
        )

        pix.save(image_path)

        result = extract_ocr_image(image_path)

        return build_pdf_ocr_blocks(
            result=result,
            page_number=page_number,
            image_path=image_path,
            source="pdf_page_ocr",
            base_metadata={
                "rendered_page": True
            }
        )

    except Exception:
        return []

#
# CONSTRUIR BLOQUES
#
def build_pdf_ocr_blocks(
    result: dict,
    page_number: int,
    image_path: str,
    source: str,
    base_metadata: dict | None = None,
    start_order: int = 1
) -> list[dict]:
    blocks = []
    base_metadata = base_metadata or {}

    for ocr_block in result.get("blocks", []):
        text = clean_text(ocr_block.get("text", ""))

        if not text:
            continue

        original_metadata = ocr_block.get("metadata", {}) or {}

        raw_ocr_score = original_metadata.get("score", 0.0)
        normalized_score = normalize_score(raw_ocr_score, max_value=2.0)
        ocr_confidence = original_metadata.get("confidence", 0.0)

        blocks.append(create_block(
            order=start_order + len(blocks),
            block_type="image_ocr",
            text=text,
            source=source,
            metadata={
                "page_number": page_number,
                "file_path": image_path,
                "ocr": {
                    "raw_score": raw_ocr_score,
                    "raw_quality": original_metadata.get("quality"),
                    "ocr_confidence": ocr_confidence,
                    "global_result": result.get("ocr", {})
                },
                "preprocessing": result.get("preprocessing", {}),
                "bbox": original_metadata.get("bbox"),
                **base_metadata
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
                    "raw_ocr_quality": original_metadata.get("quality")
                }
            }
        ))

    return blocks
#
# BLANK PAGE DETECTION
#

def is_blank_pdf_page(
    file_path: str,
    page_number: int
) -> bool:

    preview = convert_from_path(
        file_path,
        dpi=FALLBACK_RENDER_DPI,
        first_page=page_number,
        last_page=page_number
    )[0]

    image = preview.convert("L")

    pixels = np.array(image)

    white_pixels = (
        pixels > BLANK_PAGE_WHITE_THRESHOLD
    ).sum()

    total_pixels = pixels.size

    white_ratio = (
        white_pixels / total_pixels
    )

    return (
        white_ratio
        >= BLANK_PAGE_MIN_WHITE_PERCENTAGE
    )


#
# TABLAS
#

def clean_table_rows(
    table: list[list]
) -> list[list[str]]:

    rows = []

    for row in table:
        cleaned_row = [
            clean_text(cell)
            for cell in row
        ]

        if any(cleaned_row):
            rows.append(cleaned_row)

    return rows


def table_to_text(
    rows: list[list[str]]
) -> str:

    return "\n".join(
        " | ".join(row)
        for row in rows
    ).strip()


#
# SORT
#

def get_pdf_block_sort_key(
    block: dict
):

    metadata = block.get(
        "metadata",
        {}
    )

    page_number = (
        metadata.get("page_number")
        or 0
    )

    type_priority = {
        "paragraph": 1,
        "table": 2,
        "image_ocr": 3
    }

    return (
        page_number,
        type_priority.get(
            block.get("type"),
            99
        ),
        metadata.get(
            "table_index",
            0
        )
    )

