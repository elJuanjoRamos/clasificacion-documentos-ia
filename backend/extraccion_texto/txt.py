import re

from .utils import (
    clean_text,
    create_block,
    build_document_output,
    load_extraction_config,
    limit_blocks_by_chars
)


ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")

LIST_ITEM_PATTERN = re.compile(
    r"^([-*•]|\d+\.|[a-zA-Z]\))\s+"
)


def main_txt(ext, path):
    config = load_extraction_config()

    blocks = []
    error = None
    supported = False
    extraction_method = None

    try:
        blocks = get_txt_text(path)

        blocks = limit_blocks_by_chars(
            blocks=blocks,
            max_chars=config.get("max_caracteres_txt", 4000)
        )

        supported = True
        extraction_method = "txt_native"

    except Exception as e:
        error = str(e)

    return build_document_output(
        file_type="txt",
        extension=ext,
        supported=supported,
        extraction_method=extraction_method,
        blocks=blocks,
        error=error,
        file_path=path,
        metadata={
            "config": {
                "max_caracteres_txt": config.get("max_caracteres_txt", 4000)
            }
        },
        type_metadata_key="txt_metadata"
    )


def get_txt_text(file_path: str) -> list[dict]:
    content = read_txt_file(file_path)
    sections = re.split(r"\n\s*\n+", content)

    blocks = []

    for section in sections:
        text = clean_text(section)

        if not text:
            continue

        is_list = bool(LIST_ITEM_PATTERN.match(text))

        blocks.append(create_block(
            order=len(blocks) + 1,
            block_type="list_item" if is_list else "paragraph",
            text=LIST_ITEM_PATTERN.sub("", text).strip() if is_list else text,
            source="txt_section"
        ))

    return blocks


def read_txt_file(file_path: str) -> str:
    last_error = None

    for encoding in ENCODINGS:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()

        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        "No se pudo leer el archivo TXT. "
        f"Último error: {last_error}"
    )


def get_total_text_length(blocks: list[dict]) -> int:
    return sum(
        len(block.get("text", "").strip())
        for block in blocks
    )