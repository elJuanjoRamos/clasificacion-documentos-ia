import re

from .utils import clean_text, create_block, build_document_output


def main_txt(ext, path):
    blocks = []
    error = None
    supported = False
    extraction_method = None

    try:
        blocks = get_txt_text(path)
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
        metadata={},
        type_metadata_key="txt_metadata"
    )

def get_txt_text(file_path: str) -> list[dict]:
    content = read_txt_file(file_path)

    sections = split_text_sections(content)

    blocks = []

    for section in sections:
        text = clean_text(section)

        if not text:
            continue

        block_type = (
            "list_item"
            if is_list_item(text)
            else "paragraph"
        )

        if block_type == "list_item":
            text = clean_list_item(text)

        blocks.append(create_block(
            order=len(blocks) + 1,
            block_type=block_type,
            text=text,
            source="txt_section"
        ))

    return blocks


def read_txt_file(file_path: str) -> str:
    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252"
    ]

    last_error = None

    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                return file.read()

        except UnicodeDecodeError as e:
            last_error = e

    raise ValueError(
        "No se pudo leer el archivo. "
        f"Último error: {last_error}"
    )


def split_text_sections(content: str) -> list[str]:
    return re.split(r"\n\s*\n+", content)


def is_list_item(text: str) -> bool:
    patterns = [
        r"^[-*•]\s+",
        r"^\d+\.\s+",
        r"^[a-zA-Z]\)\s+"
    ]

    return any(
        re.match(pattern, text)
        for pattern in patterns
    )


def clean_list_item(text: str) -> str:
    return re.sub(
        r"^([-*•]|\d+\.|[a-zA-Z]\))\s+",
        "",
        text
    ).strip()


def get_total_text_length(blocks: list[dict]) -> int:
    return sum(
        len(block.get("text", "").strip())
        for block in blocks
    )


