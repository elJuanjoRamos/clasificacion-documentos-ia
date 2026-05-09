import xml.etree.ElementTree as ET

from .utils import clean_text, create_block, build_document_output


def main_xml(ext, path):
    blocks = []
    error = None
    supported = False
    extraction_method = None

    try:
        blocks = get_xml_text(path)
        supported = True
        extraction_method = "xml_structured"

    except Exception as e:
        error = str(e)

    return build_document_output(
        file_type="xml",
        extension=ext,
        supported=supported,
        extraction_method=extraction_method,
        blocks=blocks,
        error=error,
        file_path=path,
        metadata={},
        type_metadata_key="xml_metadata"
    )

def get_xml_text(file_path: str) -> list[dict]:
    tree = ET.parse(file_path)

    root = tree.getroot()

    blocks = []

    process_xml_element(
        element=root,
        blocks=blocks,
        path=clean_xml_tag(root.tag),
        level=0
    )

    return blocks


def process_xml_element(
    element,
    blocks: list[dict],
    path: str,
    level: int
):
    tag = clean_xml_tag(element.tag)

    attributes = clean_xml_attributes(
        element.attrib
    )

    text = clean_text(element.text)

    block_text = build_xml_text(
        text=text,
        attributes=attributes
    )

    if block_text:
        blocks.append(create_block(
            order=len(blocks) + 1,
            block_type="xml_element",
            text=block_text,
            source="xml_element",
            metadata={
                "tag": tag,
                "path": path,
                "level": level,
                "attributes": attributes
            }
        ))

    for child in list(element):
        child_tag = clean_xml_tag(child.tag)

        process_xml_element(
            element=child,
            blocks=blocks,
            path=f"{path}/{child_tag}",
            level=level + 1
        )


def build_xml_text(
    text: str,
    attributes: dict
) -> str:

    parts = []

    if attributes:
        parts.append(
            " ".join(
                f"{key}: {value}"
                for key, value in attributes.items()
            )
        )

    if text:
        parts.append(text)

    return ". ".join(parts).strip()


def clean_xml_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]

    return tag


def clean_xml_attributes(attributes: dict) -> dict:
    cleaned = {}

    for key, value in attributes.items():
        key = clean_xml_tag(key)
        value = clean_text(value)

        if value:
            cleaned[key] = value

    return cleaned


def get_total_text_length(blocks: list[dict]) -> int:
    return sum(
        len(block.get("text", "").strip())
        for block in blocks
    )