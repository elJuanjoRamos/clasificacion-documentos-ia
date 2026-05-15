import xml.etree.ElementTree as ET

from .utils import (
    clean_text,
    create_block,
    build_document_output,
    load_extraction_config,
    limit_blocks_by_chars
)


class XMLExtractor:
    def __init__(self, ext: str, path: str):
        self.ext = ext
        self.path = path
        self.blocks = []
        self.error = None
        self.supported = False
        self.extraction_method = None
        self.config = load_extraction_config()

    def extract(self) -> dict:
        try:
            self.blocks = self.get_xml_text()

            self.blocks = limit_blocks_by_chars(
                blocks=self.blocks,
                max_chars=self.config.get("max_caracteres_xml", 5000)
            )

            self.supported = True
            self.extraction_method = "xml_structured"

        except Exception as e:
            self.error = str(e)

        return build_document_output(
            file_type="xml",
            extension=self.ext,
            supported=self.supported,
            extraction_method=self.extraction_method,
            blocks=self.blocks,
            error=self.error,
            file_path=self.path,
            metadata={
                "config": {
                    "max_caracteres_xml": self.config.get("max_caracteres_xml", 5000)
                }
            },
            type_metadata_key="xml_metadata"
        )

    def get_xml_text(self) -> list[dict]:
        tree = ET.parse(self.path)
        root = tree.getroot()

        blocks = []

        self.process_xml_element(
            element=root,
            blocks=blocks,
            path=self.clean_xml_tag(root.tag),
            level=0
        )

        return blocks

    def process_xml_element(
        self,
        element,
        blocks: list[dict],
        path: str,
        level: int
    ):
        tag = self.clean_xml_tag(element.tag)

        attributes = self.clean_xml_attributes(element.attrib)

        text = clean_text(element.text)

        block_text = self.build_xml_text(
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
            child_tag = self.clean_xml_tag(child.tag)

            self.process_xml_element(
                element=child,
                blocks=blocks,
                path=f"{path}/{child_tag}",
                level=level + 1
            )

    def build_xml_text(self, text: str, attributes: dict) -> str:
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

    def clean_xml_tag(self, tag: str) -> str:
        if "}" in tag:
            return tag.split("}", 1)[1]

        return tag

    def clean_xml_attributes(self, attributes: dict) -> dict:
        cleaned = {}

        for key, value in attributes.items():
            key = self.clean_xml_tag(key)
            value = clean_text(value)

            if value:
                cleaned[key] = value

        return cleaned

    def get_total_text_length(self) -> int:
        return sum(
            len(block.get("text", "").strip())
            for block in self.blocks
        )


def main_xml(ext, path):
    extractor = XMLExtractor(ext, path)
    return extractor.extract()