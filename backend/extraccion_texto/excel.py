import pandas as pd

from .utils import clean_text, create_block, build_document_output


def main_excel(ext, path):
    blocks = []
    error = None
    supported = False
    extraction_method = None

    try:
        blocks = get_excel_text(path)
        supported = True
        extraction_method = "excel_structured"

    except Exception as e:
        error = str(e)

    return build_document_output(
        file_type="excel",
        extension=ext,
        supported=supported,
        extraction_method=extraction_method,
        blocks=blocks,
        error=error,
        file_path=path,
        metadata={},
        type_metadata_key="excel_metadata"
    )

def get_excel_text(file_path: str) -> list[dict]:
    excel_file = pd.ExcelFile(file_path)

    blocks = []

    for sheet_index, sheet_name in enumerate(excel_file.sheet_names, start=1):
        data = excel_file.parse(
            sheet_name=sheet_name,
            header=None,
            dtype=str
        )

        data = clean_excel_dataframe(data)

        if data.empty:
            continue

        table_text = dataframe_to_text(data)

        if not table_text:
            continue

        blocks.append(create_block(
            order=len(blocks) + 1,
            block_type="table",
            text=table_text,
            source="excel_sheet",
            metadata={
                "sheet_index": sheet_index,
                "sheet_name": sheet_name,
                "row_count": int(data.shape[0]),
                "column_count": int(data.shape[1])
            }
        ))

    return blocks


def clean_excel_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data = data.dropna(how="all")
    data = data.dropna(axis=1, how="all")
    data = data.fillna("")

    return data


def dataframe_to_text(data: pd.DataFrame) -> str:
    lines = []

    for _, row in data.iterrows():
        row_values = [
            clean_text(value)
            for value in row.tolist()
        ]

        if any(row_values):
            lines.append(" | ".join(row_values))

    return "\n".join(lines).strip()


def get_total_text_length(blocks: list[dict]) -> int:
    return sum(
        len(block.get("text", "").strip())
        for block in blocks
    )


