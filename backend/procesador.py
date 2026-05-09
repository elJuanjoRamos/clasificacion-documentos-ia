from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import os
import time

from backend.llm.llm import analyze_document_with_llm
from backend.embeddings.embeddings import (
    compare_text_with_memory,
    compare_semantic_profile_with_memory,
    decide_category_from_similarity,
    should_call_llm_from_similarity,
    build_decision_without_llm,
)
from backend.category_memory.category_memory import (
    load_category_memory,
    update_category_with_document,
    build_profile_from_existing_category,
)

from backend.extraccion_texto.word import main_word
from backend.extraccion_texto.pdf import main_pdf
from backend.extraccion_texto.excel import main_excel
from backend.extraccion_texto.txt import main_txt
from backend.extraccion_texto.xml import main_xml
from backend.extraccion_texto.image import main_image
from .normalizer.normalizer import normalize_extraction_result


MAX_FILE_WORKERS = 4


def process_documents(files: list[dict]) -> list[dict]:
    
    results = []

    with ThreadPoolExecutor(max_workers=MAX_FILE_WORKERS) as executor:
        futures = [
            executor.submit(process_single_document_before_decision, file)
            for file in files
        ]

        for future in as_completed(futures):
            results.append(future.result())

    return results #apply_optimized_semantic_flow(results)


def process_single_document_before_decision(file: dict) -> dict:
    path = Path(file["full_path"])
    ext = path.suffix.lower()

    
    total_start = time.perf_counter()

    try:
        start = time.perf_counter()
        extraction_result = extract_document(
            ext=ext,
            path=path
        )
        t_extraction = time.perf_counter() - start
        
        
        start = time.perf_counter()
        normalized_result = normalize_extraction_result(extraction_result)
        t_normalization = time.perf_counter() - start
        
        return normalized_result
        
        """
        normalized_result["file_name"] = file.get("name") or path.name
        normalized_result["full_path"] = str(path)

        normalized_result["processing_times"] = {
            "extraction": round(t_extraction, 3),
            "normalization": round(t_normalization, 3),
            "pre_llm_embeddings": 0.0,
            "llm": 0.0,
            "post_llm_embeddings": 0.0,
            "memory": 0.0,
            "total_before_decision": round(time.perf_counter() - total_start, 3),
            "total": round(time.perf_counter() - total_start, 3)
        }

        return normalized_result
        """
    except Exception as e:
        failed = build_failed_result(file, ext, path, str(e))
        failed["processing_times"] = {
            "extraction": 0.0,
            "normalization": 0.0,
            "pre_llm_embeddings": 0.0,
            "llm": 0.0,
            "post_llm_embeddings": 0.0,
            "memory": 0.0,
            "total_before_decision": round(time.perf_counter() - total_start, 3),
            "total": round(time.perf_counter() - total_start, 3)
        }
        return failed


def apply_optimized_semantic_flow(results: list[dict]) -> list[dict]:
    """
    Controla cuándo se llama al LLM.

    Nota: esta fase es secuencial para no saturar Ollama cuando sí haga falta usarlo.
    Los embeddings son rápidos, pero se mantienen aquí para tomar decisiones ordenadas
    y actualizar la memoria de forma consistente.
    """

    for row in results:
        if row.get("error"):
            row["semantic_profile"] = {}
            row["pre_llm_memory_matches"] = []
            row["memory_matches"] = []
            row["llm_control"] = {"call_llm": False, "level": "error"}
            row["agent_decision"] = build_error_decision(row.get("error"))
            continue

        total_start = time.perf_counter()

        try:
            classification_text = row.get("llm_text") or row.get("normalized_text", "")

            # 1. Comparación previa rápida contra memoria.
            start = time.perf_counter()
            pre_llm_matches = compare_text_with_memory(
                text=classification_text,
                top_k=3
            )
            row["processing_times"]["pre_llm_embeddings"] = round(time.perf_counter() - start, 3)
            row["pre_llm_memory_matches"] = pre_llm_matches

            # 2. Decidir si se llama al LLM.
            llm_control = should_call_llm_from_similarity(pre_llm_matches)
            row["llm_control"] = llm_control

            if not llm_control.get("call_llm"):
                # 3A. Ruta rápida: no se usa LLM.
                final_decision = build_decision_without_llm(pre_llm_matches)
                row["agent_decision"] = final_decision
                row["memory_matches"] = pre_llm_matches

                memory = load_category_memory()
                semantic_profile = build_profile_from_existing_category(
                    memory=memory,
                    category_name=final_decision.get("categoria_final"),
                    document_text=classification_text
                )
                row["semantic_profile"] = semantic_profile

            else:
                # 3B. Ruta LLM: solo si hay incertidumbre o no hay memoria.
                start = time.perf_counter()
                semantic_profile = analyze_document_with_llm(classification_text)
                row["processing_times"]["llm"] = round(time.perf_counter() - start, 3)
                row["semantic_profile"] = semantic_profile

                start = time.perf_counter()
                memory_matches = compare_semantic_profile_with_memory(
                    semantic_profile=semantic_profile,
                    top_k=3
                )
                row["processing_times"]["post_llm_embeddings"] = round(time.perf_counter() - start, 3)
                row["memory_matches"] = memory_matches

                final_decision = decide_category_from_similarity(
                    semantic_profile=semantic_profile,
                    memory_matches=memory_matches
                )
                row["agent_decision"] = final_decision

            # 4. Guardar memoria.
            # Las categorías nuevas/ambiguas se guardan como provisionales, no como definitivas.
            start = time.perf_counter()
            memory = load_category_memory()

            update_category_with_document(
                memory=memory,
                category_name=row["agent_decision"].get("categoria_final"),
                semantic_profile=row.get("semantic_profile", {}),
                final_decision=row.get("agent_decision", {}),
                document_result=row,
                save_provisional=True
            )

            row["processing_times"]["memory"] = round(time.perf_counter() - start, 3)

        except Exception as e:
            row["semantic_profile"] = row.get("semantic_profile", {})
            row["pre_llm_memory_matches"] = row.get("pre_llm_memory_matches", [])
            row["memory_matches"] = row.get("memory_matches", [])
            row["agent_decision"] = build_error_decision(str(e))

        row["processing_times"]["total"] = round(
            row["processing_times"].get("total_before_decision", 0.0)
            + (time.perf_counter() - total_start),
            3
        )

    return results


def build_error_decision(error: str) -> dict:
    return {
        "categoria_final": "No determinado",
        "origen_categoria": "revision",
        "confianza": "baja",
        "similitud": 0.0,
        "requiere_revision": True,
        "llm_usado": False,
        "motivo": f"Error durante el procesamiento: {error}"
    }


def extract_document(ext: str, path: Path) -> dict:
    
    if ext in [".docx", ".doc"]:
        return main_word(ext, path)

    if ext == ".pdf":
        return main_pdf(ext, path)

    if ext in [".xlsx", ".xls"]:
        return main_excel(ext, path)

    if ext == ".txt":
        return main_txt(ext, path)

    if ext == ".xml":
        return main_xml(ext, path)

    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        return main_image(ext, path)

    return {
        "file_type": "unknown",
        "extension": ext,
        "supported": False,
        "extraction_method": None,
        "blocks": [],
        "raw_quality": {
            "empty": True,
            "block_count": 0,
            "text_block_count": 0,
            "table_block_count": 0,
            "image_count": 0,
            "image_ocr_count": 0,
            "total_text_length": 0
        },
        "error": f"Extensión no soportada: {ext}"
    }


def build_failed_result(file: dict, ext: str, path: Path, error: str) -> dict:
    return {
        "file_name": file.get("name") or path.name,
        "full_path": str(path),
        "file_type": "unknown",
        "extension": ext,
        "supported": False,
        "extraction_method": None,
        "normalized_blocks": [],
        "normalized_text": "",
        "llm_text": "",
        "semantic_profile": {},
        "pre_llm_memory_matches": [],
        "memory_matches": [],
        "llm_control": {"call_llm": False, "level": "error"},
        "agent_decision": build_error_decision(error),
        "normalization_quality": {
            "empty": True,
            "original_block_count": 0,
            "normalized_block_count": 0,
            "removed_block_count": 0,
            "original_text_length": 0,
            "normalized_text_length": 0,
            "llm_text_length": 0,
            "reduction_ratio": 0.0
        },
        "error": error
    }
