from __future__ import annotations

import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # o "phi3"

def build_llm_prompt(document_text: str) -> str:
    schema = {
        "categoria_general": "",
        "subcategoria": "",
        "tema": "",
        "palabras_clave": [],
        "resumen": ""
    }

    return f"""
Analiza el siguiente documento y genera un perfil semántico.

Reglas IMPORTANTES:

- categoria_general = TIPO DE DOCUMENTO (no área temática) siempre en singular
  Ejemplos correctos:
  - "Informe"
  - "Factura"
  - "Contrato"
  - "Carta"
  - "Diligencia"
  - "Documento académico"

  Ejemplos incorrectos:
  - "Documentos financieros"
  - "Documentos legales"
  - "Finanzas empresariales"
  - "Economía"

- subcategoria = variante específica del tipo de documento
  Ejemplos:
  - "Informe financiero"
  - "Factura de venta"
  - "Contrato laboral"
  - "Trabajo de Fin de Máster"

- tema = contenido concreto del documento (puede ser específico)

- NO incluyas nombres propios, fechas, montos o datos únicos en categoria_general o subcategoria

- palabras_clave deben ser conceptos generales (no valores únicos)

- Devuelve SOLO JSON válido
- No agregues explicación fuera del JSON

Formato esperado:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Documento:
\"\"\"
{document_text[:5000]}
\"\"\"
""".strip()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def extract_json_from_response(response_text: str) -> dict:
    """
    Extrae el primer JSON válido de la respuesta del LLM.
    """

    if not response_text:
        return {}

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", response_text)

    if not match:
        return {}

    json_text = match.group(0)

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        return {}


def normalize_semantic_profile(profile: dict) -> dict:
    """
    Garantiza que siempre salgan los mismos campos mínimos.
    """

    return {
        "categoria_general": str(profile.get("categoria_general", "")).strip(),
        "subcategoria": str(profile.get("subcategoria", "")).strip(),
        "tema": str(profile.get("tema", "")).strip(),
        "palabras_clave": profile.get("palabras_clave", []) if isinstance(profile.get("palabras_clave", []), list) else [],
        "resumen": str(profile.get("resumen", "")).strip()
    }


def analyze_document_with_llm(document_text: str) -> dict:
    """
    Función principal del bloque LLM.
    Solo genera perfil semántico.
    No decide categoría final.
    No consulta memoria.
    No usa embeddings.
    """

    prompt = build_llm_prompt(document_text)
    raw_response = call_ollama(prompt)
    parsed = extract_json_from_response(raw_response)
    profile = normalize_semantic_profile(parsed)

    return profile


def debug_analyze_document_with_llm(document_text: str) -> dict:
    """
    Úsala temporalmente para revisar qué está pasando.
    Devuelve prompt, respuesta cruda y JSON parseado.
    """

    prompt = build_llm_prompt(document_text)
    raw_response = call_ollama(prompt)
    parsed = extract_json_from_response(raw_response)
    profile = normalize_semantic_profile(parsed)

    return {
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_response": parsed,
        "semantic_profile": profile
    }


