from __future__ import annotations

import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"  # Modelo local descargado. Otras opciones: "mistral", "llama3", "gemma3"


def build_llm_prompt(document_text: str) -> str:
    # Prompt compacto optimizado para phi3 — evita alucinaciones y JSON truncado.
    # phi3 tiende a continuar generando texto después del JSON si el prompt es largo;
    # esta versión mínima produce respuestas más cortas y parse más fiable.
    doc_snippet = document_text[:2500].replace('"', '\\"')
    return (
        'Eres un clasificador de documentos. '
        'Lee el texto y responde ÚNICAMENTE con un objeto JSON válido y completo. '
        'No escribas nada antes ni después del JSON.\n\n'
        'Campos requeridos (todos obligatorios, sin null):\n'
        '- categoria_general: tipo de documento en singular (Factura, Contrato, Informe, Carta, Acta, Certificado, Otros)\n'
        '- subcategoria: variante específica del tipo\n'
        '- tema: contenido concreto resumido en una frase\n'
        '- palabras_clave: array de 3-5 conceptos generales\n'
        '- resumen: descripción breve del documento en 1-2 oraciones\n\n'
        f'Texto del documento:\n"{doc_snippet}"\n\n'
        'Respuesta JSON:'
    )


class OllamaNotAvailableError(RuntimeError):
    """Se lanza cuando el servidor Ollama local no está disponible."""
    pass


def call_ollama(prompt: str, retries: int = 2) -> str:
    """
    Llama al servidor Ollama LOCAL (sin APIs de pago, sin internet).
    Lanza OllamaNotAvailableError con instrucciones claras si no está corriendo.

    Ollama es el runtime que corre modelos como Mistral, Phi3, LLaMA3, Gemma
    directamente en la GPU/CPU del equipo.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()

        except requests.exceptions.ConnectionError as e:
            last_error = e
            break  # No tiene sentido reintentar — el servidor no existe

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < retries:
                continue  # Reintentar en timeout
            break  # No más reintentos tras agotar intentos en timeout

        except requests.exceptions.HTTPError as e:
            last_error = e
            break

    raise OllamaNotAvailableError(
        f"No se pudo conectar a Ollama en {OLLAMA_URL}.\n"
        f"Ollama es el motor LLM LOCAL del sistema (sin APIs de pago).\n\n"
        f"Pasos para iniciarlo (Windows):\n"
        f"  1. Descarga desde https://ollama.com/download\n"
        f"  2. Instala y ejecuta en una terminal:\n"
        f"       ollama serve\n"
        f"  3. En otra terminal, descarga el modelo:\n"
        f"       ollama pull {MODEL_NAME}\n\n"
        f"Modelos recomendados para AMD RX 9060 XT (8 GB VRAM):\n"
        f"  ollama pull phi3           (3.8B, 2.2 GB)  <- más rápido\n"
        f"  ollama pull mistral        (7B,   4.1 GB)\n"
        f"  ollama pull llama3         (8B,   4.7 GB)\n\n"
        f"Error original: {last_error}"
    )


def extract_json_from_response(response_text: str) -> dict:
    """
    Extrae el primer JSON válido de la respuesta del LLM.
    Los LLMs (especialmente phi3) a veces:
      - Añaden texto antes/después del JSON
      - Generan JSON truncado (cortan a la mitad)
      - Usan markdown ```json ... ```
    Esta función intenta múltiples estrategias antes de rendirse.
    """
    if not response_text:
        return {}

    # Intento 1: respuesta directamente JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Intento 2: extraer bloque ```json ... ``` de markdown
    md_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Intento 3: buscar el primer bloque { ... } más completo
    match = re.search(r"\{[\s\S]*\}", response_text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Intento 4: phi3 a veces trunca el JSON — intentar extraer campos individuales
    # con regex para reconstruir un dict mínimo usable.
    partial: dict = {}
    field_patterns = [
        ("categoria_general", r'"categoria_general"\s*:\s*"([^"]*?)"'),
        ("subcategoria",      r'"subcategoria"\s*:\s*"([^"]*?)"'),
        ("tema",              r'"tema"\s*:\s*"([^"]*?)"'),
        ("resumen",           r'"resumen"\s*:\s*"([^"]*?)"'),
    ]
    for field, pattern in field_patterns:
        m = re.search(pattern, response_text)
        if m:
            partial[field] = m.group(1).strip()

    kw_match = re.search(r'"palabras_clave"\s*:\s*\[([^\]]*)\]', response_text)
    if kw_match:
        kws = re.findall(r'"([^"]+)"', kw_match.group(1))
        partial["palabras_clave"] = kws

    if partial:
        return partial

    return {}


def normalize_semantic_profile(profile: dict) -> dict:
    """
    Garantiza que siempre salgan los mismos campos mínimos del perfil semántico.
    """
    return {
        "categoria_general": str(profile.get("categoria_general", "")).strip(),
        "subcategoria":      str(profile.get("subcategoria", "")).strip(),
        "tema":              str(profile.get("tema", "")).strip(),
        "palabras_clave":    profile.get("palabras_clave", [])
                             if isinstance(profile.get("palabras_clave", []), list) else [],
        "resumen":           str(profile.get("resumen", "")).strip(),
    }


def fallback_profile_from_rules(document_text: str) -> dict:
    """
    Perfil mínimo generado por reglas léxicas cuando Ollama no está disponible.
    Permite que el pipeline continúe sin falla total (graceful degradation).
    Marca el resultado con _fallback=True para distinguirlo de una clasificación LLM real.
    """
    text_lower = document_text.lower()

    if any(w in text_lower for w in ["factura", "invoice", "iva", "subtotal", "total a pagar"]):
        cat, sub = "Factura", "Factura de venta"
    elif any(w in text_lower for w in ["acuerdo de confidencialidad", "confidencialidad", "información confidencial"]):
        cat, sub = "Contrato", "Acuerdo de confidencialidad"
    elif any(w in text_lower for w in ["contrato", "contratista", "contratante", "cláusula", "clausula"]):
        cat, sub = "Contrato", "Contrato general"
    elif any(w in text_lower for w in ["informe", "reporte", "resumen ejecutivo"]):
        cat, sub = "Informe", "Informe general"
    elif any(w in text_lower for w in ["acta", "reunión", "asistentes", "orden del día"]):
        cat, sub = "Acta", "Acta de reunión"
    elif any(w in text_lower for w in ["carta", "estimado", "cordialmente", "atentamente"]):
        cat, sub = "Carta", "Carta formal"
    elif any(w in text_lower for w in ["certificado", "certifica", "se hace constar"]):
        cat, sub = "Certificado", "Certificado general"
    else:
        cat, sub = "Otros", "No determinado"

    return {
        "categoria_general": cat,
        "subcategoria":      sub,
        "tema":              "Clasificación por reglas léxicas (Ollama no disponible)",
        "palabras_clave":    [],
        "resumen":           document_text[:200].strip(),
        "_fallback":         True,  # bandera: NO es clasificación real con LLM
    }


def analyze_document_with_llm(document_text: str) -> dict:
    """
    Función principal del bloque LLM. 100% LOCAL usando Ollama.
    Genera el perfil semántico del documento.

    - No decide la categoría final (eso lo hace el agente)
    - No consulta la memoria de categorías
    - No usa embeddings

    Si Ollama no está disponible, usa fallback_profile_from_rules()
    para degradar con gracia sin interrumpir el pipeline completo.
    """
    try:
        prompt = build_llm_prompt(document_text)
        raw_response = call_ollama(prompt)
        parsed = extract_json_from_response(raw_response)
        return normalize_semantic_profile(parsed)
    except OllamaNotAvailableError as e:
        print(f"\n[LLM FALLBACK] Ollama no disponible. Usando clasificación por reglas.\n{e}\n")
        return fallback_profile_from_rules(document_text)


def debug_analyze_document_with_llm(document_text: str) -> dict:
    """
    Versión de debug que devuelve el prompt enviado, la respuesta cruda del LLM
    y el JSON parseado. Útil para ajustar el prompt o depurar respuestas incorrectas.
    """
    prompt = build_llm_prompt(document_text)
    raw_response = call_ollama(prompt)
    parsed = extract_json_from_response(raw_response)
    profile = normalize_semantic_profile(parsed)

    return {
        "prompt":           prompt,
        "raw_response":     raw_response,
        "parsed_response":  parsed,
        "semantic_profile": profile,
    }
