import sys
sys.path.insert(0, ".")
from backend.llm.llm import extract_json_from_response, normalize_semantic_profile

# Simular la respuesta truncada que genera phi3
raw = (
    "```json\n"
    "{\n"
    '  "categoria_general": "Factura",\n'
    '  "subcategoria": "Comercial",\n'
    '  "tema": "Contabilidad y facturación",\n'
    " end of document analysis. Based on the provided JSON structure...\n"
)

print("=== RAW INPUT ===")
print(raw)
print("\n=== RESULTADO extract_json_from_response ===")
result = extract_json_from_response(raw)
print(result)

print("\n=== RESULTADO normalize_semantic_profile ===")
normalized = normalize_semantic_profile(result)
print(normalized)

# Test con respuesta totalmente OK
raw_ok = '{"categoria_general": "Contrato", "subcategoria": "Laboral", "tema": "Empleo", "palabras_clave": ["contrato", "trabajo"], "resumen": "Contrato de trabajo."}'
print("\n=== JSON completo válido ===")
print(extract_json_from_response(raw_ok))
