import requests, json, time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

prompt = """Analiza este documento y devuelve SOLO JSON valido sin texto adicional:

Documento: "Factura numero 001, total 500 pesos, IVA incluido"

Formato exacto requerido:
{
  "categoria_general": "",
  "subcategoria": "",
  "tema": "",
  "palabras_clave": [],
  "resumen": ""
}"""

payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {"temperature": 0.1}
}

print(f"Enviando prompt a Ollama ({MODEL})...")
t = time.perf_counter()
try:
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    elapsed = time.perf_counter() - t
    data = r.json()
    raw = data.get("response", "")
    print(f"\n=== TIEMPO: {elapsed:.1f}s ===")
    print(f"=== RESPUESTA RAW ({len(raw)} chars) ===")
    print(raw[:3000])
    print("\n=== PARSE JSON ===")
    try:
        parsed = json.loads(raw)
        print("JSON directo OK:", parsed)
    except Exception:
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                print("JSON extraido con regex OK:", parsed)
            except Exception as e:
                print("Fallo regex parse:", e)
                print("Raw match:", match.group(0)[:500])
        else:
            print("No se encontro JSON en la respuesta")
except Exception as e:
    print(f"Error: {e}")
