import json, sys, time
sys.path.insert(0, '.')

from pathlib import Path
from main import scan_folder
from backend.procesador import process_documents

folder = Path('./docs_prueba')
print('Escaneando carpeta:', folder)
files = scan_folder(folder)
print('Archivos encontrados:', len(files))
for f in files:
    print(' -', f['name'], '|', f['type'])

print()
print('Iniciando pipeline: extraccion + normalizacion + LLM + embeddings + agente...')
print()

t = time.perf_counter()
results = process_documents(files)
total = round(time.perf_counter() - t, 1)

print(f'Pipeline completado en {total}s')
print()
print('=== RESULTADOS DEL AGENTE ===')
for r in results:
    d = r.get('agent_decision', {})
    p = r.get('semantic_profile', {})
    times = r.get('processing_times', {})
    print()
    print('Archivo  :', r.get('file_name'))
    print('Categoria:', d.get('categoria_final'))
    print('Subtipo  :', p.get('subcategoria'))
    print('Tema     :', p.get('tema'))
    print('Confianza:', d.get('confianza'), '| Origen:', d.get('origen_categoria'))
    print('LLM usado:', d.get('llm_usado'), '| Fallback:', p.get('_fallback', False))
    print('Similitud:', d.get('similitud'))
    print('Tiempos  : ext=', times.get('extraction'), 's  norm=', times.get('normalization'), 's  llm=', times.get('llm'), 's  total=', times.get('total'), 's')

print()
print('Memoria guardada en backend/category_memory/category_memory.json')
import json as js
with open('backend/category_memory/category_memory.json', 'r', encoding='utf-8') as mf:
    mem = js.load(mf)
print('Categorias en memoria:', len(mem.get('categories', [])))
for c in mem.get('categories', []):
    print(' -', c.get('name'), '| status:', c.get('status'), '| documentos:', c.get('document_count'))
