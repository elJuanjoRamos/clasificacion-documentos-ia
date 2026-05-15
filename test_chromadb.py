import sys, time
sys.path.insert(0, '.')

from backend.embeddings.vector_store import _get_collection, count, query

print("=== CHROMADB STATUS ===")
print("Vectores persistidos en disco:", count())

col = _get_collection()
all_data = col.get(include=["metadatas"])
for i, cat_id in enumerate(all_data["ids"]):
    meta = all_data["metadatas"][i]
    print(f"  [{cat_id}] {meta['name']} | status={meta['status']} | docs={meta['document_count']}")

print()
print("=== QUERY RAPIDA EN CHROMADB (sin recomputar embeddings del corpus) ===")

textos = [
    "La presente factura corresponde a la venta de productos informaticos por valor de 5.000.000",
    "Por medio del presente contrato las partes se obligan a mantener confidencialidad absoluta",
    "Acta de la reunion del comite directivo, orden del dia y asistentes presentes",
]

for texto in textos:
    t = time.perf_counter()
    matches = query(texto, top_k=1)
    elapsed = round((time.perf_counter() - t) * 1000, 1)
    if matches:
        m = matches[0]
        print(f"  Texto  : \"{texto[:60]}...\"")
        print(f"  Mejor  : {m['name']} | similitud={m['similitud_raw']} | tiempo_query={elapsed}ms")
        print()
