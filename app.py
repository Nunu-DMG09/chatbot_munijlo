from fastapi import FastAPI, Query, Body
import requests
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from typing import List
import os
# === PRIMER COMMIT DE PETUSO ===

# === CONFIGURACIÓN ===
# preferir dataset TUPA si existe; si no, usar empresa.json
DATA_FILE_PRI = "tupa_data.json"
DATA_FILE_SEC = "empresa.json"
DATA_FILE = DATA_FILE_PRI if os.path.exists(DATA_FILE_PRI) else DATA_FILE_SEC

OLLAMA_MODEL = "tinyllama:latest"
OLLAMA_MAX_TOKENS = 120     # reducir para respuestas más rápidas
OLLAMA_TIMEOUT = 20        # timeout menor para mayor responsividad
OLLAMA_RETRIES = 0         # reintentos opcional
MAX_DOCS = 3               # documentos usados para generar la respuesta

# === CARGAR DATOS ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# soportar archivos con campos 'name'/'description' (TUPA) o 'nombre'/'descripcion' (empresa.json)
def _get_title(item: dict) -> str:
    return (item.get("name") or item.get("nombre") or item.get("titulo") or "").strip()

def _get_description(item: dict) -> str:
    return (item.get("description") or item.get("descripcion") or item.get("descripcion_corta") or "").strip()

textos = [f"{_get_title(item)}: {_get_description(item)}" for item in data]

# === CONFIGURAR CHROMADB ===
client = chromadb.Client()
try:
    collection = client.get_collection("empresa")
except Exception:
    collection = None

ef = None
try:
    ef_candidate = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
    try:
        _ = ef_candidate(["prueba"])
        ef = ef_candidate
    except Exception:
        ef = None
except Exception:
    ef = None

if ef is None:
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    except Exception:
        ef = None

if collection is None:
    if ef is not None:
        collection = client.create_collection("empresa", embedding_function=ef)
    else:
        collection = client.create_collection("empresa")

if collection.count() == 0:
    ids = [str(i) for i in range(len(textos))]
    collection.add(documents=textos, ids=ids)

app = FastAPI(title="API de Empresa - respuestas basadas solo en empresa.json")

def _normalize(s: str) -> str:
    return (s or "").strip()

def _is_related_query(query: str) -> bool:
    q = (query or "").lower()
    keywords = ["servici", "licencia", "certificado", "tramite", "autoriz", "construcción", "evento", "anuncio", "constancia", "numeración", "defensa civil", "arbitrios", "posesión"]
    return any(k in q for k in keywords)

def _is_greeting(query: str) -> bool:
    if not query:
        return False
    return bool(re.search(r'^\s*(hola|hey|buenas|buenos días|buenas tardes|buenas noches)\b', query, re.I))

def _build_paraphrase_from_item(item: dict, query: str) -> str:
    # título y descripción neutros
    title = _get_title(item)
    desc = _get_description(item)

    # mapear campos alternativos
    requisitos = item.get("requisitos") or item.get("requirements") or []
    duracion = item.get("duracion") or item.get("timeLimit") or ""
    costo = item.get("costo") or item.get("payment") or ""
    oficina = item.get("oficina") or item.get("submissionEntity") or item.get("approvalEntity") or ""
    observaciones = item.get("observaciones") or item.get("calification") or ""
    fundamento = item.get("fundamento_legal") or ""

    qlow = (query or "").lower()

    # detectar intención de campo simple
    if any(k in qlow for k in ("requisit","document","qué necesito","qué se necesita","requirements")):
        if requisitos:
            if isinstance(requisitos, list):
                sample = ", ".join(requisitos[:5])
            else:
                sample = str(requisitos).replace("\n", " ")
            return f"{title}: requisitos principales — {sample}."
        return f"{title}: no hay requisitos especificados."

    if any(k in qlow for k in ("duraci","tiempo","plazo","days","timeLimit")):
        return f"{title}: tiempo aproximado — {duracion}." if duracion else f"{title}: tiempo no especificado."

    if any(k in qlow for k in ("cost","costo","precio","tarifa","cuánto cuesta","cuanto cuesta","monto")):
        return f"{title}: costo — {costo}." if costo else f"{title}: costo no especificado."

    if any(k in qlow for k in ("oficina","sede","donde","dónde","submission","approval","aprobación")):
        return f"{title}: se gestiona en {oficina}." if oficina else f"{title}: oficina no especificada."

    if any(k in qlow for k in ("fundament","ley","ordenanza","legal","norma")):
        return f"{title}: fundamento legal — {fundamento}." if fundamento else f"{title}: fundamento legal no especificado."

    if any(k in qlow for k in ("observ","nota","comentario","detalle","calification")):
        return f"{title}: {observaciones}." if observaciones else f"{title}: sin observaciones."

    if any(k in qlow for k in ("descr","qué es","en qué consiste","para qué sirve","description")):
        if desc:
            snippet = re.split(r'[.!?]\s+', desc)[0]
            return f"{title}: {snippet}."
        return f"{title}: sin descripción."

    # fallback: resumen breve con meta
    parts = []
    if title and desc:
        first_sent = re.split(r'[.!?]\s+', desc)[0]
        parts.append(f"{title}: {first_sent}.")
    elif desc:
        parts.append(desc if desc.endswith(".") else desc + ".")
    elif title:
        parts.append(f"{title}.")

    meta = []
    if oficina: meta.append(f"Atendido en {oficina}")
    if duracion: meta.append(f"Tiempo aproximado: {duracion}")
    if costo: meta.append(f"Costo: {costo}")
    if meta:
        parts.append("; ".join(meta) + ".")
    if observaciones:
        parts.append(f"Nota: {observaciones}")
    out = " ".join(parts).strip()
    sents = re.split(r'(?<=[.!?])\s+', out)
    if len(sents) > 2:
        out = " ".join(sents[:2]).strip()
    return out

def _query_collection(query: str, n_results: int = 5):
    try:
        return collection.query(query_texts=[query], n_results=n_results, include=["ids","documents"])
    except Exception:
        return {"ids": [[str(i) for i in range(len(data))]], "documents": [textos]}

def _format_context_for_model(items: List[dict]) -> str:
    parts = []
    for it in items:
        # soportar ambos esquemas: TUPA (name/description/requirements/timeLimit/payment/channels/form/code)
        # y empresa.json (nombre/descripcion/requisitos/duracion/costo/oficina)
        title = _get_title(it) or ""
        desc = _get_description(it) or ""
        # campos alternativos
        requisitos = it.get("requisitos") or it.get("requirements") or None
        tiempo = it.get("duracion") or it.get("timeLimit") or ""
        costo = it.get("costo") or it.get("payment") or ""
        oficina = it.get("oficina") or it.get("submissionEntity") or it.get("approvalEntity") or ""

        block = f"- {title}. {desc}".strip()
        metas = []
        if oficina: metas.append(f"Oficina: {oficina}")
        if tiempo: metas.append(f"Duración: {tiempo}")
        if costo: metas.append(f"Costo: {costo}")
        if metas:
            block += " (" + "; ".join(metas) + ")"
        parts.append(block)
    return "\n".join(parts)

def _call_ollama(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": OLLAMA_MAX_TOKENS, "temperature": 0.0}
    try:
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=OLLAMA_TIMEOUT)
    except Exception:
        return ""
    # intentar json normal
    try:
        j = r.json()
        # extracción sencilla
        if isinstance(j, dict):
            for k in ("response","text","generated","output","answer"):
                if k in j and isinstance(j[k], str) and j[k].strip():
                    return j[k].strip()
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                ch = j["choices"][0]
                if isinstance(ch, dict):
                    if "text" in ch and isinstance(ch["text"], str):
                        return ch["text"].strip()
                    if "content" in ch:
                        c = ch["content"]
                        if isinstance(c, str): return c.strip()
                        if isinstance(c, list):
                            parts = []
                            for p in c:
                                if isinstance(p, dict):
                                    parts.append(p.get("text") or p.get("content") or "")
                                elif isinstance(p, str):
                                    parts.append(p)
                            return "".join(parts).strip()
    except Exception:
        pass
    # fallback: parsear texto por líneas JSON (streaming)
    parts = []
    text = r.text or ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            idx = line.find("{")
            if idx != -1:
                try:
                    obj = json.loads(line[idx:])
                except Exception:
                    continue
            else:
                continue
        if isinstance(obj, dict):
            if "response" in obj and isinstance(obj["response"], str):
                parts.append(obj["response"])
            elif "text" in obj and isinstance(obj["text"], str):
                parts.append(obj["text"])
            elif "generated" in obj and isinstance(obj["generated"], str):
                parts.append(obj["generated"])
            elif "choices" in obj and isinstance(obj["choices"], list):
                for ch in obj["choices"]:
                    if isinstance(ch, dict):
                        if "text" in ch and isinstance(ch["text"], str):
                            parts.append(ch["text"])
                        if "content" in ch:
                            c = ch["content"]
                            if isinstance(c, str):
                                parts.append(c)
                            elif isinstance(c, list):
                                for b in c:
                                    if isinstance(b, dict):
                                        parts.append(b.get("text","") or b.get("content",""))
                                    elif isinstance(b, str):
                                        parts.append(b)
    return "".join(parts).strip()

def _process_query_with_model(query: str):
    """
    Prioriza coincidencia exacta/contiene con el campo 'nombre' antes
    de usar la búsqueda por similaridad. Responde en español, breve,
    y siempre basado en empresa.json.
    """
    q = (query or "").strip()
    if not q:
        return "No puedo hablar sobre eso."

    # normalizador simple (quita tildes y normaliza a ascii lowercase)
    import unicodedata
    def _norm_txt(s: str) -> str:
        if not s:
            return ""
        s2 = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
        return re.sub(r'\s+', ' ', s2).strip().lower()

    nq = _norm_txt(q)

    # 1) Buscar coincidencia exacta / contiene en 'titulo' (soporta TUPA 'name' y empresa 'nombre')
    exact_candidates = []
    best_len = 0
    for idx, item in enumerate(data):
        title_raw = _get_title(item) or ""
        nn = _norm_txt(title_raw)
        if not nn:
            continue
        if nn == nq or f" {nn} " in f" {nq} " or nq in nn or nn in nq:
            ln = len(nn)
            if ln > best_len:
                exact_candidates = [(idx, item)]
                best_len = ln
            elif ln == best_len:
                exact_candidates.append((idx, item))

    if exact_candidates:
        chosen = exact_candidates[0][1]
        resp = _build_paraphrase_from_item(chosen, query)
        resp = resp.strip()
        if not resp.endswith("."):
            resp += "."
        return resp

    # 2) Si no hay coincidencia exacta, usar búsqueda por Chroma y modelo (flujo anterior)
    res = _query_collection(query, n_results=MAX_DOCS)
    docs = res.get("documents", [[]])[0] if res else []
    ids = res.get("ids", [[]])[0] if res else []
    if not docs:
        return "No puedo hablar sobre eso."

    # construir contexto a partir de documentos devueltos por chroma
    context_items = []
    for id_str in ids[:MAX_DOCS]:
        try:
            context_items.append(data[int(id_str)])
        except Exception:
            continue
    context = _format_context_for_model(context_items) or "\n".join(docs[:MAX_DOCS])

    prompt = (
        "Eres un asistente de IA conversacional que RESPONDE en primera persona y en español. "
        "Usa SOLO la información en la sección 'Información disponible' para responder. "
        "Responde de forma breve y natural (máx 1-2 oraciones). "
        "Si la pregunta pide un dato concreto (requisitos, duración, costo, oficina, fundamento legal, observaciones), "
        "responde exactamente con ese dato y no inventes nada.\n\n"
        "Información disponible:\n"
        f"{context}\n\n"
        "Pregunta: " + query + "\n\n"
        "Respuesta:"
    )

    model_out = _call_ollama(prompt)
    if model_out:
        text = model_out.strip()
        if re.search(r'no puedo (hablar|responder)', text, re.I):
            return "No puedo hablar sobre eso."
        first = re.split(r'[.!?]\s+', text.strip())[0].strip()
        if first:
            if not first.endswith("."):
                first += "."
            if len(first) < 120:
                return f"{first} ¿Quieres que te dé más detalles?"
            return first

    # fallback local si el modelo no responde
    paraphrases = []
    for d in docs[:MAX_DOCS]:
        parts = d.split(":", 1)
        if len(parts) == 2:
            title = parts[0].strip()
            desc = parts[1].strip()
            paraphrases.append(f"{title}: {desc.split('.',1)[0].strip()}")
        else:
            paraphrases.append(d.strip())
    if not paraphrases:
        return "No puedo hablar sobre eso."
    main = paraphrases[0]
    respuesta = f"Puedo ayudarte con eso: {main}."
    respuesta = respuesta[:300].strip()
    if not respuesta.endswith("."):
        respuesta += "."
    return respuesta

@app.post("/ask")
def ask_post(body: dict = Body(...)):
    query = (body.get("query") or "").strip()
    if not query:
        return {"respuesta": "Falta 'query' en el body."}

    # saludo rápido
    if _is_greeting(query):
        return {"respuesta": "Hola, soy un asistente virtual. ¿En qué puedo ayudarte?"}

    # si la consulta claramente no está relacionada, responder genérico
    if not _is_related_query(query):
        return {"respuesta": "No puedo hablar sobre eso."}

    try:
        texto = _process_query_with_model(query)
        # asegurar respuesta corta en español
        return {"respuesta": texto}
    except Exception:
        import traceback
        print(traceback.format_exc())
        return {"respuesta": "No puedo hablar sobre eso."}

@app.get("/ask")
def ask_get(query: str = Query(..., description="Pregunta sobre la empresa")):
    return ask_post({"query": query})

@app.get("/debug")
def debug():
    return {"document_count": collection.count(), "first_docs_preview": textos[:3]}