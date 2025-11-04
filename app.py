from fastapi import FastAPI, Query, Body
import requests
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from typing import List

# === CONFIGURACIÓN ===
DATA_FILE = "empresa.json"
OLLAMA_MODEL = "tinyllama:latest"
OLLAMA_MAX_TOKENS = 180
OLLAMA_TIMEOUT = 60
OLLAMA_RETRIES = 0  # reintentos opcional
MAX_DOCS = 3        # documentos usados para generar la respuesta

# === CARGAR DATOS ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

textos = [f"{item.get('nombre', item.get('titulo',''))}: {item.get('descripcion','')}" for item in data]

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

def _build_paraphrase_from_item(item: dict, query: str) -> str:
    nombre = _normalize(item.get("nombre") or item.get("titulo"))
    descripcion = _normalize(item.get("descripcion"))
    oficina = _normalize(item.get("oficina"))
    duracion = _normalize(item.get("duracion"))
    costo = _normalize(item.get("costo"))
    requisitos = item.get("requisitos", [])
    observaciones = _normalize(item.get("observaciones"))

    qlow = (query or "").lower()
    parts = []

    # Priorizar requisitos si preguntan por ello
    if "requisit" in qlow and requisitos:
        if isinstance(requisitos, list):
            sample = ", ".join(requisitos[:3])
        else:
            sample = str(requisitos)
        return f"{nombre}: requisitos principales — {sample}."

    # Si pregunta por "servicio(s)" devolver descripción resumida
    if "servici" in qlow:
        desc = descripcion.split(".")[0] if descripcion else nombre
        out = f"{nombre}: {desc}."
        meta = []
        if oficina: meta.append(f"Se gestiona en {oficina}")
        if duracion: meta.append(f"tiempo: {duracion}")
        if costo: meta.append(f"costo: {costo}")
        if meta:
            out += " " + "; ".join(meta) + "."
        return out

    # Resumen general humano
    if nombre and descripcion:
        first_sent = re.split(r'[.!?]\s+', descripcion)[0]
        parts.append(f"{nombre}: {first_sent}.")
    elif descripcion:
        parts.append(descripcion if descripcion.endswith(".") else descripcion + ".")
    elif nombre:
        parts.append(f"{nombre}.")

    meta = []
    if oficina: meta.append(f"Atendido en {oficina}")
    if duracion: meta.append(f"Tiempo aproximado: {duracion}")
    if costo: meta.append(f"Costo: {costo}")
    if meta:
        parts.append("; ".join(meta) + ".")

    if observaciones:
        parts.append(f"Nota: {observaciones}")

    out = " ".join(parts).strip()
    # limitar a 2 oraciones
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
        nombre = it.get("nombre") or it.get("titulo","")
        descripcion = it.get("descripcion","").strip()
        oficina = it.get("oficina","")
        duracion = it.get("duracion","")
        costo = it.get("costo","")
        # pequeño bloque por item
        block = f"- {nombre}. {descripcion}"
        metas = []
        if oficina: metas.append(f"Oficina: {oficina}")
        if duracion: metas.append(f"Duración: {duracion}")
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
    Usar el modelo para parafrasear SOLO la información extraída de empresa.json.
    Construye el contexto a partir de los documentos devueltos por chroma (no indexa manualmente el array).
    """
    res = _query_collection(query, n_results=MAX_DOCS)
    docs = res.get("documents", [[]])[0] if res else []
    if not docs:
        return "No puedo hablar sobre eso."

    # Contexto textual (ya contiene título: descripción)
    context_text = "\n".join(docs[:MAX_DOCS]).strip()
    if not context_text:
        return "No puedo hablar sobre eso."

    # Prompt: instruir al modelo a actuar como IA conversacional, usar SOLO el contexto y parafrasear
    prompt = (
        "Eres un asistente de IA conversacional. RESPONDE en primera persona y de forma natural, "
        "usando SOLO la información que aparece en la sección \"Información disponible\" más abajo. "
        "No inventes, no añadas detalles que no estén en esa información. Si la respuesta no puede "
        "extraerse de la información, responde exactamente: \"No puedo hablar sobre eso.\"\n\n"
        "Información disponible (usa solo esto para responder):\n"
        f"{context_text}\n\n"
        "Pregunta: " + query + "\n\n"
        "Respuesta (en español, 1-3 oraciones, tono cordial y claro):"
    )

    # Llamar al modelo local
    model_out = _call_ollama(prompt)
    if model_out:
        # si el modelo decide no puede responder, normalizar la negativa exacta
        if "no puedo hablar" in model_out.lower() or "no puedo responder" in model_out.lower():
            return "No puedo hablar sobre eso."
        # devolver la salida tal cual (ya debería ser parafraseada por el modelo)
        return model_out.strip()

    # Fallback local: parafraseo sencillo a partir de las líneas de contexto (si el modelo no responde)
    paraphrases = []
    for d in docs[:MAX_DOCS]:
        # d suele ser "Título: descripción" según 'textos' añadido a Chroma
        parts = d.split(":", 1)
        if len(parts) == 2:
            title = parts[0].strip()
            desc = parts[1].strip()
            # construir frase humana breve
            frase = f"{title}: {desc.split('.',1)[0].strip()}."
        else:
            frase = d.strip()
            if not frase.endswith("."):
                frase += "."
        paraphrases.append(frase)

    if not paraphrases:
        return "No puedo hablar sobre eso."

    # unir en tono de IA
    salida = paraphrases[0]
    if len(paraphrases) > 1:
        salida = f"{salida} Además, { ' '.join(p for p in paraphrases[1:]) }"
    # normalizar espacio/puntuación
    salida = re.sub(r'\s+', ' ', salida).strip()
    if not salida.endswith("."):
        salida += "."
    return salida

@app.post("/ask")
def ask_post(body: dict = Body(...)):
    query = (body.get("query") or "").strip()
    if not query:
        return {"respuesta": "Falta 'query' en el body."}
    # si la consulta aparentemente no está relacionada, responder genérico
    if not _is_related_query(query):
        return {"respuesta": "No puedo hablar sobre eso."}
    try:
        texto = _process_query_with_model(query)
        return {"respuesta": texto}
    except Exception as e:
        # imprimir traza en consola y responder negativa segura
        import traceback
        print(traceback.format_exc())
        return {"respuesta": "No puedo hablar sobre eso."}

@app.get("/ask")
def ask_get(query: str = Query(..., description="Pregunta sobre la empresa")):
    return ask_post({"query": query})

@app.get("/debug")
def debug():
    return {"document_count": collection.count(), "first_docs_preview": textos[:3]}