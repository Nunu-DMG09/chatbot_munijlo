from fastapi import FastAPI, Query, Body
import requests
import json
import re
import os
from typing import List, Dict, Any
import unicodedata
from uuid import uuid4

# === CONFIGURACIÓN ===
DATA_FILE = "tupa_data.json" if os.path.exists("tupa_data.json") else "empresa.json"
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

# === CARGAR DATOS ===
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# === NORMALIZACIÓN Y PRECOMPUTE PARA RENDIMIENTO ===
import time
import hashlib

def normalizar_texto(texto: str) -> str:
    """Normaliza texto quitando tildes y convirtiendo a minúsculas"""
    if not texto:
        return ""
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', texto.lower().strip())

# === PRECOMPUTE PARA RENDIMIENTO ===
import time
import hashlib

# normalizar y precomputar títulos/tokens para búsquedas rápidas
for item in data:
    title = item.get("name") or item.get("title") or item.get("nombre") or ""
    item["_norm_title"] = normalizar_texto(title)
    text = " ".join([
        str(item.get("name", "")),
        str(item.get("title", "")),
        str(item.get("nombre", "")),
        str(item.get("description", "")),
        str(item.get("requirements", "")),
        str(item.get("queryInformation", "")),
    ])
    item["_tokens"] = set(re.findall(r'\w{3,}', normalizar_texto(text)))

# cache simple en memoria para respuestas IA
_ai_cache: Dict[str, Dict[str, Any]] = {}  # key -> {"resp": str, "ts": float}
AI_CACHE_TTL = 300  # segundos

# cache del JSON truncado (no recalcular cada petición)
_raw_json_cache = None
_raw_json_cache_ts = 0
RAW_JSON_MAX_CHARS = 20000

def contexto_json_truncado() -> str:
    global _raw_json_cache, _raw_json_cache_ts
    if _raw_json_cache and (time.time() - _raw_json_cache_ts) < AI_CACHE_TTL:
        return _raw_json_cache
    try:
        raw = json.dumps(data, ensure_ascii=False)
    except Exception:
        raw = str(data)
    if len(raw) > RAW_JSON_MAX_CHARS:
        half = RAW_JSON_MAX_CHARS // 2
        out = raw[:half] + "\n\n... (truncado) ...\n\n" + raw[-half:]
    else:
        out = raw
    _raw_json_cache = out
    _raw_json_cache_ts = time.time()
    return out

app = FastAPI(title="Chatbot Municipal JLO - IA Conversacional")

# === GESTIÓN DE SESIONES ===
sessions = {}
MAX_HISTORY = 6

def normalizar_texto(texto: str) -> str:
    """Normaliza texto quitando tildes y convirtiendo a minúsculas"""
    if not texto:
        return ""
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'\s+', ' ', texto.lower().strip())

def _titulo_corto(item: Dict[str, Any]) -> str:
    """Devuelve un título más legible del ítem"""
    name = item.get("name") or item.get("title") or item.get("nombre") or ""
    # quitar comillas si las tiene en el JSON
    return name.strip().strip('"').strip()

def _detalle_corto(item: Dict[str, Any]) -> str:
    """Formato breve y humano del trámite (1-2 oraciones)"""
    nombre = _titulo_corto(item)
    desc = (item.get("description") or item.get("description", "")).strip()
    requisitos = item.get("requirements") or item.get("requirements", "")
    tiempo = item.get("timeLimit") or item.get("timeLimit", "")
    pago = item.get("payment") or item.get("payment", "")
    canales = item.get("channels") or item.get("channels") or item.get("location","")
    parts = []
    if desc:
        parts.append(re.split(r'[.!?]\s+', desc)[0].strip())
    meta = []
    if requisitos:
        meta.append("requisitos: " + (requisitos.splitlines()[0][:140].strip()))
    if tiempo:
        meta.append("tiempo: " + tiempo)
    if pago:
        meta.append("costo: " + pago.splitlines()[0][:60].strip())
    if canales:
        meta.append("canal: " + canales.splitlines()[0][:80].strip())
    out = f"{nombre}: " + (" ".join(parts) if parts else "")
    if meta:
        out = out + " (" + "; ".join(meta[:3]) + ")"
    out = out.strip()
    if not out.endswith("."):
        out += "."
    return out

def es_saludo(consulta: str) -> bool:
    """Detecta saludos"""
    saludos = ["hola", "hey", "buenas", "buenos dias", "buenas tardes", "buenas noches", "saludos"]
    consulta_norm = normalizar_texto(consulta)
    return any(saludo in consulta_norm for saludo in saludos)

def es_despedida(consulta: str) -> bool:
    """Detecta despedidas"""
    despedidas = ["adios", "hasta luego", "nos vemos", "gracias", "bye"]
    consulta_norm = normalizar_texto(consulta)
    return any(despedida in consulta_norm for despedida in despedidas)

# Nuevo: detectar consultas fuera de alcance (seguridad) — debe estar definido antes de usarlo en /chat
def es_fuera_de_alcance(consulta: str) -> bool:
    """Detecta temas que no se deben responder (suicidio, violencia, armas, drogas, pornografía, etc.)."""
    if not consulta:
        return False
    q = normalizar_texto(consulta)
    patrones = [
        r'\bsuicid', r'\bquitarme la vida', r'\bquitarse la vida', r'\bmatar(me|te|lo|la)\b',
        r'\basesinar\b', r'\bbomba\b', r'\bexplos', r'\bcrear una bomba', r'\bveneno\b',
        r'\bdroga(s)?\b', r'\bataque\b', r'\bhackear\b', r'\bporno\b', r'\bsexo explícito\b',
        r'\bhacer daño\b', r'\bplanear un ataque\b'
    ]
    return any(re.search(p, q) for p in patrones)

def extraer_palabras_clave(consulta: str) -> List[str]:
    """Extrae palabras clave relevantes de la consulta"""
    consulta_norm = normalizar_texto(consulta)
    # Palabras irrelevantes a filtrar
    stop_words = {"de", "la", "el", "en", "y", "a", "que", "es", "se", "del", "las", "los", "un", "una", 
                  "con", "por", "para", "como", "sobre", "cual", "donde", "cuando", "quien", "me", "te"}
    
    palabras = consulta_norm.split()
    return [p for p in palabras if len(p) > 2 and p not in stop_words]

def buscar_por_titulo_exacto(mensaje: str) -> List[Dict[str, Any]]:
    """Coincidencia fuerte en título (igualdad o el título está contenido en la consulta)."""
    nq = normalizar_texto(mensaje)
    resultados: List[Dict[str, Any]] = []
    for item in data:
        title = _titulo_corto(item)
        if not title:
            continue
        nn = normalizar_texto(title)
        # igualdad exacta o el título aparece como frase en la consulta o la consulta en el título
        if nn == nq or nn in nq or nq in nn:
            resultados.append(item)
    return resultados

def buscar_por_titulo_parcial(mensaje: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Coincidencia parcial por tokens en título. Devuelve lista corta (limit)."""
    tokens = [t for t in re.findall(r'\w{3,}', normalizar_texto(mensaje))]
    if not tokens:
        return []
    resultados: List[Dict[str, Any]] = []
    for item in data:
        title = _titulo_corto(item)
        if not title:
            continue
        nn = normalizar_texto(title)
        # si cualquier token aparece en el título (contención) o título comienza por token
        if any(tok in nn for tok in tokens) or any(nn.startswith(tok) for tok in tokens):
            resultados.append(item)
            if len(resultados) >= limit:
                break
    return resultados

def buscar_tramites_inteligente(consulta: str, k: int = 5) -> List[Dict[str, Any]]:
    """Búsqueda rápida usando tokens precomputados.
    Requiere un mínimo de solapamiento para considerar relevante (mejora precisión)."""
    qtokens = set(re.findall(r'\w{3,}', normalizar_texto(consulta)))
    if not qtokens:
        return []
    # umbral mínimo: si la consulta tiene 1 token, requisito 1; si >=2 tokens, requisito 2
    min_overlap = 1 if len(qtokens) <= 1 else 2
    scored = []
    for item in data:
        overlap = len(qtokens & item.get("_tokens", set()))
        if overlap >= min_overlap:
            scored.append((overlap, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for sc, it in scored[:k]]

def _lista_titulos(items: List[Dict[str, Any]], limit: int = 10) -> str:
    """Retorna lista corta de títulos legibles separada por líneas."""
    lines = []
    for it in items[:limit]:
        lines.append(_titulo_corto(it))
    return "\n".join(lines)

# Reemplaza el endpoint /chat por la lógica solicitada:
@app.post("/chat")
def chat(body: dict = Body(...)):
    mensaje = (body.get("mensaje") or "").strip()
    session_id = body.get("session_id") or uuid4().hex

    if not mensaje:
        return {"respuesta": "Escribe tu consulta, por favor.", "session_id": session_id}

    # bloqueo estricto para temas fuera de alcance
    if es_fuera_de_alcance(mensaje):
        return {"respuesta": "No puedo hablar sobre eso.", "session_id": session_id, "candidatos": 0}

    # saludos y despedidas manejadas de forma segura (no ambigüedad)
    if es_saludo(mensaje):
        return {"respuesta": "Hola, soy Leonardito, el asistente virtual. ¿En qué trámite te puedo ayudar?", "session_id": session_id}
    if es_despedida(mensaje):
        return {"respuesta": "Si necesitas algo más sobre trámites municipales, dímelo.", "session_id": session_id}

    # 1) coincidencia exacta en título
    exactos = buscar_por_titulo_exacto(mensaje)
    if len(exactos) == 1:
        # item único -> pedir a la IA que RESUMA usando SOLO ese ítem + JSON truncado
        item = exactos[0]
        contexto = _detalle_corto(item)
        ai_out = generar_respuesta_ia(contexto, mensaje)
        if ai_out:
            return {"respuesta": ai_out.strip(), "session_id": session_id, "candidatos": 1}
        return {"respuesta": contexto, "session_id": session_id, "candidatos": 1}

    if len(exactos) > 1:
        # varios exactos -> listar títulos y pedir especificar
        lista = _lista_titulos(exactos)
        texto = f"Encontré varios procesos que coinciden exactamente:\n\n{lista}\n\nPor favor indica cuál deseas consultar."
        return {"respuesta": texto, "session_id": session_id, "candidatos": len(exactos)}

    # 2) coincidencias parciales por título
    parciales = buscar_por_titulo_parcial(mensaje, limit=10)
    if len(parciales) == 1:
        item = parciales[0]
        contexto = _detalle_corto(item)
        ai_out = generar_respuesta_ia(contexto, mensaje)
        if ai_out:
            return {"respuesta": ai_out.strip(), "session_id": session_id, "candidatos": 1}
        return {"respuesta": contexto, "session_id": session_id, "candidatos": 1}

    if len(parciales) > 1:
        lista = _lista_titulos(parciales, limit=10)
        texto = f"Encontré varios procesos relacionados con tu consulta:\n\n{lista}\n\nPor favor indica cuál deseas consultar."
        return {"respuesta": texto, "session_id": session_id, "candidatos": len(parciales)}

    # 3) búsqueda por solapamiento en campos (inteligente)
    tramites = buscar_tramites_inteligente(mensaje, k=5)
    if not tramites:
        # sin coincidencias relevantes -> no responder fuera de contexto
        return {"respuesta": "No puedo hablar sobre eso. Puedo ayudar solo con trámites y servicios municipales listados.", "session_id": session_id, "candidatos": 0}

    if len(tramites) == 1:
        item = tramites[0]
        contexto = _detalle_corto(item)
        ai_out = generar_respuesta_ia(contexto, mensaje)
        if ai_out:
            return {"respuesta": ai_out.strip(), "session_id": session_id, "candidatos": 1}
        return {"respuesta": contexto, "session_id": session_id, "candidatos": 1}

    # varios resultados relevantes -> listar y pedir especificación
    lista = _lista_titulos(tramites, limit=10)
    texto = f"Encontré varios trámites que pueden coincidir:\n\n{lista}\n\nPor favor indica cuál deseas consultar."
    return {"respuesta": texto, "session_id": session_id, "candidatos": len(tramites)}

@app.post("/reset_session")
def reset_session(body: dict = Body(...)):
    """Reinicia una sesión de conversación"""
    session_id = body.get("session_id")
    if session_id and session_id in sessions:
        del sessions[session_id]
        return {"mensaje": "Sesión reiniciada correctamente"}
    return {"mensaje": "Sesión no encontrada"}

@app.get("/debug/sessions")
def debug_sessions():
    """Ver sesiones activas para debug"""
    return {
        "sesiones_activas": len(sessions),
        "total_tramites": len(data)
    }

def _call_ollama(prompt: str) -> str:
    """Llamada a Ollama con cache y ensamblado NDJSON. Menor max_tokens y timeout para velocidad."""
    # cache por hash de prompt
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    entry = _ai_cache.get(key)
    if entry and (time.time() - entry["ts"]) < AI_CACHE_TTL:
        return entry["resp"]

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        # reducir max_tokens para respuestas breves y menor latencia
        "max_tokens": 140,
        "temperature": 0.2,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=25)
        r.raise_for_status()
    except Exception:
        return ""

    # intentar parsear JSON completo primero
    try:
        j = r.json()
        # casos simples
        if isinstance(j, dict):
            for k in ("response", "text", "generated", "output", "answer"):
                v = j.get(k)
                if isinstance(v, str) and v.strip():
                    out = v.strip()
                    _ai_cache[key] = {"resp": out, "ts": time.time()}
                    return out
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                ch = j["choices"][0]
                if isinstance(ch, dict):
                    if isinstance(ch.get("text"), str) and ch["text"].strip():
                        out = ch["text"].strip()
                        _ai_cache[key] = {"resp": out, "ts": time.time()}
                        return out
        if isinstance(j, list):
            parts = []
            for obj in j:
                if isinstance(obj, dict):
                    parts.append(obj.get("response") or obj.get("text") or "")
                elif isinstance(obj, str):
                    parts.append(obj)
            out = "".join(p for p in parts if p).strip()
            if out:
                _ai_cache[key] = {"resp": out, "ts": time.time()}
                return out
    except ValueError:
        pass

    # fallback a NDJSON/stream parsing de texto
    text = r.text or ""
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            parts.append(line)
            continue
        v = None
        for k in ("response", "text", "generated", "output", "answer"):
            if k in obj:
                if isinstance(obj[k], str):
                    v = obj[k]
                    break
                if isinstance(obj[k], list):
                    v = "".join([p if isinstance(p, str) else str(p) for p in obj[k]])
                    break
                if isinstance(obj[k], dict):
                    v = obj[k].get("text") or obj[k].get("content") or None
                    if v:
                        break
        if v:
            parts.append(v)
    out = "".join(parts).strip()
    if out:
        _ai_cache[key] = {"resp": out, "ts": time.time()}
    return out

def generar_respuesta_ia(item_contexto: str, pregunta: str) -> str:
    """Construye prompt que obliga a Leonardito a PARALEFRASEAR/RESUMIR usando SOLO la info
    y realiza una limpieza si el modelo devuelve JSON crudo."""
    # contexto JSON truncado (ya definido)
    json_ctx = contexto_json_truncado()

    prompt = (
        "Eres Leonardito, asistente virtual de la Municipalidad de José Leonardo Ortiz.\n"
        "INSTRUCCIONES (MUY IMPORTANTES):\n"
        "- Usa SOLO la información provista en 'ITEM' y en 'JSON_COMPLETO'.\n"
        "- No repitas ni pegues fragmentos literales del JSON. Parafrasea y resume la información en español natural.\n"
        "- Si el usuario pide un campo concreto (requisitos, tiempo, costo, contacto), responde solo con ese dato.\n"
        "- Si la consulta no puede responderse con la información provista, responde EXACTAMENTE: \"No puedo hablar sobre eso.\"\n"
        "- Responde breve (1-2 frases) en primera persona cuando corresponda.\n\n"
        "ITEM:\n" + item_contexto + "\n\n"
        "JSON_COMPLETO:\n" + json_ctx + "\n\n"
        "PREGUNTA: " + pregunta + "\n\n"
        "RESPUESTA:"
    )

    # Llamada inicial
    out = _call_ollama(prompt)
    if not out:
        return ""

    # Si la salida contiene JSON/NDJSON/crudo o muchos caracteres de JSON, pedir reescritura breve
    if (("{" in out and "}" in out) or '"' in out or 'queryInformation' in out or 'requirements' in out or re.search(r'\{.*\}', out)):
        clean_prompt = (
            "Reescribe el siguiente texto en español natural y MUY BREVE (1 frase), "
            "parafraseando y sin incluir ningún fragmento JSON ni nombres de campos:\n\n"
            + out + "\n\nRespuesta breve:"
        )
        out2 = _call_ollama(clean_prompt)
        if out2:
            return out2.strip()
        # si no hay respuesta limpia, intentar una extracción simple: tomar la primera oración
    # tomar primera oración completa para mayor concisión
    s = re.split(r'[.!?]\s+', out.strip())
    if s:
        first = s[0].strip()
        if first and not first.endswith("."):
            first += "."
        return first
    return out.strip()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)