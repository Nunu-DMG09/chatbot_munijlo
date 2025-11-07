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
    canales = item.get("channels") or item.get("channels", "") or item.get("location","")
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
    despedidas = ["adios", "chau", "hasta luego", "nos vemos", "gracias", "bye"]
    consulta_norm = normalizar_texto(consulta)
    return any(despedida in consulta_norm for despedida in despedidas)

def extraer_palabras_clave(consulta: str) -> List[str]:
    """Extrae palabras clave relevantes de la consulta"""
    consulta_norm = normalizar_texto(consulta)
    # Palabras irrelevantes a filtrar
    stop_words = {"de", "la", "el", "en", "y", "a", "que", "es", "se", "del", "las", "los", "un", "una", 
                  "con", "por", "para", "como", "sobre", "cual", "donde", "cuando", "quien", "me", "te"}
    
    palabras = consulta_norm.split()
    return [p for p in palabras if len(p) > 2 and p not in stop_words]

def buscar_tramites_inteligente(consulta: str) -> List[Dict[str, Any]]:
    """Búsqueda inteligente que considera múltiples factores"""
    palabras_clave = extraer_palabras_clave(consulta)
    resultados_con_score = []
    
    for idx, item in enumerate(data):
        score = 0
        nombre = normalizar_texto(item.get("name", ""))
        descripcion = normalizar_texto(item.get("description", ""))
        requisitos = normalizar_texto(str(item.get("requirements", "")))
        
        texto_completo = f"{nombre} {descripcion} {requisitos}"
        
        # Puntuación por coincidencias exactas en nombre (mayor peso)
        for palabra in palabras_clave:
            if palabra in nombre:
                score += 10
            elif palabra in descripcion:
                score += 5
            elif palabra in requisitos:
                score += 2
        
        # Puntuación por coincidencias parciales
        for palabra in palabras_clave:
            if any(palabra in palabra_texto for palabra_texto in texto_completo.split()):
                score += 1
        
        if score > 0:
            resultados_con_score.append({
                "item": item,
                "score": score,
                "index": idx
            })
    
    # Ordenar por score descendente y tomar los mejores
    resultados_con_score.sort(key=lambda x: x["score"], reverse=True)
    return [r["item"] for r in resultados_con_score[:3]]

def generar_respuesta_con_ia(tramites: List[Dict], consulta: str, historial: List[Dict] = None) -> str:
    """Genera respuesta usando tinyllama con contexto conversacional"""
    if not tramites:
        return "No encontré información específica sobre tu consulta. ¿Podrías ser más específico sobre qué trámite o servicio municipal necesitas?"
    
    # Construir contexto de trámites
    contexto_tramites = ""
    for i, tramite in enumerate(tramites[:3], 1):
        nombre = tramite.get("name", "Trámite")
        descripcion = tramite.get("description", "")
        requisitos = tramite.get("requirements", "")
        costo = tramite.get("payment", "")
        tiempo = tramite.get("timeLimit", "")
        
        contexto_tramites += f"\n{i}. {nombre}:\n"
        if descripcion:
            contexto_tramites += f"   Descripción: {descripcion[:200]}...\n"
        if requisitos:
            contexto_tramites += f"   Requisitos: {requisitos[:150]}...\n"
        if costo:
            contexto_tramites += f"   Costo: {costo}\n"
        if tiempo:
            contexto_tramites += f"   Tiempo: {tiempo}\n"
    
    # Construir historial conversacional
    contexto_historial = ""
    if historial:
        contexto_historial = "\nHistorial de conversación:\n"
        for msg in historial[-4:]:  # Últimos 4 mensajes
            rol = "Usuario" if msg["role"] == "user" else "Asistente"
            contexto_historial += f"{rol}: {msg['content'][:100]}...\n"
    
    # Prompt para tinyllama
    prompt = f"""Eres un asistente virtual de la Municipalidad de José Leonardo Ortiz. Responde de manera amigable y profesional.

INFORMACIÓN DISPONIBLE:{contexto_tramites}

{contexto_historial}

CONSULTA ACTUAL: {consulta}

INSTRUCCIONES:
- Responde en español de manera conversacional y amigable
- Usa SOLO la información proporcionada
- Si preguntan por requisitos, tiempo o costo, sé específico
- Si la consulta no está clara, pide aclaración
- Mantén un tono profesional pero cercano
- No inventes información que no esté en los datos

RESPUESTA:"""

    try:
        # Llamada a tinyllama
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 200,
                "top_p": 0.9
            }
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        respuesta_ia = result.get("response", "").strip()
        
        if respuesta_ia and len(respuesta_ia) > 10:
            return respuesta_ia
        else:
            # Fallback si la IA no responde bien
            return generar_respuesta_fallback(tramites[0], consulta)
            
    except Exception as e:
        print(f"Error con Ollama: {e}")
        return generar_respuesta_fallback(tramites[0], consulta)

def generar_respuesta_fallback(tramite: Dict, consulta: str) -> str:
    """Respuesta de fallback cuando la IA no está disponible"""
    nombre = tramite.get("name", "")
    descripcion = tramite.get("description", "")[:150]
    costo = tramite.get("payment", "")
    tiempo = tramite.get("timeLimit", "")
    
    respuesta = f"Te ayudo con información sobre {nombre}. "
    
    if descripcion:
        respuesta += f"{descripcion}. "
    
    detalles = []
    if costo and costo != "Gratuito":
        detalles.append(f"Costo: {costo}")
    elif costo == "Gratuito":
        detalles.append("Es gratuito")
    
    if tiempo:
        detalles.append(f"Tiempo: {tiempo}")
    
    if detalles:
        respuesta += f"({', '.join(detalles)}). "
    
    respuesta += "¿Necesitas información específica sobre requisitos o algún otro detalle?"
    
    return respuesta

def buscar_por_titulo_exacto(mensaje: str) -> List[Dict[str, Any]]:
    """Busca coincidencia exacta o muy cercana en campos name/title/nombre."""
    nq = normalizar_texto(mensaje)
    candidatos: List[Dict[str, Any]] = []
    for item in data:
        title = _titulo_corto(item)
        if not title:
            continue
        nn = normalizar_texto(title)
        if nn == nq or f" {nn} " in f" {nq} " or nq in nn or nn in nq:
            candidatos.append(item)
    return candidatos

def buscar_por_titulo_parcial(mensaje: str) -> List[Dict[str, Any]]:
    """Busca títulos que parcialmente coincidan (tokens o prefijos)."""
    tokens = [t for t in re.findall(r'\w{4,}', normalizar_texto(mensaje))]
    resultados: List[Dict[str, Any]] = []
    if not tokens:
        return resultados
    for item in data:
        title = _titulo_corto(item)
        if not title:
            continue
        nn = normalizar_texto(title)
        if any(tok in nn for tok in tokens) or any(nn.startswith(tok) for tok in tokens):
            resultados.append(item)
    return resultados

@app.get("/")
def inicio():
    return {
        "mensaje": "¡Hola! Soy Leonardito, el asistente virtual de la Municipalidad de José Leonardo Ortiz.",
        "uso": "Usa POST /chat con {'mensaje': 'tu consulta', 'session_id': 'opcional'}"
    }

@app.post("/chat")
def chat(body: dict = Body(...)):
    mensaje = (body.get("mensaje") or "").strip()
    session_id = body.get("session_id") or uuid4().hex

    if not mensaje:
        return {"respuesta": "Escribe tu consulta, por favor.", "session_id": session_id}

    # saludos/despedidas rápidos
    if es_saludo(mensaje):
        return {"respuesta": "Hola, soy Leonardito, el asistente virtual. ¿En qué trámite te puedo ayudar?", "session_id": session_id}
    if es_despedida(mensaje):
        return {"respuesta": "Gracias. Si necesitas más ayuda, escríbeme.", "session_id": session_id}

    # 1) Prioridad: coincidencia exacta en título -> devolver detalle directo (si único)
    exact = buscar_por_titulo_exacto(mensaje)
    if len(exact) == 1:
        detalle = _detalle_corto(exact[0])
        return {"respuesta": detalle, "session_id": session_id, "candidatos": 1}

    if len(exact) > 1:
        # si hay varias coincidencias exactas, listar títulos y pedir especificar
        lista = [ _titulo_corto(it) for it in exact ]
        texto = "Encontré varios procesos que coinciden exactamente:\n\n" + "\n".join(lista) + "\n\nPor favor indica cuál deseas consultar."
        return {"respuesta": texto, "session_id": session_id, "candidatos": len(lista)}

    # 2) Búsqueda parcial por título -> si varios, listar y pedir especificación
    parc = buscar_por_titulo_parcial(mensaje)
    if len(parc) == 1:
        detalle = _detalle_corto(parc[0])
        return {"respuesta": detalle, "session_id": session_id, "candidatos": 1}
    if len(parc) > 1:
        lista = [ _titulo_corto(it) for it in parc ]
        texto = f"Encontré varios procesos relacionados con tu consulta:\n\n" + "\n".join(lista) + "\n\nPor favor indica cuál deseas consultar."
        return {"respuesta": texto, "session_id": session_id, "candidatos": len(lista)}

    # 3) Si no hay matches en títulos, intentar búsqueda inteligente (nombre/descripcion/requisitos)
    tramites_encontrados = buscar_tramites_inteligente(mensaje)
    if not tramites_encontrados:
        return {"respuesta": "No puedo hablar sobre eso.", "session_id": session_id, "candidatos": 0}

    # si la búsqueda inteligente devuelve un solo trámite relevante, mostrar detalle breve
    if len(tramites_encontrados) == 1:
        detalle = _detalle_corto(tramites_encontrados[0])
        return {"respuesta": detalle, "session_id": session_id, "candidatos": 1}

    # si hay varios resultados de búsqueda inteligente, listar títulos y pedir especificar
    lista = [ _titulo_corto(it) for it in tramites_encontrados ]
    texto = "Encontré varios trámites que pueden coincidir:\n\n" + "\n".join(lista) + "\n\nPor favor indica cuál deseas consultar."
    return {"respuesta": texto, "session_id": session_id, "candidatos": len(lista)}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)