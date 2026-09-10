import sys
import re
import json
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


# =====================================================================
# 1. ESQUEMAS PYDANTIC V2 PARA VALIDACIÓN
# =====================================================================

class InventarioInput(BaseModel):
    nombre_item: str = Field(description="Nombre o término de búsqueda del producto. Ejemplo: 'teclado mecanico'")

class DescuentoInput(BaseModel):
    precio_original: float = Field(description="Precio base del producto en formato numérico (ej. 100.0)")
    porcentaje_descuento: float = Field(description="Porcentaje a descontar entre 0 y 100 (ej. 15.0)")


# =====================================================================
# 2. DEFINICIÓN DE HERRAMIENTAS (CUSTOM TOOLS)
# =====================================================================

@tool(args_schema=InventarioInput)
def consultar_inventario_tienda(nombre_item: str) -> str:
    """Consulta la disponibilidad de stock y ubicación de un artículo en la tienda."""
    inventario = {
        "teclado mecanico": {"stock": 14, "pasillo": "A-3"},
        "mouse inalambrico": {"stock": 0, "pasillo": "A-1"},
        "monitor 27 pulgadas": {"stock": 5, "pasillo": "B-2"},
        "auriculares bluetooth": {"stock": 22, "pasillo": "C-4"}
    }
    
    query = str(nombre_item).lower().strip()
    
    for prod, datos in inventario.items():
        if prod in query or query in prod:
            if datos["stock"] > 0:
                return f"El producto '{prod}' SI tiene stock: {datos['stock']} unidades disponibles en Pasillo {datos['pasillo']}."
            return f"El producto '{prod}' está actualmente AGOTADO."
            
    return f"El producto '{nombre_item}' no existe en el catálogo de la tienda."


@tool(args_schema=DescuentoInput)
def calcular_descuento_producto(precio_original: float, porcentaje_descuento: float) -> str:
    """Calcula el precio final de un producto aplicando un porcentaje de descuento."""
    try:
        descuento = precio_original * (porcentaje_descuento / 100.0)
        precio_final = precio_original - descuento
        
        # SINTAXIS CORREGIDA: Uso del carácter '|' sin comandos LaTeX
        return f"Precio original: ${precio_original:.2f} | Descuento: ${descuento:.2f} | Total a pagar: ${precio_final:.2f}"
    except Exception as e:
        return f"Error al calcular el descuento: {str(e)}"


herramientas = [consultar_inventario_tienda, calcular_descuento_producto]
mapa_herramientas = {t.name: t for t in herramientas}


# =====================================================================
# 3. EXTRACTOR DEFENSIVO (FALLBACK PARA LLMS PEQUEÑOS)
# =====================================================================

def extraer_tool_call_manual(texto: str, consulta_usuario: str) -> Dict[str, Any]:
    """Fallback: Extrae la intención de llamada a herramienta si el modelo de 1B
    escribió la llamada en texto plano en lugar de usar la API nativa.
    """
    for nombre_tool in mapa_herramientas.keys():
        if nombre_tool in texto:
            if nombre_tool == "consultar_inventario_tienda":
                match = re.search(r'"nombre_item"\s*:\s*"([^"]+)"', texto)
                item = match.group(1) if match else consulta_usuario
                return {"tool_name": nombre_tool, "args": {"nombre_item": item}}
                
            elif nombre_tool == "calcular_descuento_producto":
                numeros = re.findall(r'\d+(?:\.\d+)?', texto)
                if len(numeros) >= 2:
                    return {
                        "tool_name": nombre_tool, 
                        "args": {"precio_original": float(numeros[0]), "porcentaje_descuento": float(numeros[1])}
                    }
    return {}


# =====================================================================
# 4. BUCLE DE EJECUCIÓN DEL AGENTE
# =====================================================================

def ejecutar_agente(pregunta_usuario: str):
    print(f"\n=======================================================")
    print(f" Pregunta: '{pregunta_usuario}'")
    print(f"=======================================================")
    
    # Inicialización del LLM local con Ollama
    llm = ChatOllama(
        model="qwen2.5:7b",
        temperature=0.0
    )
    
    llm_con_tools = llm.bind_tools(herramientas)
    
    mensajes = [
        SystemMessage(content=(
            "Eres un asistente virtual de tienda de tecnología. "
            "Responde a los usuarios amablemente. "
            "Cuenta chistes malos en cada iteracion de tecnologia"
            "Usa la herramienta 'consultar_inventario_tienda' para preguntas de stock. "
            "Usa la herramienta 'calcular_descuento_producto' para preguntas de precios."
        )),
        HumanMessage(content=pregunta_usuario)
    ]
    
    respuesta = llm_con_tools.invoke(mensajes)
    mensajes.append(respuesta)
    
    # Capa 1: Tool Calls Nativos
    if hasattr(respuesta, 'tool_calls') and respuesta.tool_calls:
        print(f"\n[DECISIÓN DEL AGENTE]: Invocación Nativa ({len(respuesta.tool_calls)} herramienta/s).")
        
        for call in respuesta.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            tool_id = call["id"]
            
            print(f"  └─ Ejecutando Tool: '{tool_name}' con Args: {tool_args}")
            
            if tool_name in mapa_herramientas:
                resultado = mapa_herramientas[tool_name].invoke(tool_args)
            else:
                resultado = f"Herramienta '{tool_name}' no encontrada."
                
            print(f"  └─ Observación: {resultado}")
            mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_id))
            
        respuesta_final = llm_con_tools.invoke(mensajes)
        print(f"\n[RESPUESTA FINAL]:\n{respuesta_final.content}")
        
    else:
        # Capa 2: Fallback por Regex si el modelo escribió la llamada en texto plano
        fallback_call = extraer_tool_call_manual(respuesta.content, pregunta_usuario)
        
        if fallback_call:
            tool_name = fallback_call["tool_name"]
            tool_args = fallback_call["args"]
            
            print(f"\n[DECISIÓN DEL AGENTE - RECUPERADA POR FALLBACK]:")
            print(f"  └─ Ejecutando Tool: '{tool_name}' con Args: {tool_args}")
            
            resultado = mapa_herramientas[tool_name].invoke(tool_args)
            print(f"  └─ Observación: {resultado}")
            
            mensajes.append(HumanMessage(content=f"Resultado de la consulta: {resultado}. Ahora responde al usuario amablemente."))
            respuesta_final = llm_con_tools.invoke(mensajes)
            print(f"\n[RESPUESTA FINAL]:\n{respuesta_final.content}")
        else:
            print(f"\n[DECISIÓN DEL AGENTE]: Respuesta Directa.")
            print(f"\n[RESPUESTA FINAL]:\n{respuesta.content}")


# =====================================================================
# 5. PUNTO DE ENTRADA
# =====================================================================

if __name__ == "__main__":
    print("Agente de tienda listo. Escribe 'salir' para terminar.\n")
    while True:
        pregunta = input("Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("¡Hasta luego!")
            break
        if not pregunta:
            continue
        ejecutar_agente(pregunta)