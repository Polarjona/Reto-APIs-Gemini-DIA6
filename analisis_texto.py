import os
import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

textos_ejemplo = [
    "URGENT: Your account has been compromised. Please reset your password immediately.",
    "El nuevo restaurante de Madrid tiene una comida increíble, pero el servicio fue pésimo.",
    "Apple is expected to announce the new iPhone 16 in September at their California headquarters.",
    "Estimado equipo, la reunión trimestral se pospone al próximo martes a las 10:00 AM.",
    "La inteligencia artificial generativa está transformando rápidamente el sector tecnológico global.",
    "Me encantó la película Dune 2, los efectos especiales fueron alucinantes. ¡Totalmente recomendada!",
    "Flight BA245 to London is delayed by 2 hours due to severe weather conditions.",
    "El paquete que pedí en Amazon llegó roto y con 3 días de retraso. Exijo un reembolso total.",
    "Tesla's stock price surged today after Elon Musk announced a new gigafactory in Texas.",
    "Por favor, revisa el informe de ventas Q3 adjunto antes de la presentación de mañana con el cliente."
]

prompt_texto = """
Analiza el texto proporcionado y extrae la información solicitada.
Usa el siguiente esquema JSON:
{
    "idioma": "Idioma detectado (ej. Español, Inglés)",
    "entidades": ["Entidad 1", "Entidad 2"],
    "sentimiento": "Positivo, Negativo o Neutral",
    "resumen": "Resumen de máximo 10 palabras"
}
Texto: 
"""

resultados = []
pausa_entre_solicitudes = 15
max_intentos = 3
archivo_resultados = "resultados_texto.json"

if os.path.exists(archivo_resultados):
    try:
        with open(archivo_resultados, "r", encoding="utf-8") as f:
            resultados = json.load(f)
        print(f"Resultados existentes cargados: {len(resultados)}")
    except (json.JSONDecodeError, OSError):
        resultados = []

print("Procesando textos...")
for i, texto in enumerate(textos_ejemplo):
    if any(resultado.get("texto_original") == texto for resultado in resultados):
        print(f"Texto {i+1} ya estaba procesado")
        continue

    for intento in range(1, max_intentos + 1):
        try:
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=prompt_texto + texto,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            datos = json.loads(response.text)
            datos['texto_original'] = texto
            resultados.append(datos)
            print(f"Texto {i+1} procesado")
            break
        except Exception as e:
            mensaje = str(e)
            es_reintentable = "429" in mensaje or "503" in mensaje or "UNAVAILABLE" in mensaje
            if not es_reintentable or intento == max_intentos:
                print(f"Error en texto {i+1}: {e}")
                break

            espera = 45 if "429" in mensaje else 15 * (2 ** (intento - 1))
            print(f"Reintentando texto {i+1} en {espera} segundos ({intento}/{max_intentos})")
            time.sleep(espera)

    if i < len(textos_ejemplo) - 1:
        time.sleep(pausa_entre_solicitudes)

# Guardamos los resultados para que el Notebook los pueda leer
with open(archivo_resultados, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=4)
print("Resultados guardados en 'resultados_texto.json'")