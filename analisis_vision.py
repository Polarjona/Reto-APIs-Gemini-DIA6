import os
import json
import time
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

rutas_imagenes = [
    "fotos/img1.jpg", 
    "fotos/img2.jpg", 
    "fotos/img3.jpg", 
    "fotos/img4.jpg", 
    "fotos/img5.jpg"
]

prompt_vision = """
Analiza esta imagen y extrae la información solicitada. 
Devuelve el resultado usando este esquema JSON:
{
    "ocr": "Texto exacto extraído de la imagen (o 'Ninguno' si no hay)",
    "descripcion": "Descripción detallada del contenido visual (1 oración)",
    "escena": "Categoría de la escena (ej. Documento, Exterior, Producto, Retrato)",
    "objetos_detectados": ["Objeto A", "Objeto B", "Objeto C"]
}
"""

resultados = []
pausa_entre_solicitudes = 15
max_intentos = 3

print("Procesando imágenes...")
for indice, ruta in enumerate(rutas_imagenes):
    if not os.path.exists(ruta):
        print(f"Imagen no encontrada: {ruta}")
        continue

    for intento in range(1, max_intentos + 1):
        try:
            img = Image.open(ruta)
            response = client.models.generate_content(
                model='gemini-3.5-flash',
                contents=[prompt_vision, img],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            datos = json.loads(response.text)
            datos['archivo'] = ruta
            resultados.append(datos)
            print(f"Imagen {ruta} procesada")
            break
        except Exception as e:
            mensaje = str(e)
            es_reintentable = "429" in mensaje or "503" in mensaje or "UNAVAILABLE" in mensaje
            if not es_reintentable or intento == max_intentos:
                print(f"Error en imagen {ruta}: {e}")
                break

            espera = 45 if "429" in mensaje else 15
            print(f"Reintentando {ruta} en {espera} segundos ({intento}/{max_intentos})")
            time.sleep(espera)

    if indice < len(rutas_imagenes) - 1:
        time.sleep(pausa_entre_solicitudes)

# Guardamos los resultados
with open("resultados_vision.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=4)
print("Resultados guardados en 'resultados_vision.json'")