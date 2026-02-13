# DDI Sentiment Analyzer - API Backend (Colab)
# Este script carga el modelo RoBERTuito V2 y expone una API REST para análisis de sentimiento.
#
# Instrucciones:
# 1. Subir este archivo a Google Colab
# 2. Ejecutar todas las celdas en orden
# 3. Copiar la URL pública generada por ngrok
# 4. Configurar esa URL en la app de Streamlit
# 5. Los analistas pueden usar la app web para procesar archivos

# ============================================================================
# CELDA 1: Instalar dependencias
# ============================================================================
!pip install -q transformers pysentimiento flask flask-cors pyngrok

# ============================================================================
# CELDA 2: Imports
# ============================================================================
import os
from transformers import pipeline
from pysentimiento.preprocessing import preprocess_tweet
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from pyngrok import ngrok
from getpass import getpass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CELDA 3: Configurar Tokens (Hugging Face + Ngrok)
# ============================================================================
print("CONFIGURACIÓN DE SEGURIDAD")
print("-" * 30)

# 1. Hugging Face Token (para descargar modelo)
# HF_TOKEN = os.getenv('HF_TOKEN') # Opción variable de entorno
print("\n1. Token de Hugging Face:")
HF_TOKEN = getpass('   Ingresa tu HF Token: ')
os.environ["HF_TOKEN"] = HF_TOKEN

# 2. Ngrok Authtoken (para túnel público)
# IMPORTANTE: No uses una "API Key". Usa el "Authtoken" que está en el menú izquierdo.
# Obtener aquí: https://dashboard.ngrok.com/get-started/your-authtoken
print("\n2. Authtoken de Ngrok (Obligatorio para Colab):")
print("   Ve a 'Your Authtoken' en el menú izquierdo de Ngrok.")
print("   URL directa: https://dashboard.ngrok.com/get-started/your-authtoken")
print("   (Debe empezar con '2o...')")
NGROK_TOKEN = getpass('   Ingresa tu Ngrok Authtoken: ')
ngrok.set_auth_token(NGROK_TOKEN)

print("\nCredenciales configuradas")

# ============================================================================
# CELDA 4: Cargar modelo RoBERTuito V2
# ============================================================================
print("\nCargando modelo RoBERTuito V2...")
print("   (Esto puede tomar 1-2 minutos)")

MODEL_ID = "ejerez003/robertuito-guatemala-v2.0"

try:
    classifier = pipeline(
        'text-classification',
        model=MODEL_ID,
        token=HF_TOKEN,
        device=0  # GPU si está disponible, sino CPU
    )
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error cargando modelo: {e}")
    print("Verifica que tu token de HF tenga permisos de lectura.")

# ... (omitting unchanged mapping and flask setup code) ...

        logger.info(f"Procesando {len(texts)} comentarios...")
        
        results = []
        for text in texts:
            try:
                # Preprocesar con pysentimiento
                text_prep = preprocess_tweet(str(text))
                
                # Predecir sentimiento
                prediction = classifier(text_prep, truncation=True, max_length=128)
                
                sentiment = prediction[0]['label']
                # Convertir a inglés
                sentiment = LABEL_MAPPING.get(sentiment, "neutral")
                confidence = round(prediction[0]['score'], 4)
                
                results.append({
                    "sentiment": sentiment,
                    "confidence": confidence
                })
            except Exception as e:
                logger.error(f"Error procesando texto: {e}")
                results.append({
                    "sentiment": "error",
                    "confidence": 0.0
                })
        
        logger.info(f"{len(results)} comentarios procesados")
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error en /analyze: {e}")
        return jsonify({"error": str(e)}), 500

print("API Flask configurada")

# ============================================================================
# CELDA 6: Exponer API con ngrok
# ============================================================================
# Cerrar túneles previos si existen
ngrok.kill()

# Iniciar túnel
try:
    public_url = ngrok.connect(5000)
    print("\n" + "="*70)
    print("API PÚBLICA DISPONIBLE")
    print("="*70)
    print(f"URL: {public_url.public_url}")
    print(f"Health check: {public_url.public_url}/health")
    print(f"Analyze endpoint: {public_url.public_url}/analyze")
    print("="*70)
    print("\nIMPORTANTE: Copia esta URL y configúrala en la app de Streamlit")
    print("\nEl servidor se ejecutará hasta que detengas esta celda o Colab se desconecte")
    print("="*70 + "\n")
except Exception as e:
    print(f"Error iniciando ngrok: {e}")

    print("Verifica que tu Authtoken sea correcto.")

# ============================================================================
# CELDA 7: Iniciar servidor Flask
# ============================================================================
# NOTA: Esta celda se ejecutará indefinidamente hasta que la detengas manualmente
app.run(port=5000)
