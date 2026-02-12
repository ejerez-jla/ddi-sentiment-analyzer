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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CELDA 3: Configurar token de Hugging Face
# ============================================================================
# OPCIÓN 1: Usar variable de entorno (recomendado)
# HF_TOKEN = os.getenv('HF_TOKEN')

# OPCIÓN 2: Ingresar manualmente (para Colab)
from getpass import getpass
HF_TOKEN = getpass('Ingresa tu Hugging Face token: ')

os.environ["HF_TOKEN"] = HF_TOKEN
print("✅ Token configurado")

# ============================================================================
# CELDA 4: Cargar modelo RoBERTuito V2
# ============================================================================
print("🤖 Cargando modelo RoBERTuito V2...")
print("   (Esto puede tomar 1-2 minutos)")

MODEL_ID = "ejerez003/robertuito-guatemala-v2.0"

classifier = pipeline(
    'text-classification',
    model=MODEL_ID,
    token=HF_TOKEN,
    device=0  # GPU si está disponible, sino CPU
)

print("✅ Modelo cargado exitosamente")

# Mapping: Español → Inglés
LABEL_MAPPING = {
    "positivo": "positive",
    "negativo": "negative",
    "neutro": "neutral"
}

# ============================================================================
# CELDA 5: Crear API Flask
# ============================================================================
app = Flask(__name__)
CORS(app)  # Permitir requests desde cualquier origen

@app.route('/health', methods=['GET'])
def health():
    """Endpoint para verificar que el servidor está activo"""
    return jsonify({"status": "ok", "model": MODEL_ID})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint principal para análisis de sentimiento.
    
    Request JSON:
    {
        "texts": ["comentario 1", "comentario 2", ...]
    }
    
    Response JSON:
    {
        "results": [
            {"sentiment": "positive", "confidence": 0.95},
            {"sentiment": "negative", "confidence": 0.87},
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
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
        
        logger.info(f"✅ {len(results)} comentarios procesados")
        return jsonify({"results": results})
        
    except Exception as e:
        logger.error(f"Error en /analyze: {e}")
        return jsonify({"error": str(e)}), 500

print("✅ API Flask configurada")

# ============================================================================
# CELDA 6: Exponer API con ngrok
# ============================================================================
from pyngrok import ngrok

# Configurar authtoken de ngrok (OPCIONAL - para URLs estables)
# ngrok.set_auth_token("TU_NGROK_TOKEN")  # Obtener gratis en https://ngrok.com

# Iniciar túnel
public_url = ngrok.connect(5000)
print("\n" + "="*70)
print("🌐 API PÚBLICA DISPONIBLE")
print("="*70)
print(f"URL: {public_url}")
print(f"Health check: {public_url}/health")
print(f"Analyze endpoint: {public_url}/analyze")
print("="*70)
print("\n⚠️  IMPORTANTE: Copia esta URL y configúrala en la app de Streamlit")
print("\n🔄 El servidor se ejecutará hasta que detengas esta celda o Colab se desconecte")
print("="*70 + "\n")

# ============================================================================
# CELDA 7: Iniciar servidor Flask
# ============================================================================
# NOTA: Esta celda se ejecutará indefinidamente hasta que la detengas manualmente
app.run(port=5000)
