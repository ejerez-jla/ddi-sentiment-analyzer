import requests
import streamlit as st
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Cliente para el API de análisis de sentimiento en Colab"""
    
    def __init__(self, api_url=None):
        """
        Inicializa el cliente del API.
        
        Args:
            api_url: URL del endpoint de Colab (ej: https://xxxx.ngrok.io)
        """
        self.api_url = api_url
        
    def analyze(self, df):
        """
        Analiza sentimientos usando el API de Colab.
        
        Args:
            df: DataFrame con columna 'Comentario'
            
        Returns:
            DataFrame con columnas 'sentiment' y 'confidence' agregadas
        """
        if not self.api_url:
            st.error("❌ URL del API no configurada. Ver instrucciones en la barra lateral.")
            return df
            
        # Validar que el API está disponible
        try:
            health_response = requests.get(f"{self.api_url}/health", timeout=5)
            if health_response.status_code != 200:
                st.error(f"❌ El API no responde correctamente: {health_response.status_code}")
                return df
        except Exception as e:
            st.error(f"❌ No se puede conectar al API: {e}")
            st.info("💡 Asegúrate de que el notebook de Colab esté ejecutándose y la URL sea correcta.")
            return df
        
        # Preparar textos
        texts = df['Comentario'].fillna("").astype(str).tolist()
        
        # Llamar al API en batches (para evitar timeouts)
        batch_size = 20
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            status_text.text(f"Procesando {i+1}-{min(i+batch_size, len(texts))} de {len(texts)} comentarios...")
            
            try:
                response = requests.post(
                    f"{self.api_url}/analyze",
                    json={"texts": batch},
                    timeout=120  # 2 minutos max por batch
                )
                
                if response.status_code == 200:
                    results = response.json()['results']
                    all_results.extend(results)
                else:
                    st.error(f"❌ Error en batch {i//batch_size + 1}: {response.status_code}")
                    # Agregar resultados vacíos para este batch
                    all_results.extend([{"sentiment": "error", "confidence": 0.0}] * len(batch))
                    
            except Exception as e:
                logger.error(f"Error procesando batch {i//batch_size + 1}: {e}")
                st.error(f"❌ Error procesando batch {i//batch_size + 1}: {e}")
                all_results.extend([{"sentiment": "error", "confidence": 0.0}] * len(batch))
            
            progress_value = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.empty()
        
        # Agregar resultados al DataFrame
        df['sentiment'] = [r['sentiment'] for r in all_results]
        df['confidence'] = [r['confidence'] for r in all_results]
        
        return df
