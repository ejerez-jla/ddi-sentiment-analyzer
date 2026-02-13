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
        
        # Configurar sesión con reintentos automáticos
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Llamar al API en batches pequeños (para evitar timeouts y errores SSL de ngrok)
        batch_size = 10 
        all_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            current_batch = (i // batch_size) + 1
            status_text.text(f"Procesando lote {current_batch}/{total_batches} ({min(i+batch_size, len(texts))}/{len(texts)} comentarios)...")
            
            # Reintento manual para errores de conexión/SSL específicos
            max_retries = 3
            batch_success = False
            
            for attempt in range(max_retries):
                try:
                    response = session.post(
                        f"{self.api_url}/analyze",
                        json={"texts": batch},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        results = response.json().get('results', [])
                        if results:
                            all_results.extend(results)
                            batch_success = True
                            break
                    else:
                        logger.warning(f"Batch {current_batch} falló con status {response.status_code}. Reintento {attempt+1}/{max_retries}")
                        time.sleep(2 * (attempt + 1)) # Backoff lineal
                        
                except Exception as e:
                    logger.warning(f"Excepción en batch {current_batch} (intento {attempt+1}): {e}")
                    time.sleep(2 * (attempt + 1))
            
            if not batch_success:
                st.error(f"❌ Error definitivo en lote {current_batch}. Se omiten {len(batch)} comentarios.")
                all_results.extend([{"sentiment": "error", "confidence": 0.0}] * len(batch))
            
            # Pequeña pausa para no saturar ngrok
            time.sleep(0.5)
            
            progress_value = min((i + batch_size) / len(texts), 1.0)
            progress_bar.progress(progress_value)
        
        progress_bar.empty()
        status_text.empty()
        
        # Agregar resultados al DataFrame
        df['sentiment'] = [r['sentiment'] for r in all_results]
        df['confidence'] = [r['confidence'] for r in all_results]
        
        return df
