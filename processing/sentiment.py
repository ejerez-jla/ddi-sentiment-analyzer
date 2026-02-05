import pandas as pd
from transformers import pipeline
from pysentimiento.preprocessing import preprocess_tweet
import streamlit as st
import torch

class SentimentAnalyzer:
    def __init__(self, model_id="ejerez003/robertuito-guatemala-v2.0"):
        self.model_id = model_id
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1

    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Carga el modelo de Hugging Face y lo cachea."""
        try:
            classifier = pipeline(
                'text-classification',
                model=_self.model_id,
                device=_self.device
            )
            return classifier
        except Exception as e:
            st.error(f"Error cargando el modelo de sentimiento: {e}")
            return None

    def analyze(self, df, text_column='Comentario'):
        """Analiza el sentimiento de un DataFrame."""
        
        self.classifier = self.load_model()
        if not self.classifier:
            return df

        # Mapeo de resultados
        label_mapping = {
            "positivo": "positive",
            "negativo": "negative",
            "neutro": "neutral"
        }

        sentiments = []
        scores = []
        
        total = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, text in enumerate(df[text_column]):
            try:
                # Preprocesamiento
                text_prep = preprocess_tweet(str(text))
                
                # Predicción
                result = self.classifier(text_prep, truncation=True, max_length=128)
                
                label = result[0]['label']
                label = label_mapping.get(label, "neutral")
                score = result[0]['score']

                sentiments.append(label)
                scores.append(score)

                # Actualizar progreso cada 10 items o al final
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    progress_bar.progress((i + 1) / total)
                    status_text.text(f"Procesando sentimiento: {i + 1}/{total}")

            except Exception:
                sentiments.append("error")
                scores.append(0.0)

        df['sentiment'] = sentiments
        df['confidence'] = scores
        
        status_text.empty()
        progress_bar.empty()
        
        return df
