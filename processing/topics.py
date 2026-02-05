import pandas as pd
from transformers import pipeline
import streamlit as st
import torch

class TopicDetector:
    def __init__(self, model_id="facebook/bart-large-mnli"):
        self.model_id = model_id
        self.classifier = None
        self.device = 0 if torch.cuda.is_available() else -1
        self.default_topics = [
            "Servicio al Cliente",
            "Producto/Calidad",
            "Precios",
            "Seguridad",
            "Promociones",
            "Tecnología/Apps",
            "Entrega/Disponibilidad",
            "Reclamaciones"
        ]

    @st.cache_resource(show_spinner=False)
    def load_model(_self):
        """Carga el modelo Zero Shot y lo cachea."""
        try:
            classifier = pipeline(
                "zero-shot-classification",
                model=_self.model_id,
                device=_self.device
            )
            return classifier
        except Exception as e:
            st.error(f"Error cargando el modelo de tópicos: {e}")
            return None

    def detect(self, df, text_column='Comentario', candidate_labels=None):
        """Detecta tópicos usando Zero Shot Classification."""
        
        self.classifier = self.load_model()
        if not self.classifier:
            return df

        if candidate_labels is None:
            candidate_labels = self.default_topics

        topics = []
        scores = []
        
        total = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, text in enumerate(df[text_column]):
            try:
                # Predicción
                result = self.classifier(
                    str(text),
                    candidate_labels,
                    multi_label=False # Un solo tópico principal
                )
                
                # El top 1 es el primero
                top_topic = result['labels'][0]
                top_score = result['scores'][0]

                topics.append(top_topic)
                scores.append(top_score)

                # Actualizar progreso
                if (i + 1) % 10 == 0 or (i + 1) == total:
                    progress_bar.progress((i + 1) / total)
                    status_text.text(f"Detectando tópicos: {i + 1}/{total}")

            except Exception:
                topics.append("Desconocido")
                scores.append(0.0)

        df['topic'] = topics
        df['topic_confidence'] = scores
        
        status_text.empty()
        progress_bar.empty()
        
        return df
