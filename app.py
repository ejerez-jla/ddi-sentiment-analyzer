"""
DDI Sentiment Analyzer - RoBERTuito V2.0
Aplicación Streamlit para análisis de sentimiento en español
Modelo: ejerez003/robertuito-guatemala-v2.0
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pysentimiento.preprocessing import preprocess_tweet
import pandas as pd
import torch
from datetime import datetime
import io

# ==================== CONFIGURACIÓN ====================
st.set_page_config(
    page_title="DDI Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CARGA DEL MODELO ====================
@st.cache_resource
def load_model():
    """
    Carga el modelo RoBERTuito V2.0 desde Hugging Face Hub
    Usa @st.cache_resource para cargar solo una vez y cachear
    """
    with st.spinner("🔄 Cargando modelo RoBERTuito V2.0 desde Hugging Face Hub..."):
        try:
            model_name = "ejerez003/robertuito-guatemala-v2.0"
            
            # Cargar tokenizer y modelo
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Configurar para CPU
            model.eval()
            
            st.success("✅ Modelo cargado exitosamente")
            return tokenizer, model
            
        except Exception as e:
            st.error(f"❌ Error al cargar el modelo: {str(e)}")
            st.stop()

# ==================== FUNCIONES DE PREDICCIÓN ====================
def predict_sentiment_batch(texts, tokenizer, model, batch_size=32):
    """
    Procesa textos en batches con preprocesamiento pysentimiento
    
    Args:
        texts: Lista de textos a analizar
        tokenizer: Tokenizer del modelo
        model: Modelo de clasificación
        batch_size: Tamaño del batch (default: 32)
    
    Returns:
        Lista de diccionarios con 'label' y 'score'
    """
    all_predictions = []
    
    # Crear progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
        # Actualizar progress bar
        progress = (batch_idx + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Procesando batch {batch_idx + 1}/{total_batches}...")
        
        # Extraer batch
        batch = texts[i:i+batch_size]
        
        # ⚠️ CRÍTICO: Preprocesar con pysentimiento
        # Esto normaliza URLs, menciones, emojis, etc.
        batch_preprocessed = [preprocess_tweet(text) for text in batch]
        
        # Tokenizar
        inputs = tokenizer(
            batch_preprocessed,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt'
        )
        
        # Predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1)
            confidence = torch.max(predictions, dim=-1)[0]
        
        # Extraer resultados
        for pred_id, conf in zip(predicted_class, confidence):
            label = model.config.id2label[pred_id.item()]
            all_predictions.append({
                'label': label,
                'score': conf.item()
            })
    
    # Limpiar progress bar
    progress_bar.empty()
    status_text.empty()
    
    return all_predictions

# ==================== PROCESAMIENTO DE ARCHIVO ====================
def process_file(uploaded_file, tokenizer, model):
    """
    Procesa archivo CSV/XLSX y retorna DataFrame con predicciones
    """
    try:
        # Leer archivo según extensión
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Formato no soportado. Use CSV o Excel (.xlsx, .xls)")
            return None
        
        # Validar columna de texto
        text_columns = ['texto', 'text', 'content', 'mensaje', 'comment', 'comentario']
        text_col = None
        
        for col in text_columns:
            if col.lower() in [c.lower() for c in df.columns]:
                text_col = [c for c in df.columns if c.lower() == col.lower()][0]
                break
        
        if text_col is None:
            st.error(f"❌ No se encontró columna de texto. Columnas esperadas: {', '.join(text_columns)}")
            st.info(f"📋 Columnas encontradas: {', '.join(df.columns)}")
            return None
        
        # Extraer textos
        texts = df[text_col].fillna("").astype(str).tolist()
        
        if len(texts) == 0:
            st.error("❌ No se encontraron textos para analizar")
            return None
        
        # Realizar predicciones
        st.info(f"🔄 Analizando {len(texts)} textos...")
        predictions = predict_sentiment_batch(texts, tokenizer, model)
        
        # Agregar predicciones al DataFrame
        df['sentiment_v2'] = [pred['label'] for pred in predictions]
        df['confidence_v2'] = [pred['score'] for pred in predictions]
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error al procesar archivo: {str(e)}")
        return None

# ==================== VISUALIZACIÓN DE RESULTADOS ====================
def show_results(df):
    """
    Muestra resultados y estadísticas
    """
    st.success(f"✅ Análisis completado: {len(df)} textos procesados")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    sentiment_counts = df['sentiment_v2'].value_counts()
    
    with col1:
        st.metric(
            "Total Textos",
            len(df)
        )
    
    with col2:
        pos_count = sentiment_counts.get('POS', 0)
        pos_pct = (pos_count / len(df)) * 100
        st.metric(
            "Positivos",
            f"{pos_count}",
            f"{pos_pct:.1f}%"
        )
    
    with col3:
        neu_count = sentiment_counts.get('NEU', 0)
        neu_pct = (neu_count / len(df)) * 100
        st.metric(
            "Neutrales",
            f"{neu_count}",
            f"{neu_pct:.1f}%"
        )
    
    with col4:
        neg_count = sentiment_counts.get('NEG', 0)
        neg_pct = (neg_count / len(df)) * 100
        st.metric(
            "Negativos",
            f"{neg_count}",
            f"{neg_pct:.1f}%"
        )
    
    # Distribución
    st.subheader("📊 Distribución de Sentimientos")
    st.bar_chart(sentiment_counts)
    
    # Tabla de resultados
    st.subheader("📋 Resultados Detallados")
    
    # Mostrar primeros 100 registros por defecto
    display_df = df.head(100)
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    if len(df) > 100:
        st.info(f"ℹ️ Mostrando primeros 100 de {len(df)} registros. Descarga el archivo completo abajo.")

# ==================== INTERFAZ PRINCIPAL ====================
def main():
    """
    Función principal de la aplicación
    """
    # Header
    st.title("🎯 DDI Sentiment Analyzer")
    st.markdown("### Análisis de Sentimiento con RoBERTuito V2.0")
    st.markdown("---")
    
    # Sidebar con información
    with st.sidebar:
        st.header("ℹ️ Información")
        st.markdown("""
        **Modelo:** RoBERTuito V2.0  
        **Precisión:** 82.65%  
        **Idioma:** Español (LATAM)
        
        **Categorías:**
        - 🟢 POS: Positivo
        - 🟡 NEU: Neutral
        - 🔴 NEG: Negativo
        
        **Formato de archivo:**
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        
        **Columna de texto requerida:**
        - texto / text / content
        - mensaje / comment / comentario
        """)
        
        st.markdown("---")
        st.markdown("**Desarrollado por:** JLA Consulting Group")
        st.markdown("**Cliente:** DDI Guatemala")
    
    # Cargar modelo
    tokenizer, model = load_model()
    
    # Upload file
    st.header("📤 Cargar Archivo")
    uploaded_file = st.file_uploader(
        "Selecciona un archivo CSV o Excel",
        type=['csv', 'xlsx', 'xls'],
        help="El archivo debe contener una columna con textos a analizar"
    )
    
    if uploaded_file is not None:
        # Mostrar información del archivo
        st.info(f"📄 Archivo cargado: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Procesar archivo
        with st.spinner("🔄 Procesando archivo..."):
            df_results = process_file(uploaded_file, tokenizer, model)
        
        if df_results is not None:
            # Mostrar resultados
            show_results(df_results)
            
            # Botón de descarga
            st.subheader("💾 Descargar Resultados")
            
            # Generar nombre de archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"sentiment_analysis_{timestamp}.xlsx"
            
            # Convertir a Excel en memoria
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Resultados')
            output.seek(0)
            
            st.download_button(
                label="📥 Descargar Excel con Resultados",
                data=output,
                file_name=output_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    else:
        # Instrucciones iniciales
        st.info("""
        👆 **Para comenzar:**
        1. Carga un archivo CSV o Excel
        2. Asegúrate que tenga una columna de texto llamada: `texto`, `text`, `content`, `mensaje`, `comment` o `comentario`
        3. El análisis comenzará automáticamente
        4. Descarga los resultados cuando estén listos
        """)
        
        # Ejemplo de formato
        with st.expander("📋 Ver ejemplo de formato esperado"):
            example_df = pd.DataFrame({
                'texto': [
                    'Me encanta este producto, es excelente!',
                    'No funciona bien, muy decepcionante',
                    'Es un producto normal, nada especial'
                ],
                'fecha': ['2025-01-01', '2025-01-02', '2025-01-03']
            })
            st.dataframe(example_df)

# ==================== EJECUCIÓN ====================
if __name__ == "__main__":
    main()
