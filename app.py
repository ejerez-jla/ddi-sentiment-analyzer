import streamlit as st
import pandas as pd
import io
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importar módulos locales
from processing.sentiment import SentimentAnalyzer
from components.visualizer import (
    plot_sentiment_distribution,
    plot_confusion_matrix,
    plot_comparison_bars
)

# Configuración de la página
st.set_page_config(
    page_title="DDI Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# --- ESTILO PERSONALIZADO DDI ---
ddi_blue = "#142B5F"
ddi_gold = "#D5AB3E"
ddi_teal = "#59ADA8"
ddi_navy = "#052649"

st.markdown(f"""
    <style>
        /* Sidebar: Fondo Azul Marino DDI */
        [data-testid="stSidebar"] {{
            background-color: {ddi_navy};
        }}
        /* Sidebar: Textos en Blanco */
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: white !important;
        }}
        
        /* Títulos Principales: Azul DDI */
        h1, h2, h3 {{
            color: {ddi_blue} !important;
        }}
        
        /* Botones: Dorado DDI */
        div.stButton > button:first-child {{
            background-color: {ddi_gold} !important;
            color: white !important;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }}
        div.stButton > button:first-child:hover {{
            background-color: #B58B2E !important; /* Dorado más oscuro al hover */
        }}
        
        /* Barras de Progreso: Teal DDI */
        .stProgress > div > div > div > div {{
            background-color: {ddi_teal} !important;
        }}
    </style>
""", unsafe_allow_html=True)

# Estilos CSS
st.markdown("""
<style>
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: {ddi_blue};
        text-align: center;
        margin-bottom: 1rem;
    }}
    .metric-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">DDI Sentiment Analyzer</p>', unsafe_allow_html=True)
st.markdown(f"<div style='text-align: center; color: gray;'>Modelo: RoBERTuito V2.0 (Fine-tuned para Guatemala)</div>", unsafe_allow_html=True)

# Sidebar - Configuración
with st.sidebar:
    # Logo DDI en la parte superior
    logo_url = "https://ddilatam.com/wp-content/uploads/2024/08/DDI-LOGO-COLOR-OK-09_0064002e0_2244.png"
    st.image(logo_url, use_column_width=True)
    
    st.header("⚙️ Configuración")
    
    st.markdown("### 🌐 URL del API (Colab)")
    
    # Instrucciones con estilo personalizado (Blanco para contraste con fondo Navy)
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 5px; font-size: 0.9em;'>
        <strong>Instrucciones:</strong>
        <ol style='margin-left: -20px;'>
            <li>Abre el notebook <code>DDI_Sentiment_API_Colab.ipynb</code> en Google Colab</li>
            <li>Ejecuta todas las celdas</li>
            <li>Copia la URL pública generada (ej: <code>https://xxxx.ngrok.io</code>)</li>
            <li>Pégala abajo</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    api_url = st.text_input(
        "URL del API",
        placeholder="https://xxxx.ngrok.io",
        help="URL pública del notebook de Colab"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Opciones de Análisis")
    use_sentiment = st.checkbox("Análisis de Sentimiento V2", value=True, disabled=True)
    
    st.markdown("---")
    st.markdown("### ℹ️ Información")
    st.caption("Versión: 2.0.0")
    st.caption("Modelo: accesosddi/Sentimiento2")

# Main content
st.markdown("### 📂 Cargar Archivo")
st.markdown("Sube un archivo Excel o CSV con las columnas **`Comentario`** y **`sentiment`** (original)")

uploaded_file = st.file_uploader(
    "Selecciona tu archivo",
    type=['csv', 'xlsx', 'xls'],
    help="El archivo debe contener una columna 'Comentario' con el texto y 'sentiment' con la etiqueta original (-5, 0, 5)"
)

if uploaded_file:
    try:
        # Leer archivo
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
        
        # Validar columnas requeridas
        if 'Comentario' not in df.columns:
            st.error("❌ El archivo debe contener una columna llamada 'Comentario'")
            st.stop()
        
        # Buscar columna de sentimiento original
        sentiment_col = None
        for col in ['sentiment', 'Sentiment', 'sentimiento', 'Sentimiento']:
            if col in df.columns:
                sentiment_col = col
                break
        
        if not sentiment_col:
            st.warning("⚠️ No se encontró columna de sentimiento original. Se procesará sin comparación.")
            has_original = False
        else:
            st.info(f"📊 Columna de sentimiento original detectada: `{sentiment_col}`")
            has_original = True
        
        # Mostrar preview
        with st.expander("👀 Vista previa del archivo"):
            st.dataframe(df.head(10))
        
        # Botón para procesar
        if st.button("🚀 Analizar Sentimientos", type="primary"):
            if not api_url:
                st.error("❌ Debes configurar la URL del API en la barra lateral")
                st.stop()
            
            # Convertir sentimiento original a labels si existe
            if has_original:
                def convert_numeric_sentiment(val):
                    """Convierte escala numérica a labels"""
                    try:
                        num = float(val)
                        if num < 0:
                            return 'negativo'
                        elif num > 0:
                            return 'positivo'
                        else:
                            return 'neutro'
                    except:
                        return 'neutro'
                
                df['sentiment_original'] = df[sentiment_col].apply(convert_numeric_sentiment)
                st.success(f"✅ Sentimiento original convertido: {df['sentiment_original'].value_counts().to_dict()}")
            
            # Procesar con RoBERTuito V2
            st.markdown("---")
            st.subheader("🤖 Procesando con RoBERTuito V2...")
            
            analyzer = SentimentAnalyzer(api_url=api_url)
            result_df = analyzer.analyze(df)
            
            # Traducir resultados del API a Español
            translation_map = {
                'positive': 'positivo',
                'negative': 'negativo',
                'neutral': 'neutro',
                'error': 'error'
            }
            if 'sentiment' in result_df.columns:
                result_df['sentiment'] = result_df['sentiment'].map(lambda x: translation_map.get(x, x))
            
            if 'sentiment' not in result_df.columns:
                st.error("❌ Error en el procesamiento. Verifica que el API esté funcionando.")
                st.stop()
            
            st.success("✅ Análisis completado!")
            
            # Dashboard de Resultados
            st.markdown("---")
            st.subheader("📊 Resultados del Análisis")
            
            # Si hay sentimiento original, mostrar comparación
            if has_original and 'sentiment_original' in result_df.columns:
                st.markdown("### 🔍 Evaluación: Original vs V2")
                
                # Calcular métricas
                y_true = result_df['sentiment_original'].values
                y_pred = result_df['sentiment'].values
                
                # Filtrar errores
                valid_mask = (y_pred != 'error')
                y_true_valid = y_true[valid_mask].astype(str)
                y_pred_valid = y_pred[valid_mask].astype(str)
                
                if len(y_true_valid) > 0:
                    accuracy = accuracy_score(y_true_valid, y_pred_valid)
                    precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    
                    # Mostrar métricas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2%}")
                    with col2:
                        st.metric("Precision", f"{precision:.2%}")
                    with col3:
                        st.metric("Recall", f"{recall:.2%}")
                    with col4:
                        st.metric("F1-Score", f"{f1:.2%}")
                    
                    # Gráficos de comparación
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cm = plot_confusion_matrix(y_true_valid, y_pred_valid)
                        if fig_cm:
                            st.plotly_chart(fig_cm, use_container_width=True)
                    
                    with col2:
                        fig_comp = plot_comparison_bars(result_df)
                        if fig_comp:
                            st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning("⚠️ No hay predicciones válidas para calcular métricas")
            
            else:
                # Solo mostrar distribución V2
                st.markdown("### 📈 Distribución de Sentimientos V2")
                fig_sent = plot_sentiment_distribution(result_df)
                if fig_sent:
                    st.plotly_chart(fig_sent, use_container_width=True)
            
            # Tabla de resultados
            st.markdown("---")
            st.subheader("📋 Datos Procesados")
            
            # Reordenar columnas para mostrar comparación
            if has_original and 'sentiment_original' in result_df.columns:
                # Poner columnas de sentimiento juntas
                cols = result_df.columns.tolist()
                if 'sentiment_original' in cols and 'sentiment' in cols:
                    cols.remove('sentiment_original')
                    sent_idx = cols.index('sentiment')
                    cols.insert(sent_idx, 'sentiment_original')
                    result_df = result_df[cols]
            
            # Aplicar estilo
            def highlight_v2_columns(df):
                """Aplica fondo amarillo a columnas V2"""
                v2_cols = ['sentiment', 'confidence']
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for col in v2_cols:
                    if col in df.columns:
                        styles[col] = 'background-color: #fff9c4'
                return styles
            
            styled_df = result_df.style.apply(highlight_v2_columns, axis=None)
            st.dataframe(styled_df, use_container_width=True)
            
            # Descarga
            st.markdown("---")
            st.subheader("📥 Descargar Resultados")
            
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV con Resultados",
                data=csv,
                file_name=f"analisis_ddi_{uploaded_file.name.split('.')[0]}_v2.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error procesando el archivo: {e}")
        st.exception(e)

else:
    st.info("👆 Sube un archivo para comenzar el análisis")
