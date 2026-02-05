import streamlit as st
import pandas as pd
import io
import time

# Importar módulos locales
from processing.sentiment import SentimentAnalyzer
from processing.topics import TopicDetector
from components.visualizer import plot_sentiment_distribution, plot_topic_distribution

# Configuración de la página
st.set_page_config(
    page_title="DDI Analytics - Sentimiento & Tópicos",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size: 1.5rem; color: #4B5563; }
</style>
""", unsafe_allow_html=True)

# Lazy loading - no pre-cargar modelos para ahorrar memoria
# Los modelos se cargarán solo cuando el usuario haga click en "Iniciar Análisis"

def convert_df(df):
    """Convierte DataFrame a CSV para descarga."""
    return df.to_csv(index=False).encode('utf-8')

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
        st.title("DDI Analytics")
        st.info("Herramienta de IA para análisis de sentimiento y tópicos.")
        
        st.divider()
        st.subheader("Configuración")
        use_sentiment = st.checkbox("Analizar Sentimiento", value=True)
        use_topics = st.checkbox("Detectar Tópicos", value=False, disabled=True, 
                                  help="⚠️ Deshabilitado temporalmente: El modelo requiere >1GB de RAM (límite de Streamlit Free Tier)")
        
        if use_topics:
            st.warning("⚠️ La detección de tópicos requiere recursos adicionales. Considera usar solo análisis de sentimiento.")
        
        st.divider()
        st.caption("v1.0.0 | Powered by RoBERTuito & ZeroShot")

    # Header
    st.markdown('<div class="main-header">Motor de Inteligencia de Datos DDI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Carga tu archivo de comentarios para comenzar.</div>', unsafe_allow_html=True)
    st.divider()

    # Carga de archivo
    uploaded_file = st.file_uploader("Sube un archivo Excel (.xlsx) o CSV", type=['xlsx', 'csv'])

    if uploaded_file is not None:
        try:
            # Detectar formato
            if uploaded_file.name.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(uploaded_file)
                except:
                   df = pd.read_csv(uploaded_file, encoding='latin-1') 
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validación
            if 'Comentario' not in df.columns:
                st.error("❌ El archivo NO contiene la columna obligatoria 'Comentario'.")
                st.dataframe(df.head())
                return

            st.success(f"✅ Archivo cargado: {uploaded_file.name} ({len(df)} filas)")
            
            with st.expander("👀 Vista previa de datos", expanded=False):
                st.dataframe(df.head())

            # Botón de Procesar
            if st.button("🚀 Iniciar Análisis", type="primary"):
                
                start_time = time.time()
                result_df = df.copy()

                # 1. Análisis de Sentimiento (lazy loading)
                if use_sentiment:
                    with st.spinner('🤖 Cargando modelo de sentimiento...'):
                        analyzer = SentimentAnalyzer()
                    with st.spinner('🤖 Analizando Sentimientos (RoBERTuito v2.0)...'):
                        result_df = analyzer.analyze(result_df)
                    # Liberar memoria
                    del analyzer

                # 2. Detección de Tópicos (lazy loading)
                if use_topics:
                    with st.spinner('🧠 Cargando modelo de tópicos...'):
                        detector = TopicDetector()
                    with st.spinner('🧠 Detectando Tópicos (Zero-Shot)...'):
                        result_df = detector.detect(result_df)
                    # Liberar memoria
                    del detector

                duration = time.time() - start_time
                st.success(f"🎉 Procesamiento completado en {duration:.1f} segundos!")

                # Resultados Visuales
                st.divider()
                st.subheader("📊 Dashboard de Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if use_sentiment:
                        fig_sent = plot_sentiment_distribution(result_df)
                        if fig_sent: st.plotly_chart(fig_sent, use_container_width=True)
                
                with col2:
                    if use_topics:
                        fig_topic = plot_topic_distribution(result_df)
                        if fig_topic: st.plotly_chart(fig_topic, use_container_width=True)

                # Tabla de Resultados
                st.subheader("📋 Datos Procesados")
                
                # Detectar si hay columna de sentimiento original
                sentiment_cols = ['sentiment', 'sentimiento', 'Sentiment', 'Sentimiento']
                original_sentiment_col = None
                for col in sentiment_cols:
                    if col in df.columns and col != 'sentiment':  # Evitar la columna que acabamos de crear
                        original_sentiment_col = col
                        break
                
                # Si hay sentimiento original, convertir escala numérica a labels
                if original_sentiment_col:
                    def convert_numeric_sentiment(val):
                        """Convierte escala -5 a 5 en labels"""
                        try:
                            num = float(val)
                            if num < -1:
                                return 'negative'
                            elif num > 1:
                                return 'positive'
                            else:
                                return 'neutral'
                        except:
                            return 'unknown'
                    
                    result_df['sentiment_original'] = df[original_sentiment_col].apply(convert_numeric_sentiment)
                    
                    # Reordenar columnas para poner comparación lado a lado
                    cols = result_df.columns.tolist()
                    # Buscar índice de 'sentiment' (modelo V2)
                    if 'sentiment' in cols and 'sentiment_original' in cols:
                        sent_idx = cols.index('sentiment')
                        cols.remove('sentiment_original')
                        cols.insert(sent_idx, 'sentiment_original')
                        result_df = result_df[cols]
                
                # Aplicar estilo con fondo amarillo a columnas del modelo V2
                def highlight_v2_columns(df):
                    """Aplica fondo amarillo claro a columnas del modelo V2"""
                    v2_cols = ['sentiment', 'confidence']
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)
                    for col in v2_cols:
                        if col in df.columns:
                            styles[col] = 'background-color: #fff9c4'  # Amarillo claro
                    return styles
                
                styled_df = result_df.style.apply(highlight_v2_columns, axis=None)
                st.dataframe(styled_df, use_container_width=True)

                # Descarga
                csv = convert_df(result_df)
                st.download_button(
                    label="📥 Descargar CSV con Resultados",
                    data=csv,
                    file_name=f"analisis_ddi_{uploaded_file.name.split('.')[0]}.csv",
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")

if __name__ == "__main__":
    main()
