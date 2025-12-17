import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from pysentimiento.preprocessing import preprocess_tweet
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os
from openai import OpenAI

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="DDI Sentiment Analyzer V2.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def cargar_modelo_huggingface(model_name="ejerez003/robertuito-guatemala-v2.0"):
    """Carga el modelo RoBERTuito V2 desde Hugging Face Hub"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=-1)
        return classifier, tokenizer
    except Exception as e:
        st.error(f"Error al cargar modelo: {e}")
        return None, None

@st.cache_resource
def cargar_modelo_local(model_path):
    """Carga el modelo RoBERTuito V2 desde directorio local (solo para uso local)"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, device=-1)
        return classifier, tokenizer
    except Exception as e:
        st.error(f"Error al cargar modelo local: {e}")
        return None, None

def predecir_sentimiento_v2(textos, classifier, tokenizer, batch_size=32):
    """Predice sentimiento con Modelo V2"""
    resultados = []
    progress_bar = st.progress(0)
    
    for i in range(0, len(textos), batch_size):
        batch = textos[i:i+batch_size]
        batch_preprocesado = [preprocess_tweet(str(texto)) for texto in batch]
        
        # Tokenizar con truncación
        inputs = tokenizer(
            batch_preprocesado,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors='pt'
        )
        
        # Predicción
        with torch.no_grad():
            predictions = classifier(batch_preprocesado)
        
        resultados.extend(predictions)
        progress_bar.progress(min((i + batch_size) / len(textos), 1.0))
    
    return resultados

def generar_verdad_absoluta(textos, api_key, modelo="gpt-3.5-turbo", max_samples=None):
    """Genera verdad absoluta con OpenAI"""
    client = OpenAI(api_key=api_key)
    
    SYSTEM_PROMPT = """Eres un experto en análisis de sentimientos en español guatemalteco.
Clasifica cada texto en: positivo, negativo o neutro.
Considera sarcasmo, jerga guatemalteca y contexto empresarial.
Responde SOLO con una palabra: positivo, negativo o neutro"""
    
    if max_samples:
        textos = textos[:max_samples]
    
    resultados = []
    progress_bar = st.progress(0)
    
    for idx, texto in enumerate(textos):
        try:
            response = client.chat.completions.create(
                model=modelo,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Texto: {texto}"}
                ],
                max_tokens=10,
                temperature=0.3
            )
            label = response.choices[0].message.content.strip().lower()
            resultados.append(label)
        except Exception as e:
            st.warning(f"Error en texto {idx}: {e}")
            resultados.append("neutro")  # Default
        
        progress_bar.progress((idx + 1) / len(textos))
    
    return resultados

def calcular_metricas(y_true, y_pred, nombre_modelo):
    """Calcula métricas de clasificación"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    
    return {
        'modelo': nombre_modelo,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

# ============================================================================
# SIDEBAR: CONFIGURACIÓN
# ============================================================================

st.sidebar.header("⚙️ Configuración")

# 1. Configuración del modelo
st.sidebar.subheader("1️⃣ Modelo RoBERTuito V2")
usar_modelo_local = st.sidebar.checkbox("Usar modelo local (solo para desarrollo)", value=False)

if usar_modelo_local:
    model_path = st.sidebar.text_input(
        "Ruta del modelo local:",
        value="./robertuito-guatemala-v2.0",
        help="Solo para uso local"
    )
else:
    st.sidebar.info("📡 Usando modelo desde Hugging Face Hub")
    model_path = "ejerez003/robertuito-guatemala-v2.0"

# 2. API Key OpenAI (opcional)
st.sidebar.subheader("2️⃣ Verdad Absoluta (Opcional)")
usar_verdad = st.sidebar.checkbox("Generar verdad absoluta con LLM", value=False)
openai_key = ""
if usar_verdad:
    # Intentar cargar desde secrets de Streamlit Cloud
    try:
        openai_key = st.secrets["OPENAI_API_KEY"]
        st.sidebar.success("✅ API Key cargada desde secrets")
    except:
        openai_key = st.sidebar.text_input(
            "API Key OpenAI:",
            type="password",
            help="Necesaria para generar verdad absoluta"
        )
    
    max_samples_verdad = st.sidebar.number_input(
        "Máximo de muestras para verdad absoluta:",
        min_value=10,
        max_value=5000,
        value=100,
        help="Limita el costo de OpenAI"
    )

# 3. Opciones de procesamiento
st.sidebar.subheader("3️⃣ Opciones")
batch_size = st.sidebar.slider("Batch size:", 8, 64, 32)

# ============================================================================
# MAIN: INTERFAZ PRINCIPAL
# ============================================================================

st.title("🎯 DDI Sentiment Analyzer V2.0")
st.markdown("**Análisis de Sentimiento con RoBERTuito V2 Fine-tuned para Guatemala**")
st.markdown("---")

# PASO 1: Cargar modelo
st.header("📦 Paso 1: Cargar Modelo")
if st.button("🔄 Cargar Modelo RoBERTuito V2"):
    with st.spinner("Cargando modelo..."):
        if usar_modelo_local:
            classifier, tokenizer = cargar_modelo_local(model_path)
        else:
            classifier, tokenizer = cargar_modelo_huggingface(model_path)
        
        if classifier:
            st.session_state['classifier'] = classifier
            st.session_state['tokenizer'] = tokenizer
            st.success("✅ Modelo cargado correctamente")
        else:
            st.error("❌ Error al cargar modelo.")

# PASO 2: Upload archivo
st.header("📂 Paso 2: Cargar Archivo de Datos")
uploaded_file = st.file_uploader(
    "Sube tu archivo CSV o XLSX",
    type=['csv', 'xlsx'],
    help="Debe contener columnas: 'ID', 'texto', 'sentiment' (opcional)"
)

if uploaded_file:
    # Cargar datos
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Validar columnas
    if 'texto' not in df.columns and 'content' in df.columns:
        df.rename(columns={'content': 'texto'}, inplace=True)
    
    if 'sentiment' in df.columns:
        df.rename(columns={'sentiment': 'sentiment_plataforma'}, inplace=True)
        tiene_sentimiento_previo = True
    else:
        tiene_sentimiento_previo = False
    
    st.success(f"✅ Archivo cargado: {len(df)} registros")
    
    # Vista previa
    with st.expander("🔍 Vista previa (primeras 10 filas)"):
        st.dataframe(df.head(10))
    
    # Información
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total registros", len(df))
    with col2:
        st.metric("Columnas", len(df.columns))
    with col3:
        if tiene_sentimiento_previo:
            st.metric("Sentimiento previo", "✅ Detectado")
        else:
            st.metric("Sentimiento previo", "❌ No detectado")
    
    # PASO 3: Procesar
    st.header("🚀 Paso 3: Analizar Sentimiento")
    
    if st.button("▶️ INICIAR ANÁLISIS", type="primary"):
        if 'classifier' not in st.session_state:
            st.error("⚠️ Primero carga el modelo en el Paso 1")
        else:
            # Predicción Modelo V2
            st.subheader("🤖 Procesando con Modelo V2...")
            textos = df['texto'].astype(str).tolist()
            
            predicciones_v2 = predecir_sentimiento_v2(
                textos,
                st.session_state['classifier'],
                st.session_state['tokenizer'],
                batch_size
            )
            
            df['modelo_v2_label'] = [p['label'].lower() for p in predicciones_v2]
            df['modelo_v2_score'] = [p['score'] for p in predicciones_v2]
            
            st.success("✅ Predicciones Modelo V2 completadas")
            
            # Verdad Absoluta (opcional)
            if usar_verdad and openai_key:
                st.subheader("🧠 Generando Verdad Absoluta con LLM...")
                df['verdad_absoluta'] = generar_verdad_absoluta(
                    textos,
                    openai_key,
                    max_samples=max_samples_verdad
                )
                st.success("✅ Verdad absoluta generada")
            
            # Guardar en session state
            st.session_state['df_resultados'] = df
            st.success("🎉 Análisis completado")

# PASO 4: Resultados
if 'df_resultados' in st.session_state:
    st.header("📊 Paso 4: Resultados y Comparación")
    df_res = st.session_state['df_resultados']
    
    # Tabs para organizar resultados
    tab1, tab2, tab3 = st.tabs(["📈 Métricas", "📋 Datos", "📥 Descargar"])
    
    with tab1:
        st.subheader("Comparación de Modelos")
        
        # Calcular métricas si hay verdad absoluta
        if 'verdad_absoluta' in df_res.columns:
            metricas = []
            
            if 'sentiment_plataforma' in df_res.columns:
                metricas.append(calcular_metricas(
                    df_res['verdad_absoluta'],
                    df_res['sentiment_plataforma'],
                    'Plataforma Original'
                ))
            
            metricas.append(calcular_metricas(
                df_res['verdad_absoluta'],
                df_res['modelo_v2_label'],
                'Modelo V2 (RoBERTuito)'
            ))
            
            # Mostrar métricas
            df_metricas = pd.DataFrame(metricas)
            
            # Cards de métricas
            cols = st.columns(len(metricas))
            for idx, row in df_metricas.iterrows():
                with cols[idx]:
                    st.metric(
                        row['modelo'],
                        f"{row['accuracy']:.1%}",
                        delta=f"F1: {row['f1']:.1%}"
                    )
            
            # Gráfico comparativo
            fig = px.bar(
                df_metricas,
                x='modelo',
                y=['accuracy', 'precision', 'recall', 'f1'],
                barmode='group',
                title='Comparación de Métricas',
                labels={'value': 'Score', 'variable': 'Métrica'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribución de sentimientos
        st.subheader("Distribución de Sentimientos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'sentiment_plataforma' in df_res.columns:
                fig1 = px.pie(
                    df_res,
                    names='sentiment_plataforma',
                    title='Plataforma Original'
                )
                st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.pie(
                df_res,
                names='modelo_v2_label',
                title='Modelo V2'
            )
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Tabla de Resultados")
        st.dataframe(df_res, use_container_width=True)
    
    with tab3:
        st.subheader("Descargar Resultados")
        
        # CSV
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV",
            data=csv,
            file_name="resultados_sentimiento_ddi.csv",
            mime="text/csv"
        )
        
        # Excel
        from io import BytesIO
        buffer = BytesIO()
        df_res.to_excel(buffer, index=False)
        buffer.seek(0)
        
        st.download_button(
            label="📥 Descargar Excel",
            data=buffer,
            file_name="resultados_sentimiento_ddi.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("**DDI Sentiment Analyzer V2.0** | Powered by JLA Consulting | RoBERTuito Fine-tuned")
