# DDI Sentiment Analyzer - RoBERTuito V2

Aplicación web para análisis de sentimiento usando el modelo RoBERTuito V2.0 fine-tuned para Guatemala.

## 🏗️ Arquitectura

```
┌─────────────┐      HTTP      ┌──────────────────┐
│  Analista   │ ────────────▶  │  Streamlit App   │
│  (Browser)  │                │  (UI Frontend)   │
└─────────────┘                └──────────────────┘
                                        │
                                        │ API Call
                                        ▼
                               ┌──────────────────┐
                               │  Google Colab    │
                               │  (Flask API)     │
                               │  RoBERTuito V2   │
                               └──────────────────┘
```

## ✨ Características

- ✅ Análisis de sentimiento con RoBERTuito V2.0
- ✅ Comparación automática: Sentimiento Original vs V2
- ✅ Métricas de evaluación: Accuracy, Precision, Recall, F1-Score
- ✅ Matriz de confusión interactiva
- ✅ Soporte para Excel (.xlsx) y CSV
- ✅ Descarga de resultados procesados
- ✅ 100% GRATIS (usando Google Colab)

## 🚀 Guía de Uso

### Paso 1: Configurar el Backend (Colab)

1. Abre el notebook [`DDI_Sentiment_API_Colab.ipynb`](./DDI_Sentiment_API_Colab.ipynb) en Google Colab
2. Ejecuta todas las celdas en orden (Runtime → Run all)
3. Espera a que se cargue el modelo (~1-2 minutos)
4. Copia la URL pública generada (ej: `https://xxxx.ngrok.io`)

> **Nota**: El notebook debe permanecer ejecutándose mientras uses la app. Colab Free desconecta después de ~12 horas o 90 minutos de inactividad.

### Paso 2: Usar la App Web

1. Accede a la app: [https://ddi-sentiment-analyzer.streamlit.app](https://ddi-sentiment-analyzer.streamlit.app)
2. En la barra lateral, pega la URL del API de Colab
3. Sube tu archivo Excel/CSV con:
   - Columna **`Comentario`**: Texto a analizar
   - Columna **`sentiment`**: Sentimiento original (numérico: -5=negativo, 0=neutro, 5=positivo)
4. Haz clic en **"🚀 Analizar Sentimientos"**
5. Revisa los resultados:
   - Métricas de evaluación
   - Matriz de confusión
   - Gráficos comparativos
6. Descarga el CSV con resultados

## 📊 Formato del Archivo de Entrada

### Ejemplo Excel/CSV

| Comentario | sentiment |
|---|---|
| Me encanta este producto! | 5 |
| No funciona bien | -5 |
| Es normal | 0 |

### Columnas Requeridas

- **`Comentario`** (obligatorio): Texto a analizar
- **`sentiment`** (opcional): Sentimiento original para comparación
  - Valores negativos (ej: -5) → Negativo
  - Valor 0 → Neutro
  - Valores positivos (ej: 5) → Positivo

## 📈 Salida

El archivo descargado incluirá las columnas originales más:

- **`sentiment_original`**: Sentimiento original convertido a labels (negative/neutral/positive)
- **`sentiment`**: Predicción del modelo V2 (negative/neutral/positive) ⚡ *Fondo amarillo*
- **`confidence`**: Confianza de la predicción (0.0 - 1.0) ⚡ *Fondo amarillo*

## 🔧 Desarrollo Local

```bash
# Clonar repo
git clone https://github.com/ejerez-jla/ddi-sentiment-analyzer.git
cd ddi-sentiment-analyzer

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar app
streamlit run app.py
```

## 💰 Costos

- **Streamlit Cloud**: GRATIS
- **Google Colab**: GRATIS (Free Tier)
- **Modelo RoBERTuito V2**: GRATIS (open source)

**Total: $0/mes** 🎉

## ⚠️ Limitaciones

- **Concurrencia**: 1 procesamiento a la vez por sesión de Colab
- **Disponibilidad**: Requiere re-ejecutar Colab cada ~12 horas
- **URL dinámica**: La URL de ngrok cambia cada vez (usar cloudflared para URL fija)

## 🔮 Escalabilidad Futura

Para soportar 20+ usuarios simultáneos:

- **Opción A**: Hugging Face Inference Endpoints (~$432/mes)
- **Opción B**: Servidor AWS con auto-scaling (~$100-200/mes)

## 📝 Modelo

- **Nombre**: RoBERTuito V2.0
- **Base**: `pysentimiento/robertuito-base-uncased`
- **Fine-tuning**: Datos de redes sociales de Guatemala
- **Hub**: `accesosddi/Sentimiento2`

## 📄 Licencia

MIT License

## 👥 Contacto

Para soporte técnico, contacta al equipo de DDI.
