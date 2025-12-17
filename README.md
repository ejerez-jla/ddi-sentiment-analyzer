# 🎯 DDI Sentiment Analyzer V2.0

Aplicación de análisis de sentimiento para DDI Guatemala usando RoBERTuito V2.0 fine-tuned.

## 🚀 Uso Local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Preparar el modelo

1. Descargar `robertuito-guatemala-v2.0.zip`
2. Descomprimir en la misma carpeta que esta app
3. Debe quedar: `./robertuito-guatemala-v2.0/` con los archivos del modelo

### 3. Ejecutar la app

```bash
streamlit run app.py
```

Se abrirá automáticamente en http://localhost:8501

## 📂 Estructura de Archivos

```
streamlit_app/
├── app.py                      # Aplicación principal
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
├── robertuito-guatemala-v2.0/  # Modelo (descargar por separado)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
```

## 📊 Uso de la App

### Paso 1: Cargar Modelo
- Click en "🔄 Cargar Modelo RoBERTuito V2"
- Espera a que se cargue (~30 segundos)

### Paso 2: Subir Datos
- Sube archivo CSV o Excel
- Debe tener columna `texto` (obligatorio)
- Puede tener columna `sentiment` (opcional, para comparación)

### Paso 3: Analizar
- Click en "▶️ INICIAR ANÁLISIS"
- Espera el procesamiento

### Paso 4: Resultados
- Ver métricas, gráficos y datos
- Descargar resultados en CSV/Excel

## ⚙️ Configuración Opcional

### Verdad Absoluta con OpenAI (Sidebar)
1. Activar checkbox "Generar verdad absoluta con LLM"
2. Ingresar API Key de OpenAI
3. Definir máximo de muestras (controla costo)

## 💡 Consejos

- **Batch size**: Aumentar si tienes buena RAM (acelera procesamiento)
- **Verdad absoluta**: Usar solo para validación (genera costo en OpenAI)
- **Modelo local**: Debe estar en la ruta especificada en el sidebar

## 📞 Soporte

JLA Consulting Group  
Ernesto Jerez - ejerez@jlagrp.com
