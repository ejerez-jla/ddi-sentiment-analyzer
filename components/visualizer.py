import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_sentiment_distribution(df):
    """Genera un gráfico de torta para la distribución de sentimientos."""
    if 'sentiment' not in df.columns:
        return None
    
    counts = df['sentiment'].value_counts()
    
    # Colores personalizados
    colors = {
        'positive': '#4CAF50',
        'neutral': '#FFC107',
        'negative': '#F44336',
        'error': '#9E9E9E'
    }
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title='Distribución de Sentimientos (V2)',
        color=counts.index,
        color_discrete_map=colors
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_topic_distribution(df):
    """Genera un gráfico de barras para la distribución de tópicos."""
    if 'topic' not in df.columns:
        return None
    
    counts = df['topic'].value_counts().head(10)
    
    fig = px.bar(
        x=counts.values,
        y=counts.index,
        orientation='h',
        title='Top 10 Tópicos Detectados',
        labels={'x': 'Cantidad', 'y': 'Tópico'}
    )
    
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive']):
    """
    Genera un heatmap de la matriz de confusión.
    
    Args:
        y_true: Etiquetas reales
        y_pred: Etiquetas predichas
        labels: Lista de labels en orden
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Matriz de Confusión: Sentimiento Original vs V2',
        xaxis_title='Predicción V2',
        yaxis_title='Sentimiento Original',
        width=500,
        height=500
    )
    
    return fig

def plot_comparison_bars(df, original_col='sentiment_original', v2_col='sentiment'):
    """
    Genera gráfico de barras comparando distribución original vs V2.
    
    Args:
        df: DataFrame con ambas columnas de sentimiento
        original_col: Nombre de la columna de sentimiento original
        v2_col: Nombre de la columna de sentimiento V2
    """
    if original_col not in df.columns or v2_col not in df.columns:
        return None
    
    # Contar distribuciones
    original_counts = df[original_col].value_counts()
    v2_counts = df[v2_col].value_counts()
    
    # Crear DataFrame para plotly
    labels = ['negative', 'neutral', 'positive']
    comparison_data = pd.DataFrame({
        'Sentimiento': labels * 2,
        'Cantidad': [original_counts.get(l, 0) for l in labels] + [v2_counts.get(l, 0) for l in labels],
        'Fuente': ['Original'] * 3 + ['V2'] * 3
    })
    
    fig = px.bar(
        comparison_data,
        x='Sentimiento',
        y='Cantidad',
        color='Fuente',
        barmode='group',
        title='Comparación: Sentimiento Original vs V2',
        color_discrete_map={'Original': '#2196F3', 'V2': '#FF9800'}
    )
    
    return fig
