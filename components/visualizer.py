import plotly.express as px
import streamlit as st
import pandas as pd

def plot_sentiment_distribution(df):
    """Genera un gráfico de torta para la distribución de sentimientos."""
    if 'sentiment' not in df.columns:
        return None
    
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentimiento', 'Cantidad']
    
    # Colores personalizados (DDI Palette ish)
    color_map = {
        'positive': '#10B981',  # Green
        'negative': '#EF4444',  # Red
        'neutral': '#6B7280',   # Gray
        'error': '#000000'
    }
    
    fig = px.pie(
        sentiment_counts, 
        values='Cantidad', 
        names='Sentimiento',
        title='Distribución de Sentimientos',
        color='Sentimiento',
        color_discrete_map=color_map,
        hole=0.4
    )
    return fig

def plot_topic_distribution(df):
    """Genera un gráfico de barras para la distribución de tópicos."""
    if 'topic' not in df.columns:
        return None
    
    topic_counts = df['topic'].value_counts().reset_index().head(10) # Top 10
    topic_counts.columns = ['Tópico', 'Cantidad']
    
    fig = px.bar(
        topic_counts,
        x='Cantidad',
        y='Tópico',
        orientation='h',
        title='Top 10 Tópicos Detectados',
        text='Cantidad',
        color='Cantidad',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig
