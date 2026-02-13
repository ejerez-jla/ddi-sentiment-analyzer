# Documentación Ejecutiva: DDI Sentiment Analyzer V2.0

## 1. Visión General
Esta herramienta representa la evolución del análisis de sentimiento para DDI, pasando de procesos manuales o locales a una arquitectura moderna, escalable y accesible desde la nube. Su objetivo principal es democratizar el acceso al modelo de Inteligencia Artificial **RoBERTuito V2.0**, permitiendo que cualquier analista pueda procesar grandes volúmenes de comentarios sin necesidad de conocimientos de programación o hardware costoso.

## 2. Nuestra Filosofía de Desarrollo: "Agile & Lean"
Para este proyecto, hemos adoptado una filosofía centrada en la entrega rápida de valor y la eficiencia de recursos:

*   **User-Centric (Centrado en el Usuario):** Priorizamos una interfaz limpia e intuitiva (Streamlit). Lo complejo (el código, la matemática) queda oculto; el usuario solo ve "Cargar Archivo" y "Ver Resultados".
*   **Cost-Effective (Costo-Eficiente):** En lugar de contratar servidores costosos ($50-$100/mes) para mantener el modelo encendido 24/7 (incluso cuando nadie lo usa), utilizamos la potencia de cálculo gratuita de Google Colab para el procesamiento pesado.
*   **Prueba de Concepto Evolutiva:** Esta versión no es un prototipo desechable, sino una base sólida. Está diseñada para funcionar hoy y escalar mañana. Permite validar el valor del modelo antes de invertir en infraestructura definitiva.

## 3. Arquitectura (Cómo funciona por dentro)
Imagina el sistema como un restaurante:

1.  **El Cliente (Streamlit):** Es la página web bonita donde tú (el comensal) interactúas. Aquí subes tu archivo Excel, ves los gráficos y descargas los resultados. Es ligero y rápido.
2.  **El Cocinero (Google Colab):** Es donde ocurre la magia pesada. Aquí vive el modelo de Inteligencia Artificial. Cuando tú pides un análisis, Streamlit le manda la "orden" a Colab, Colab cocina los datos (procesa miles de textos) y devuelve el plato listo.
3.  **El Mesero (Ngrok):** Es el túnel seguro que conecta tu navegador con la cocina de Google. Sin él, la cocina estaría aislada del mundo.

**¿Por qué esta separación?**
Porque el modelo de IA es "pesado" (como un horno industrial). Streamlit Cloud es "ligero" (como un mostrador). Si intentamos meter el horno en el mostrador, se rompe. Por eso, dejamos el horno en Google (que nos lo presta gratis) y usamos el mostrador solo para atenderte.

## 4. El Modelo: RoBERTuito V2.0
No estamos usando un diccionario simple de palabras (donde "malo" = negativo). Estamos usando un **Transformer** (tecnología similar a GPT/ChatGPT), pero especializado:

*   **Especialista Regional:** A diferencia de modelos genéricos, RoBERTuito ha sido entrenado específicamente con tweets y comentarios de la región (Guatemala/Centroamérica). Entiende modismos, jerga y sarcasmo local mejor que un modelo global.
*   **Contextual:** Entiende que *"No está nada mal"* es positivo, aunque tenga la palabra "mal".
*   **Calibrado:** Devuelve no solo el sentimiento (Positivo/Negativo/Neutro), sino también un nivel de **confianza** (0-100%). Esto nos permite filtrar predicciones dudosas.

## 5. Valor para el Negocio
*   **Autonomía:** Los analistas ya no dependen del equipo de Data Science para obtener métricas cada vez.
*   **Estandarización:** Todos usan el mismo criterio (el del modelo) para evaluar campañas, eliminando la subjetividad humana variable.
*   **Velocidad:** Procesar 1,000 comentarios toma segundos/minutos, versus horas de lectura manual.

## 6. Siguientes Pasos: Evolución a Producto (SaaS)
Lo que hemos construido hoy es una "Prueba de Concepto" funcional. El siguiente paso natural es convertir esto en una aplicación web autónoma que funcione 24/7 sin intervención técnica.

**La Diferencia:**
*   **Hoy (Fase 1):** Un analista técnico debe "encender el horno" (Google Colab) cada vez que se quiere cocinar.
*   **Futuro (Fase 2):** El horno está siempre encendido y listo. El usuario solo entra a la web y usa la herramienta.

**Beneficios de la Fase 2:**
1.  **Disponibilidad 24/7:** No hay que esperar a que arranque el servidor.
2.  **Cero Mantenimiento para el Usuario:** El analista no lidia con notebooks, tokens o conexiones.
3.  **Escalabilidad:** Puede atender a múltiples usuarios de DDI simultáneamente.

---
*Desarrollado para DDI - Febrero 2026*
