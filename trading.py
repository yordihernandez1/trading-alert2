import os
import yfinance as yf
import ta
import requests
import numpy as np
import pandas as pd
import nltk
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))

symbols = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

def es_ventana_de_noticias():
    ahora = datetime.utcnow()
    return ahora.hour == 13 and 30 <= ahora.minute <= 35

def get_news_headlines(ticker, num_headlines=3):
    query = f"{ticker} stock"
    url = f"https://www.google.com/search?q={query}&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [g.select_one('div.JheGif.nDgy9d').text for g in soup.select('div.dbsr')[:num_headlines] if g.select_one('div.JheGif.nDgy9d')]
        return headlines
    except:
        return []

def analizar_sentimiento_vader(titulares):
    if not titulares:
        return "Sin noticias recientes."
    analizador = SentimentIntensityAnalyzer()
    textos = " ".join(titulares)
    scores = analizador.polarity_scores(textos)
    compound = scores['compound']
    if compound >= 0.05:
        sentimiento = "Positivo"
    elif compound <= -0.05:
        sentimiento = "Negativo"
    else:
        sentimiento = "Neutro"
    resumen = "; ".join(titulares[:2])
    return f"{resumen}\nSentimiento general: {sentimiento}"

def detectar_tendencia(close):
    try:
        val_actual = float(close.iloc[-1])
        val_antiguo = float(close.iloc[-10])
        if val_actual > val_antiguo:
            return "Alcista"
        elif val_actual < val_antiguo:
            return "Bajista"
        else:
            return "Lateral"
    except:
        return "Desconocida"

def encontrar_soporte_resistencia(close, periodo=14):
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)

# ... [rest of the code remains unchanged until building the message] ...

    sentimiento = (
        resumen_noticia.splitlines()[-1]
        if isinstance(resumen_noticia, str) and 'Sentimiento general:' in resumen_noticia
        else 'No disponible'
    )

    mensaje += f"""
📍 Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}

🤖 Pregunta para ChatGPT:
Con base en los siguientes datos técnicos para {mejor['ticker']}:
- RSI: {mejor['rsi']}
- Tendencia: {mejor['tendencia']}
- Señales: {', '.join(señales)}
- Sentimiento: {sentimiento}
- Soporte: {mejor['soporte']}, Resistencia: {mejor['resistencia']}
¿Consideras que es una buena oportunidad para entrar en {'compra' if tipo.startswith('largo') else 'venta'}? Justifica brevemente.
"""
