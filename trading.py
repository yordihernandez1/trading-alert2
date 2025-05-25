import os
import yfinance as yf
import ta
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ⚙️ Usar lexicón local desde la raíz del proyecto
def cargar_analizador_personalizado():
    lexicon_path = './vader_lexicon.txt'
    return SentimentIntensityAnalyzer(lexicon_file=lexicon_path)

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
    analizador = cargar_analizador_personalizado()
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

def analizar_ticker(ticker):
    try:
        data = yf.download(ticker, period="3mo", interval="1d")
        if data.empty or len(data) < 20:
            return None
    except:
        return None

    if any(col not in data.columns for col in ["High", "Low", "Close"]):
        return None

    close = data["Close"].squeeze()
    data["rsi"] = ta.momentum.RSIIndicator(close).rsi()
    macd = ta.trend.MACD(close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["sma_50"] = ta.trend.SMAIndicator(close, window=50).sma_indicator()
    data["sma_200"] = ta.trend.SMAIndicator(close, window=200).sma_indicator()

    try:
        atr = ta.volatility.AverageTrueRange(high=data["High"], low=data["Low"], close=close).average_true_range().iloc[-1]
    except:
        atr = np.nan

    try:
        rsi = float(data["rsi"].iloc[-1])
        macd_val = float(data["macd"].iloc[-1])
        macd_sig = float(data["macd_signal"].iloc[-1])
        sma_50 = float(data["sma_50"].iloc[-1])
        sma_200 = float(data["sma_200"].iloc[-1])
        precio = float(data["Close"].iloc[-1])
        cierre_anterior = float(data["Close"].iloc[-2])
    except:
        return None

    señales_bajistas = []
    señales_alcistas = []
    score_bajista = 0
    score_alcista = 0

    if rsi > RSI_SOBRECOMPRA:
        señales_bajistas.append("RSI en sobrecompra → posible venta")
        score_bajista += 1
    if macd_val < macd_sig:
        señales_bajistas.append("MACD cruzando a la baja")
        score_bajista += 1
    if precio < sma_50:
        señales_bajistas.append("Precio por debajo de la SMA 50")
        score_bajista += 1
    if sma_50 < sma_200:
        señales_bajistas.append("SMA 50 por debajo de SMA 200")
        score_bajista += 1
    if precio < cierre_anterior:
        señales_bajistas.append("Última vela cerró en rojo")
        score_bajista += 1

    if rsi < RSI_SOBREVENTA:
        señales_alcistas.append("RSI en sobreventa → posible compra")
        score_alcista += 1
    if macd_val > macd_sig:
        señales_alcistas.append("MACD cruzando al alza")
        score_alcista += 1
    if precio < sma_50 and sma_50 < sma_200:
        señales_alcistas.append("Precio bajo con posible recuperación")
        score_alcista += 1
    if precio > cierre_anterior:
        señales_alcistas.append("Última vela cerró en verde")
        score_alcista += 1

    tendencia = detectar_tendencia(close)
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100
    soporte, resistencia = encontrar_soporte_resistencia(close)

    return {
        "ticker": ticker,
        "precio": round(precio, 2),
        "rsi": round(rsi, 2),
        "score_bajista": score_bajista,
        "score_alcista": score_alcista,
        "señales_bajistas": señales_bajistas,
        "señales_alcistas": señales_alcistas,
        "tendencia": tendencia,
        "retorno_7d": round(retorno_7d, 2),
        "volatilidad": round(volatilidad, 2),
        "atr": round(atr, 2) if not np.isnan(atr) else "N/D",
        "soporte": soporte,
        "resistencia": resistencia
    }

# 🚀 Ejecución principal
resultados = [analizar_ticker(sym) for sym in symbols]
resultados = [r for r in resultados if r]

if not resultados:
    mensaje = "❌ No se pudo analizar ningún activo."
else:
    mejor = max(resultados, key=lambda r: max(r["score_bajista"], r["score_alcista"]))
    tipo = "corto 🔻" if mejor["score_bajista"] >= mejor["score_alcista"] else "largo 🚀"
    señales = mejor["señales_bajistas"] if tipo.startswith("corto") else mejor["señales_alcistas"]

    if es_ventana_de_noticias():
        titulares = get_news_headlines(mejor["ticker"])
        resumen_noticia = analizar_sentimiento_vader(titulares)
    else:
        resumen_noticia = "🕓 Análisis de noticias disponible a las 13:30 UTC."

    sentimiento = (
        resumen_noticia.splitlines()[-1]
        if isinstance(resumen_noticia, str) and 'Sentimiento general:' in resumen_noticia
        else 'No disponible'
    )

    mensaje = f"""
📊 OPORTUNIDAD DESTACADA: {mejor['ticker']}
💰 Precio: {mejor['precio']} USD
📈 RSI: {mejor['rsi']}

📌 Entrada sugerida: en {tipo}

🔍 Análisis técnico cualitativo:
- Tendencia general: {mejor['tendencia']}

📉 Datos cuantitativos:
- Volatilidad 14d: {mejor['volatilidad']}%
- Retorno 7d: {mejor['retorno_7d']}%
- ATR: {mejor['atr']}

📌 Señales detectadas:
""" + "\n".join(f"- {s}" for s in señales) + f"""

📍 Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}

📰 Noticias recientes:
{resumen_noticia}

🤖 Pregunta para ChatGPT:
Con base en los siguientes datos técnicos para {mejor['ticker']}:
- RSI: {mejor['rsi']}
- Tendencia: {mejor['tendencia']}
- Señales: {', '.join(señales)}
- Sentimiento: {sentimiento}
- Soporte: {mejor['soporte']}, Resistencia: {mejor['resistencia']}
¿Consideras que es una buena oportunidad para entrar en {'compra' if tipo.startswith('largo') else 'venta'}? Justifica brevemente.
"""

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": mensaje
    }
    response = requests.post(url, data=payload)

    if response.status_code == 200:
        print("✅ Alerta enviada correctamente")
    else:
        print("❌ Error al enviar alerta:", response.text)

