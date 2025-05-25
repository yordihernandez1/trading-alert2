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

# Descargar diccionario de VADER (solo la primera vez)
nltk.download('vader_lexicon')

# ✅ Configuración
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))

symbols = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

# 🕐 Ventana para análisis de noticias
def es_ventana_de_noticias():
    ahora = datetime.utcnow()
    return ahora.hour == 13 and 30 <= ahora.minute <= 35

# 📰 Scraping de titulares
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

# 🧠 Análisis de sentimiento con VADER
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

# 📈 Análisis técnico
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
    data["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()
    macd = ta.trend.MACD(close=close)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    data["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

    try:
        atr = ta.volatility.AverageTrueRange(
            high=data["High"], low=data["Low"], close=close
        ).average_true_range().iloc[-1]
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
        señales_bajistas.append("RSI elevado → posible sobrecompra")
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
        señales_alcistas.append("RSI bajo → posible sobreventa")
        score_alcista += 1
    if macd_val > macd_sig:
        señales_alcistas.append("MACD cruzando al alza")
        score_alcista += 1
    if precio < sma_50 and sma_50 < sma_200:
        señales_alcistas.append("Precio bajo con posible recuperación (por debajo de SMAs)")
        score_alcista += 1
    if precio > cierre_anterior:
        señales_alcistas.append("Última vela cerró en verde")
        score_alcista += 1

    tendencia = detectar_tendencia(close)
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100

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
        "atr": round(atr, 2) if not np.isnan(atr) else "N/D"
    }

# 📊 Backtesting
def detectar_entrada_alcista(df):
    entradas = []
    for i in range(5, len(df)):
        rsi = df["rsi"].iloc[i]
        macd = df["macd"].iloc[i]
        macd_sig = df["macd_signal"].iloc[i]
        precio = df["Close"].iloc[i]
        sma50 = df["sma_50"].iloc[i]
        sma200 = df["sma_200"].iloc[i]
        precio_ant = df["Close"].iloc[i - 1]

        score = 0
        if rsi < RSI_SOBREVENTA:
            score += 1
        if macd > macd_sig:
            score += 1
        if precio < sma50 and sma50 < sma200:
            score += 1
        if precio > precio_ant:
            score += 1

        if score >= 2:
            entradas.append(i)
    return entradas

def analizar_backtest(ticker, dias_salida=5):
    data = yf.download(ticker, period="1y", interval="1d")
    if data.empty or len(data) < 200:
        return None

    df = data.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["sma_50"] = ta.trend.SMAIndicator(df["Close"], 50).sma_indicator()
    df["sma_200"] = ta.trend.SMAIndicator(df["Close"], 200).sma_indicator()

    entradas = detectar_entrada_alcista(df)
    resultados = []

    for i in entradas:
        if i + dias_salida >= len(df):
            continue
        precio_entrada = df["Close"].iloc[i]
        precio_salida = df["Close"].iloc[i + dias_salida]
        retorno = (precio_salida - precio_entrada) / precio_entrada
        resultados.append(retorno)

    if not resultados:
        return None

    resultados = np.array(resultados)
    return {
        "ticker": ticker,
        "trades": len(resultados),
        "ganadores": int((resultados > 0).sum()),
        "retorno_promedio": round(resultados.mean() * 100, 2),
        "retorno_total": round(resultados.sum() * 100, 2),
        "max_drawdown": round(resultados.min() * 100, 2),
        "porcentaje_ganadores": round((resultados > 0).mean() * 100, 2)
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

    backtests = [analizar_backtest(sym) for sym in symbols]
    backtests = [r for r in backtests if r]
    resumen_backtest = "\n".join(
        f"{r['ticker']}: {r['porcentaje_ganadores']}% winrate | Prom: {r['retorno_promedio']}% | MaxDD: {r['max_drawdown']}%"
        for r in backtests
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

📰 Noticias recientes:
{resumen_noticia}

📊 Backtest 12 meses:
{resumen_backtest}
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
