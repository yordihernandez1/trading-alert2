import os
import yfinance as yf
import ta
import requests
from datetime import datetime
import numpy as np
from bs4 import BeautifulSoup
from transformers import pipeline

# 📦 Modelo local de análisis de sentimiento
sentiment_model = pipeline("sentiment-analysis")

# 🧾 Configuración de entorno
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))

# 📍 Lista de activos a analizar
symbols = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

# 🕐 Función: ¿es momento de incluir noticias?
def es_ventana_de_noticias():
    ahora = datetime.utcnow()
    return ahora.hour == 13 and 30 <= ahora.minute <= 35

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

# 🤖 Análisis de sentimiento local
def analizar_sentimiento(titulares):
    if not titulares:
        return "Sin noticias recientes."
    
    resultados = sentiment_model(titulares)
    positivos = sum(1 for r in resultados if r["label"] == "POSITIVE")
    negativos = sum(1 for r in resultados if r["label"] == "NEGATIVE")
    sentimiento = "Positivo" if positivos > negativos else "Negativo" if negativos > positivos else "Neutro"
    resumen = "; ".join(titulares[:2])
    return f"{resumen}\nSentimiento general: {sentimiento}"

# 🔍 Ejecutar análisis técnico
resultados = [analizar_ticker(sym) for sym in symbols]
resultados = [r for r in resultados if r]

if not resultados:
    mensaje = "❌ No se pudo analizar ningún activo."
else:
    mejor = max(resultados, key=lambda r: max(r["score_bajista"], r["score_alcista"]))
    tipo = "corto 🔻" if mejor["score_bajista"] >= mejor["score_alcista"] else "largo 🚀"
    señales = mejor["señales_bajistas"] if tipo.startswith("corto") else mejor["señales_alcistas"]

    # 📩 Incluir análisis de noticias solo en la ventana adecuada
    if es_ventana_de_noticias():
        titulares = get_news_headlines(mejor["ticker"])
        resumen_noticia = analizar_sentimiento(titulares)
    else:
        resumen_noticia = "🕓 Análisis de noticias disponible a las 13:30 UTC."

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
"""

# 🚀 Enviar mensaje por Telegram
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
