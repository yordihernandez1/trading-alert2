import os
import yfinance as yf
import ta
import requests
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configuración de entorno
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Faltan BOT_TOKEN o CHAT_ID en las variables de entorno")

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

def es_ventana_de_noticias():
    ahora = datetime.utcnow()
    return ahora.hour == 13 and 30 <= ahora.minute <= 35

def get_news_sentiment(ticker, num_headlines=3):
    url = f"https://www.google.com/search?q={ticker}+stock&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = [
            el.select_one('div.JheGif.nDgy9d').text
            for el in soup.select('div.dbsr')[:num_headlines]
            if el.select_one('div.JheGif.nDgy9d')
        ]
        if not headlines:
            return "Sin noticias recientes.", "No disponible"
        text = " ".join(headlines)
        sia = SentimentIntensityAnalyzer()
        score = sia.polarity_scores(text)['compound']
        sentimiento = "Positivo" if score >= 0.05 else "Negativo" if score <= -0.05 else "Neutro"
        return "; ".join(headlines), sentimiento
    except Exception as e:
        print(f"⚠️ Error obteniendo noticias para {ticker}: {e}")
        return "Error obteniendo noticias.", "No disponible"

def analizar_ticker(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 50:
            return None
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        rsi = ta.momentum.RSIIndicator(close.squeeze()).rsi()
        macd = ta.trend.MACD(close.squeeze())
        sma_50 = ta.trend.SMAIndicator(close.squeeze(), window=50).sma_indicator()
        sma_200 = ta.trend.SMAIndicator(close.squeeze(), window=200).sma_indicator()
        atr = ta.volatility.AverageTrueRange(high.squeeze(), low.squeeze(), close.squeeze()).average_true_range()
    except Exception as e:
        print(f"❌ Error analizando {ticker}: {e}")
        return None

    try:
        precio = close.iloc[-1]
        cierre_ant = close.iloc[-2]
        rsi_val = rsi.iloc[-1]
        macd_val = macd.macd().iloc[-1]
        macd_sig = macd.macd_signal().iloc[-1]
        sma50 = sma_50.iloc[-1]
        sma200 = sma_200.iloc[-1]
        atr_val = atr.iloc[-1]
    except:
        return None

    señales_alcistas, señales_bajistas = [], []
    if rsi_val < RSI_SOBREVENTA: señales_alcistas.append("RSI en sobreventa")
    if rsi_val > RSI_SOBRECOMPRA: señales_bajistas.append("RSI en sobrecompra")
    if macd_val > macd_sig: señales_alcistas.append("MACD cruzando al alza")
    if macd_val < macd_sig: señales_bajistas.append("MACD cruzando a la baja")
    if precio > cierre_ant: señales_alcistas.append("Última vela verde")
    if precio < cierre_ant: señales_bajistas.append("Última vela roja")
    if precio > sma50: señales_alcistas.append("Precio sobre SMA50")
    else: señales_bajistas.append("Precio bajo SMA50")
    if sma50 > sma200: señales_alcistas.append("SMA50 sobre SMA200")
    else: señales_bajistas.append("SMA50 bajo SMA200")

    tendencia = "Alcista" if close.iloc[-1] > close.iloc[-10] else "Bajista"
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100
    soporte = round(min(close[-14:]), 2)
    resistencia = round(max(close[-14:]), 2)

    return {
        "ticker": ticker,
        "precio": round(precio, 2),
        "rsi": round(rsi_val, 2),
        "tendencia": tendencia,
        "retorno_7d": round(retorno_7d, 2),
        "volatilidad": round(volatilidad, 2),
        "atr": round(atr_val, 2),
        "soporte": soporte,
        "resistencia": resistencia,
        "alcistas": señales_alcistas,
        "bajistas": señales_bajistas
    }

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("⚠️ Error enviando mensaje:", e)

# Ejecución principal
resultados = []
for sym in SYMBOLS:
    print(f"⏳ Procesando {sym}...")
    datos = analizar_ticker(sym)
    if datos:
        resultados.append(datos)

if not resultados:
    enviar_telegram("No se pudo analizar ningún activo.")
else:
    mejor = max(resultados, key=lambda r: len(r["alcistas"]) + len(r["bajistas"]))
    tipo = "largo" if len(mejor["alcistas"]) >= len(mejor["bajistas"]) else "corto"
    señales = mejor["alcistas"] if tipo == "largo" else mejor["bajistas"]
    resumen_noticias, sentimiento = get_news_sentiment(mejor["ticker"]) if es_ventana_de_noticias() else ("Noticias disponibles a las 13:30 UTC.", "No disponible")

    mensaje = f"""📈 *Oportunidad destacada: {mejor['ticker']}*
Precio actual: {mejor['precio']} USD
RSI: {mejor['rsi']}
Tendencia: {mejor['tendencia']}
Volatilidad: {mejor['volatilidad']:.2f}% | ATR: {mejor['atr']}
Retorno 7d: {mejor['retorno_7d']:.2f}%
Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}
Entrada sugerida: en *{tipo}*

*Señales detectadas:*
{chr(10).join(f"- {s}" for s in señales)}

*Noticias recientes:*
{resumen_noticias}
_Sentimiento: {sentimiento}_"""

    enviar_telegram(mensaje)

