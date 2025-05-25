import os
import yfinance as yf
import ta
import requests
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

def es_ventana_de_noticias():
    ahora = datetime.utcnow()
    return ahora.hour == 13 and 30 <= ahora.minute <= 35

def get_news_headlines(ticker, num_headlines=3):
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
        return headlines
    except:
        return []

def analizar_sentimiento_vader(titulares):
    if not titulares:
        return "Sin noticias recientes."
    sia = SentimentIntensityAnalyzer()
    text = " ".join(titulares)
    score = sia.polarity_scores(text)["compound"]
    sentimiento = "Positivo" if score >= 0.05 else "Negativo" if score <= -0.05 else "Neutro"
    resumen = "; ".join(titulares[:2])
    return f"{resumen}\nSentimiento general: {sentimiento}"

def detectar_tendencia(close):
    if len(close) < 10:
        return "Insuficiente"
    return "Alcista" if close.iloc[-1] > close.iloc[-10] else "Bajista"

def encontrar_soporte_resistencia(close, periodo=14):
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)

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
    except:
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

    tendencia = detectar_tendencia(close)
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100
    soporte, resistencia = encontrar_soporte_resistencia(close)

    return {
        "ticker": ticker,
        "precio": round(precio, 2),
        "rsi": round(rsi_val, 2),
        "score_bajista": len(señales_bajistas),
        "score_alcista": len(señales_alcistas),
        "señales_bajistas": señales_bajistas,
        "señales_alcistas": señales_alcistas,
        "tendencia": tendencia,
        "retorno_7d": round(retorno_7d, 2),
        "volatilidad": round(volatilidad, 2),
        "atr": round(atr_val, 2),
        "soporte": soporte,
        "resistencia": resistencia
    }

def analizar_intradía(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", auto_adjust=True, progress=False)
        if df.empty or len(df) < 30:
            return None

        close = df["Close"].squeeze()
        ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close).rsi()

        cruce_ema = "ninguno"
        if ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]:
            cruce_ema = "Cruce alcista EMA9/21"
        elif ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]:
            cruce_ema = "Cruce bajista EMA9/21"

        rsi_val = rsi.iloc[-1]
        zona_rsi = "Normal"
        if rsi_val >= 70:
            zona_rsi = "Sobrecompra"
        elif rsi_val <= 30:
            zona_rsi = "Sobreventa"

        return {
            "cruce_ema": cruce_ema,
            "zona_rsi": zona_rsi,
            "rsi": round(rsi_val, 2)
        }
    except Exception as e:
        print(f"❌ Error intradía en {ticker}: {e}")
        return None

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
    print(f"🔍 Analizando {sym}...")
    res = analizar_ticker(sym)
    if res:
        resultados.append(res)

if not resultados:
    enviar_telegram("No se pudo analizar ningún activo.")
else:
    mejor = max(resultados, key=lambda r: max(r["score_bajista"], r["score_alcista"]))
    tipo = "corto" if mejor["score_bajista"] >= mejor["score_alcista"] else "largo"
    señales = mejor["señales_bajistas"] if tipo == "corto" else mejor["señales_alcistas"]

    if es_ventana_de_noticias():
        titulares = get_news_headlines(mejor["ticker"])
        resumen_noticia = analizar_sentimiento_vader(titulares)
    else:
        resumen_noticia = "Análisis de noticias disponible a las 13:30 UTC."

    sentimiento = (
        resumen_noticia.splitlines()[-1]
        if isinstance(resumen_noticia, str) and 'Sentimiento general:' in resumen_noticia
        else 'No disponible'
    )

    # Análisis intradía
    intradia = analizar_intradía(mejor["ticker"])
    señales_extra = []
    prob_sube = 0
    prob_baja = 0
    if intradia:
        if intradia["cruce_ema"] == "Cruce alcista EMA9/21":
            señales_extra.append(intradia["cruce_ema"])
            prob_sube += 25
        elif intradia["cruce_ema"] == "Cruce bajista EMA9/21":
            señales_extra.append(intradia["cruce_ema"])
            prob_baja += 25
        if intradia["zona_rsi"] == "Sobreventa":
            señales_extra.append("RSI 5min en sobreventa")
            prob_sube += 25
        elif intradia["zona_rsi"] == "Sobrecompra":
            señales_extra.append("RSI 5min en sobrecompra")
            prob_baja += 25

    mensaje = f"""📌 *Oportunidad destacada: {mejor['ticker']}*
Precio: {mejor['precio']} USD
RSI Diario: {mejor['rsi']}
Tendencia: {mejor['tendencia']}
Volatilidad: {mejor['volatilidad']}% | ATR: {mejor['atr']}
Retorno 7d: {mejor['retorno_7d']}%
Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}
Entrada sugerida: en *{tipo}*

*Señales técnicas:*
{chr(10).join(f"- {s}" for s in señales)}

*Señales intradía (5m):*
{chr(10).join(f"- {s}" for s in señales_extra)}

📈 *Probabilidad de subida:* {prob_sube}%
📉 *Probabilidad de bajada:* {prob_baja}%

📰 *Noticias recientes:*
{resumen_noticia}
"""

    enviar_telegram(mensaje)
