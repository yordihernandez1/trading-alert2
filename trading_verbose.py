import os
import yfinance as yf
import ta
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from pathlib import Path

LOG_ALERTA = "ultima_alerta.json"
TIEMPO_RESUMEN_MINUTOS = 30


BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
UMBRAL_ALERTA = 50

def es_mercado_abierto():
    ahora = datetime.utcnow()
    return ahora.weekday() < 5 or "BTC-USD" in SYMBOLS

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

def analizar_tecnico_diario(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 50:
            return None
        close = df["Close"].squeeze().squeeze()
        high = df["High"].squeeze().squeeze()
        low = df["Low"].squeeze().squeeze()

        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator()
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        tendencia = "Alcista" if close.iloc[-1] > close.iloc[-10] else "Bajista"

        señales = []
        if rsi.iloc[-1] > RSI_SOBRECOMPRA:
            señales.append("RSI en sobrecompra")
        if rsi.iloc[-1] < RSI_SOBREVENTA:
            señales.append("RSI en sobreventa")
        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]:
            señales.append("MACD cruzando al alza")
        else:
            señales.append("MACD cruzando a la baja")
        if close.iloc[-1] > sma_50.iloc[-1]:
            señales.append("Precio sobre SMA50")
        if sma_50.iloc[-1] > sma_200.iloc[-1]:
            señales.append("SMA50 sobre SMA200")

        return {
            "precio": round(close.iloc[-1], 2),
            "rsi": round(rsi.iloc[-1], 2),
            "tendencia": tendencia,
            "volatilidad": round(np.std(close[-14:]) / close.iloc[-1] * 100, 2),
            "atr": round(atr.iloc[-1], 2),
            "señales": señales
        }
    except:
        return None

def analizar_intradía(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", auto_adjust=True, progress=False)
        if df.empty or len(df) < 30 or float(df["Volume"].squeeze().iloc[-1]) == 0:
            return None, None

        close = df["Close"].squeeze()
        low = df["Low"].squeeze()
        high = df["High"].squeeze()
        volume = df["Volume"].squeeze()

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
        if rsi_val >= RSI_SOBRECOMPRA:
            zona_rsi = "Sobrecompra"
        elif rsi_val <= RSI_SOBREVENTA:
            zona_rsi = "Sobreventa"

        volumen_fuerte = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5

        # riesgo recompensa
        riesgo = close.iloc[-1] - min(low.iloc[-6:-1])
        recompensa = max(high.iloc[-1:].repeat(6).values[0] - close.iloc[-1], 0)
        rr = round(recompensa / riesgo, 2) if riesgo > 0 else "N/A"

        señales = []
        prob_sube = prob_baja = 0

        if cruce_ema == "Cruce alcista EMA9/21":
            señales.append("📈 Cruce alcista EMA9/21")
            prob_sube += 30
        elif cruce_ema == "Cruce bajista EMA9/21":
            señales.append("📉 Cruce bajista EMA9/21")
            prob_baja += 30

        if zona_rsi == "Sobreventa":
            señales.append("🔽 RSI en sobreventa")
            prob_sube += 20
        elif zona_rsi == "Sobrecompra":
            señales.append("🔼 RSI en sobrecompra")
            prob_baja += 20

        if volumen_fuerte:
            señales.append("🔥 Volumen inusualmente alto")
            prob_sube += 10
            prob_baja += 10

        return {
            "señales": señales,
            "prob_sube": prob_sube,
            "prob_baja": prob_baja,
            "rr": rr,
            "direccion": "subida" if prob_sube >= prob_baja else "bajada"
        }, df
    except Exception as e:
        print(f"❌ Error intradía {ticker}: {e}")
        return None, None

def enviar_telegram(mensaje):
            registrar_alerta():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print("⚠️ Error enviando mensaje:", e)

def enviar_imagen(path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as photo:
        requests.post(url, files={"photo": photo}, data={"chat_id": CHAT_ID})

def generar_grafico(df, ticker):
    plt.figure(figsize=(10, 4))
    plt.plot(df["Close"].squeeze(), label="Precio", linewidth=1.2)
    plt.plot(ta.trend.EMAIndicator(df["Close"].squeeze(), window=9).ema_indicator(), label="EMA9")
    plt.plot(ta.trend.EMAIndicator(df["Close"].squeeze(), window=21).ema_indicator(), label="EMA21")
    plt.title(f"{ticker} - Intradía 5m")
    plt.legend()
    plt.grid()
    filename = f"{ticker}_chart.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return filename


def registrar_alerta():
    with open(LOG_ALERTA, "w") as f:
        json.dump({"ultima": datetime.utcnow().isoformat()}, f)

def tiempo_desde_ultima_alerta():
    if not Path(LOG_ALERTA).exists():
        return 9999
    try:
        with open(LOG_ALERTA, "r") as f:
            data = json.load(f)
        ultima = datetime.fromisoformat(data["ultima"])
        return (datetime.utcnow() - ultima).total_seconds() / 60
    except:
        return 9999

# Ejecución principal
if es_mercado_abierto():
    candidatos = []

    for ticker in SYMBOLS:
        print(f"🔍 Evaluando {ticker}...")
        diario = analizar_tecnico_diario(ticker)
        intradia, df = analizar_intradía(ticker)
        if diario and intradia:
            prob_total = max(intradia["prob_sube"], intradia["prob_baja"])
            candidatos.append({
                "ticker": ticker,
                "diario": diario,
                "intradia": intradia,
                "df": df,
                "prob_total": prob_total
            })

    if candidatos:
        mejor = max(candidatos, key=lambda r: r["prob_total"])
        if mejor["prob_total"] >= UMBRAL_ALERTA:
            titulares = get_news_headlines(mejor["ticker"])
            resumen_noticia = analizar_sentimiento_vader(titulares)

            mensaje = f"""🚨 *Mejor oportunidad: {mejor['ticker']}*
{'🟢 Largo' if mejor['intradia']['direccion'] == 'subida' else '🔴 Corto'}

*Señales diarias:*
{chr(10).join(f"- {s}" for s in mejor['diario']['señales'])}

*Señales intradía:*
{chr(10).join(f"- {s}" for s in mejor['intradia']['señales'])}

📈 *Prob. subida:* {mejor['intradia']['prob_sube']}%
📉 *Prob. bajada:* {mejor['intradia']['prob_baja']}%
🎯 *Riesgo/Recompensa estimado:* {mejor['intradia']['rr']}

📰 *Noticias recientes:*
{resumen_noticia}
"""
            enviar_telegram(mensaje)
            registrar_alerta()
            img_path = generar_grafico(mejor["df"], mejor["ticker"])
            enviar_imagen(img_path)
        else:
            print("⚠️ No se detectó ninguna oportunidad relevante.")
    else:
        print("❌ Ningún activo válido para análisis.")
else:
    print("⏳ Mercado cerrado.")
