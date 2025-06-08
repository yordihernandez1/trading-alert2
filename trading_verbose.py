import os
import yfinance as yf
import ta
import requests
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from pathlib import Path
from datetime import datetime, time

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import requests
import ta
import yfinance as yf
from bs4 import BeautifulSoup

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN no está definido. Asegúrate de configurarlo como variable de entorno.")
    logger.error("BOT_TOKEN no está definido. Asegúrate de configurarlo como variable de entorno.")
    raise ValueError("BOT_TOKEN no definido")
if not CHAT_ID:
    raise ValueError("❌ CHAT_ID no está definido. Asegúrate de configurarlo como variable de entorno.")
    logger.error("CHAT_ID no está definido. Asegúrate de configurarlo como variable de entorno.")
    raise ValueError("CHAT_ID no definido")

SYMBOLS = [
    # ACCIONES VOLÁTILES
    "TSLA",     # Tesla
    "NVDA",     # Nvidia
    "AMD",      # AMD – correlación con NVDA
    "META",     # Meta
    "AMZN",     # Amazon

    # ETFs ESTRATÉGICOS
    "SPY",      # S&P 500
    "QQQ",      # Nasdaq 100
    "TQQQ",     # Nasdaq apalancado
    "SQQQ",     # Nasdaq inverso

    # CRIPTO (sin BTC)
    "ETH-USD",  # Ethereum
    "SOL-USD",  # Solana

    # ÍNDICES
    "^GSPC",    # S&P 500
    "^IXIC",    # Nasdaq Composite

    # MATERIAS PRIMAS
    "GC=F"      # Oro
]
@@ -204,51 +195,53 @@ def get_news_headlines_bing(ticker, num_headlines=3):
        return []

def analizar_sentimiento_vader(titulares):
    if not titulares:
        return "Sin noticias recientes."
    sia = SentimentIntensityAnalyzer(lexicon_file="vader_lexicon.txt")
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
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()

        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close)
        sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator()
        sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator()
@@ -197,51 +212,51 @@ def analizar_tecnico_diario(ticker):
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        tendencia = "Alcista" if close.iloc[-1] > close.iloc[-10] else "Bajista"
        soporte, resistencia = encontrar_soporte_resistencia(close)
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
            "soporte": soporte,
            "resistencia": resistencia,
            "señales": señales
        }
    except Exception as e:
@@ -257,80 +250,159 @@ def analizar_tecnico_diario(ticker):
        return None

def analizar_intradía(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", auto_adjust=True, progress=False)

        if df.empty or len(df) < 30 or df["Volume"].iloc[-1].item() == 0:
            return None, None

        close = df["Close"].squeeze()
        low = df["Low"].squeeze()
        high = df["High"].squeeze()
        volume = df["Volume"].squeeze()

        señales = []

        ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close).rsi()

        cruce_ema = "ninguno"
        if ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]:
            cruce_ema = "Cruce alcista EMA9/21"
        elif ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]:
            cruce_ema = "Cruce bajista EMA9/21"
@@ -329,241 +344,263 @@ def analizar_intradía(ticker):

        rsi_val = rsi.iloc[-1]
        if rsi_val >= RSI_SOBRECOMPRA:
            zona_rsi = "Sobrecompra"
        elif rsi_val <= RSI_SOBREVENTA:
            zona_rsi = "Sobreventa"
        else:
            zona_rsi = "Neutral"

        señales.append(f"RSI en zona {zona_rsi.lower()} ({round(rsi_val, 1)})")

        media_volumen = volume.rolling(20).mean().iloc[-1]
        volumen_actual = volume.iloc[-1]
        volumen_fuerte = volumen_actual > media_volumen * 1.5

        # 🔍 Volumen proporcional
        multiplicador = volumen_actual / media_volumen if media_volumen > 0 else 1
        if multiplicador >= 2:
            señales.append("📊 Volumen x2 o más")
            vol_score = 15
        elif multiplicador >= 1.5:
            señales.append("📊 Volumen moderadamente alto")
            vol_score = 10
        elif multiplicador >= 1.2:
            señales.append("📊 Volumen ligeramente alto")
            vol_score = 5
        else:
            vol_score = 0

        precio_actual = close.iloc[-1]
        minimo_reciente = low[-6:-1].min()
        maximo_esperado = high[-6:].max()

        riesgo = round(precio_actual - minimo_reciente, 2)
        recompensa = round(maximo_esperado - precio_actual, 2)

        if riesgo <= 0 or recompensa <= 0:
            rr = "No válido"
            tiempo_estimado = "N/A"
        else:
            rr = round(recompensa / riesgo, 2)
            velas = close[-6:]
            cambios = velas.diff().dropna()

            if cruce_ema == "Cruce alcista EMA9/21":
                velocidad = cambios[cambios > 0].mean()
            else:
                velocidad = abs(cambios[cambios < 0].mean())

            tiempo_estimado = round((recompensa / velocidad) * 5) if velocidad and velocidad > 0 else "N/A"

        prob_sube = prob_baja = 0

        # Señales principales
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

        # Volumen influye en ambas direcciones
        prob_sube += vol_score
        prob_baja += vol_score

        # Última vela fuerte o débil
        ultima = df.iloc[-1]
        cuerpo = abs(ultima["Close"] - ultima["Open"])
        rango = ultima["High"] - ultima["Low"]
        fuerza = cuerpo / rango if rango > 0 else 0

        if ultima["Close"] > ultima["Open"]:
            if fuerza > 0.6:
                señales.append("🟩 Vela alcista fuerte")
                prob_sube += 10
            elif fuerza > 0.3:
                señales.append("🟩 Vela alcista moderada")
                prob_sube += 5
        elif ultima["Close"] < ultima["Open"]:
            if fuerza > 0.6:
                señales.append("🟥 Vela bajista fuerte")
                prob_baja += 10
            elif fuerza > 0.3:
                señales.append("🟥 Vela bajista moderada")
                prob_baja += 5

        # La dirección se mantiene como antes
        direccion = "subida" if prob_sube >= prob_baja else "bajada"

        return {
            "señales": señales,
            "prob_sube": prob_sube,
            "prob_baja": prob_baja,
            "rr": rr,
            "tiempo_estimado": tiempo_estimado,
            "direccion": direccion
        }, df

    except Exception as e:
        print(f"❌ Error intradía {ticker}: {e}")
        logger.error("Error intradía %s: %s", ticker, e)
        return None, None

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Error al enviar mensaje Telegram: {res.status_code} - {res.text}")
            logger.warning("Error al enviar mensaje Telegram: %s - %s", res.status_code, res.text)
        else:
            print("✅ Mensaje enviado correctamente a Telegram.")
            logger.info("Mensaje enviado correctamente a Telegram")
    except Exception as e:
        print("⚠️ Error enviando mensaje a Telegram:", e)
        logger.error("Error enviando mensaje a Telegram: %s", e)

def enviar_imagen(path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as photo:
        requests.post(url, files={"photo": photo}, data={"chat_id": CHAT_ID})
        try:
            res = requests.post(url, files={"photo": photo}, data={"chat_id": CHAT_ID}, timeout=10)
            if res.status_code != 200:
                logger.warning("Error al enviar imagen a Telegram: %s - %s", res.status_code, res.text)
            else:
                logger.info("Imagen enviada correctamente a Telegram")
