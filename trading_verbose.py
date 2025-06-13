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
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Variables de entorno
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN:
    logger.error("BOT_TOKEN no está definido. Asegúrate de configurarlo como variable de entorno.")
    raise ValueError("❌ BOT_TOKEN no está definido.")

if not CHAT_ID:
    logger.error("CHAT_ID no está definido. Asegúrate de configurarlo como variable de entorno.")
    raise ValueError("❌ CHAT_ID no está definido.")

# Configuración
SYMBOLS = ["TSLA", "NVDA", "AMD", "META", "AMZN", "SPY", "QQQ", "TQQQ", "SQQQ", "ETH-USD", "SOL-USD", "^GSPC", "^IXIC", "GC=F"]
CRIPTOS = ["ETH-USD", "SOL-USD"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
UMBRAL_ALERTA = 30
LOG_ALERTA = "ultima_alerta.json"
LOG_RESUMEN = "ultimo_resumen.json"
TIEMPO_RESUMEN_MINUTOS = 30

# Funciones auxiliares

def registrar_alerta():
    try:
        with open(LOG_ALERTA, "w") as f:
            json.dump({"ultima": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        logger.warning(f"No se pudo registrar la alerta: {e}")

def tiempo_desde_ultima_alerta():
    path = Path(LOG_ALERTA)
    if not path.exists():
        return 9999
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ultima = datetime.fromisoformat(data.get("ultima"))
        return (datetime.utcnow() - ultima).total_seconds() / 60
    except:
        return 9999

def registrar_resumen():
    try:
        with open(LOG_RESUMEN, "w") as f:
            json.dump({"ultimo": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        logger.warning(f"No se pudo registrar el resumen: {e}")

def tiempo_desde_ultimo_resumen():
    path = Path(LOG_RESUMEN)
    if not path.exists():
        return 9999
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ultimo = datetime.fromisoformat(data.get("ultimo"))
        return (datetime.utcnow() - ultimo).total_seconds() / 60
    except:
        return 9999

def es_mercado_abierto():
    ahora = datetime.utcnow()
    if ahora.weekday() >= 5:
        return False
    hora_actual = ahora.time()
    return time(14, 30) <= hora_actual <= time(21, 0)

def encontrar_soporte_resistencia(close, periodo=14):
    return round(min(close[-periodo:]), 2), round(max(close[-periodo:]), 2)

def get_news_headlines(ticker, num_headlines=3):
    url = f"https://www.google.com/search?q={ticker}+stock&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}
    try:
        res = requests.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.text, 'html.parser')
        return [el.text.strip() for el in soup.select('div.JheGif.nDgy9d')[:num_headlines]]
    except:
        return []

def get_news_headlines_bing(ticker, num_headlines=3):
    url = f"https://www.bing.com/news/search?q={ticker}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.text, 'html.parser')
        return [item.text.strip() for item in soup.select("a.title")[:num_headlines]]
    except:
        return []

def analizar_sentimiento_vader(titulares):
    if not titulares:
        return "Sin noticias recientes."
    sia = SentimentIntensityAnalyzer(lexicon_file="vader_lexicon.txt")
    score = sia.polarity_scores(" ".join(titulares))["compound"]
    sentimiento = "Positivo" if score >= 0.05 else "Negativo" if score <= -0.05 else "Neutro"
    return f"{'; '.join(titulares[:2])}\nSentimiento general: {sentimiento}"

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            logger.warning(f"Error al enviar mensaje Telegram: {res.status_code} - {res.text}")
        else:
            logger.info("Mensaje enviado correctamente a Telegram")
    except Exception as e:
        logger.error(f"Error enviando mensaje a Telegram: {e}")

def enviar_imagen(path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as photo:
            res = requests.post(url, files={"photo": photo}, data={"chat_id": CHAT_ID}, timeout=10)
            if res.status_code != 200:
                logger.warning(f"Error al enviar imagen: {res.status_code} - {res.text}")
    except Exception as e:
        logger.error(f"Error al enviar imagen a Telegram: {e}")

def generar_grafico(df, ticker):
    plt.figure(figsize=(10, 4))
    close = df["Close"].squeeze()
    ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
    plt.plot(close, label="Precio", linewidth=1.2)
    plt.plot(ema9, label="EMA9")
    plt.plot(ema21, label="EMA21")
    plt.title(f"{ticker} - Intradía 5m")
    plt.legend()
    plt.grid()
    filename = f"{ticker}_chart.png"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    return filename

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
        logger.error(f"Error en análisis técnico diario de {ticker}: {e}")
        return None


def analizar_intradia(ticker):
    try:
        df = yf.download(ticker, period="2d", interval="5m", auto_adjust=True, progress=False)
        if df.empty or len(df) < 30:
            return None, None

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()
        ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close).rsi()

        prob_sube, prob_baja = 0, 0
        rr = "1:1"
        tiempo_estimado = 30
        señales = []

        if ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]:
            señales.append("Cruce alcista EMA9/21")
            prob_sube += 10
        elif ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]:
            señales.append("Cruce bajista EMA9/21")
            prob_baja += 10

        rsi_val = rsi.iloc[-1]
        if rsi_val >= RSI_SOBRECOMPRA:
            señales.append("RSI en sobrecompra")
            prob_baja += 10
        elif rsi_val <= RSI_SOBREVENTA:
            señales.append("RSI en sobreventa")
            prob_sube += 10

        media_volumen = volume.rolling(20).mean().iloc[-1]
        volumen_actual = volume.iloc[-1]
        if media_volumen > 0:
            multiplicador = volumen_actual / media_volumen
            if multiplicador >= 2:
                señales.append("Volumen x2 o más")
                prob_sube += 10
            elif multiplicador >= 1.5:
                señales.append("Volumen moderadamente alto")
                prob_sube += 5

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
        logger.error(f"Error en análisis intradía de {ticker}: {e}")
        return None, None


if __name__ == "__main__":
    if not es_mercado_abierto():
        print("El mercado está cerrado.")
        exit()

    candidatos = []
    for ticker in SYMBOLS:
        print(f"Analizando {ticker}...")
        diario = analizar_tecnico_diario(ticker)
        intradia, df = analizar_intradia(ticker)

        if not diario or not intradia:
            continue

        entrada = diario["precio"]
        atr = diario["atr"]

        if atr > entrada * 0.015:
            continue

        sl_pct = 0.002
        tp_pct = 0.003
        stop = entrada * (1 - sl_pct) if intradia["direccion"] == "subida" else entrada * (1 + sl_pct)
        take_profit = entrada * (1 + tp_pct) if intradia["direccion"] == "subida" else entrada * (1 - tp_pct)

        stop_pct = abs(entrada - stop) / entrada * 100
        take_pct = abs(take_profit - entrada) / entrada * 100

        if stop_pct > 2 or take_pct < stop_pct * 1.33:
            continue

        prob_total = max(intradia["prob_sube"], intradia["prob_baja"])
        prob_total = int((prob_total / 40) * 100)

        candidatos.append({
            "ticker": ticker,
            "diario": diario,
            "intradia": intradia,
            "df": df,
            "entrada": round(entrada, 2),
            "stop": round(stop, 2),
            "take_profit": round(take_profit, 2),
            "stop_pct": round(stop_pct, 2),
            "tp_pct": round(take_pct, 2),
            "prob_total": prob_total
        })

    minutos_alerta = tiempo_desde_ultima_alerta()
    minutos_resumen = tiempo_desde_ultimo_resumen()

    if minutos_alerta >= TIEMPO_RESUMEN_MINUTOS and minutos_resumen >= minutos_alerta and candidatos:
        resumen = "
".join([
            f"{c['ticker']} | Prob: {c['prob_total']}% | Dirección: {'↑' if c['intradia']['direccion'] == 'subida' else '↓'} | TP: {c['tp_pct']}% | SL: {c['stop_pct']}%"
            for c in sorted(candidatos, key=lambda x: x["prob_total"], reverse=True)
        ]))
        enviar_telegram(f"📊 *Resumen de oportunidades*

{resumen}")
        registrar_resumen()

    if candidatos:
        mejor = max(candidatos, key=lambda x: x["prob_total"])
        if mejor["prob_total"] >= UMBRAL_ALERTA:
            titulares = get_news_headlines(mejor["ticker"])
            if not titulares:
                titulares = get_news_headlines_bing(mejor["ticker"])
            resumen_noticia = analizar_sentimiento_vader(titulares)
            mensaje = f"""🚨 *Mejor oportunidad: {mejor['ticker']}*
{'Largo' if mejor['intradia']['direccion'] == 'subida' else 'Corto'}

*Señales diarias:*
{chr(10).join(f'- {s}' for s in mejor['diario']['señales'])}

*Señales intradía:*
{chr(10).join(f'- {s}' for s in mejor['intradia']['señales'])}

📈 *Prob. subida:* {mejor['intradia']['prob_sube']}%
📉 *Prob. bajada:* {mejor['intradia']['prob_baja']}%
🎯 *Riesgo/Recompensa estimado:* {mejor['intradia']['rr']}
⏳ *Tiempo estimado para alcanzar ganancia:* {mejor['intradia']['tiempo_estimado']} min

💵 *Entrada sugerida:* {mejor['entrada']}
🔻 *Stop Loss:* {mejor['stop']} ({mejor['stop_pct']}%)
🎯 *Take Profit:* {mejor['take_profit']} ({mejor['tp_pct']}%)
📊 *Soporte:* {mejor['diario']['soporte']} | 📈 *Resistencia:* {mejor['diario']['resistencia']}

📰 *Noticias recientes:*
{resumen_noticia}
"""
            enviar_telegram(mensaje)
            registrar_alerta()
            img_path = generar_grafico(mejor["df"], mejor["ticker"])
            enviar_imagen(img_path)
