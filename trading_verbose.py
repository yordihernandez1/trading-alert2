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

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
UMBRAL_ALERTA = 50
LOG_ALERTA = "ultima_alerta.json"
TIEMPO_RESUMEN_MINUTOS = 30

def registrar_alerta():
    try:
        print("📝 Registrando alerta en ultima_alerta.json...")
        with open(LOG_ALERTA, "w") as f:
            json.dump({"ultima": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        print(f"⚠️ No se pudo registrar la alerta: {e}")

def tiempo_desde_ultima_alerta():
    path = Path(LOG_ALERTA)
    if not path.exists():
        return 9999
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ultima = datetime.fromisoformat(data.get("ultima", "2000-01-01T00:00:00"))
        delta = (datetime.utcnow() - ultima).total_seconds() / 60
        print(f"⏱ Última alerta enviada hace {int(delta)} minutos.")
        return delta
    except Exception as e:
        print(f"⚠️ Error leyendo archivo de alerta: {e}")
        return 9999

LOG_RESUMEN = "ultimo_resumen.json"

def registrar_resumen():
    try:
        with open(LOG_RESUMEN, "w") as f:
            json.dump({"ultimo": datetime.utcnow().isoformat()}, f)
    except Exception as e:
        print(f"⚠️ No se pudo registrar el resumen: {e}")

def tiempo_desde_ultimo_resumen():
    path = Path(LOG_RESUMEN)
    if not path.exists():
        return 9999
    try:
        with open(path, "r") as f:
            data = json.load(f)
        ultimo = datetime.fromisoformat(data.get("ultimo", "2000-01-01T00:00:00"))
        delta = (datetime.utcnow() - ultimo).total_seconds() / 60
        print(f"📊 Último resumen enviado hace {int(delta)} minutos.")
        return delta
    except Exception as e:
        print(f"⚠️ Error leyendo archivo de resumen: {e}")
        return 9999

def encontrar_soporte_resistencia(close, periodo=14):
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)

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

        señales = []  # ✅ inicializado al principio

        ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, window=21).ema_indicator()
        rsi = ta.momentum.RSIIndicator(close).rsi()

        cruce_ema = "ninguno"
        if ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]:
            cruce_ema = "Cruce alcista EMA9/21"
        elif ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]:
            cruce_ema = "Cruce bajista EMA9/21"

        rsi_val = rsi.iloc[-1]
        if rsi_val >= RSI_SOBRECOMPRA:
            zona_rsi = "Sobrecompra"
        elif rsi_val <= RSI_SOBREVENTA:
            zona_rsi = "Sobreventa"
        else:
            zona_rsi = "Neutral"

        señales.append(f"RSI en zona {zona_rsi.lower()} ({round(rsi_val, 1)})")

        volumen_fuerte = volume.iloc[-1] > volume.rolling(20).mean().iloc[-1] * 1.5

        riesgo = close.iloc[-1] - min(low.iloc[-6:-1])
        recompensa = max(high.iloc[-1:].repeat(6).values[0] - close.iloc[-1], 0)
        rr = round(recompensa / riesgo, 2) if riesgo > 0 else "N/A"

        velas = close[-6:]
        cambios = velas.diff().dropna()

        if riesgo > 0:
            if cruce_ema == "Cruce alcista EMA9/21":
                velocidad = cambios[cambios > 0].mean()
            else:
                velocidad = abs(cambios[cambios < 0].mean())

            if velocidad and velocidad > 0:
                tiempo_estimado = round((recompensa / velocidad) * 5)
            else:
                tiempo_estimado = "N/A"
        else:
            tiempo_estimado = "N/A"

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
            "tiempo_estimado": tiempo_estimado,
            "direccion": "subida" if prob_sube >= prob_baja else "bajada"
        }, df
    except Exception as e:
        print(f"❌ Error intradía {ticker}: {e}")
        return None, None


def enviar_telegram(mensaje):
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
            entrada = mejor["diario"]["precio"]

            if mejor["intradia"]["direccion"] == "subida":
                stop = mejor["diario"]["soporte"]
                take_profit = entrada + (entrada - stop) * 2
            else:
                stop = mejor["diario"]["resistencia"]
                take_profit = entrada - (stop - entrada) * 2

    entrada = round(entrada, 2)
    stop = round(stop, 2)
    take_profit = round(take_profit, 2)

    titulares = get_news_headlines(mejor["ticker"])
    resumen_noticia = analizar_sentimiento_vader(titulares)

    mensaje = f"""🚨 *Mejor oportunidad: {mejor['ticker']}*
{'Largo' if mejor['intradia']['direccion'] == 'subida' else 'Corto'}

*Señales diarias:*
{chr(10).join(f"- {s}" for s in mejor['diario']['señales'])}

*Señales intradía:*
{chr(10).join(f"- {s}" for s in mejor['intradia']['señales'])}

📈 *Prob. subida:* {mejor['intradia']['prob_sube']}%  
📉 *Prob. bajada:* {mejor['intradia']['prob_baja']}%  
🎯 *Riesgo/Recompensa estimado:* {mejor['intradia']['rr']}  
⏳ *Tiempo estimado para alcanzar ganancia:* {mejor['intradia']['tiempo_estimado']} min  
📊 *Soporte:* {mejor['diario']['soporte']} | 📈 *Resistencia:* {mejor['diario']['resistencia']}  
💵 *Entrada sugerida:* {entrada}  
🛑 *Stop Loss:* {stop}  
🎯 *Take Profit:* {take_profit}  

📰 *Noticias recientes:*
{resumen_noticia}
"""

    enviar_telegram(mensaje)
    registrar_alerta()
    img_path = generar_grafico(mejor["df"], mejor["ticker"])
    enviar_imagen(img_path)

else:
    minutos_alerta = tiempo_desde_ultima_alerta()
    minutos_resumen = tiempo_desde_ultimo_resumen()

    if minutos_alerta >= TIEMPO_RESUMEN_MINUTOS and minutos_resumen >= TIEMPO_RESUMEN_MINUTOS:
        resumen = "⏱ *Sin alertas en los últimos 30 minutos.*\n\n*Probabilidades actuales:*\n\n"
        for c in candidatos:
            resumen += f"{c['ticker']}: 📈 {c['intradia']['prob_sube']}% subida | 📉 {c['intradia']['prob_baja']}% bajada\n"
        enviar_telegram(resumen)
        registrar_resumen()
    else:
        print("🕒 Aún dentro del margen de espera para resumen.")
