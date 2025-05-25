import os
import yfinance as yf
import ta
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class CustomSentimentIntensityAnalyzer(SentimentIntensityAnalyzer):
    def __init__(self):
        lexicon_path = os.path.join(os.path.dirname(__file__), "vader_lexicon.txt")
        super().__init__(lexicon_file=lexicon_path)

def cargar_analizador_personalizado():
    return CustomSentimentIntensityAnalyzer()

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("Faltan BOT_TOKEN o CHAT_ID en las variables de entorno")

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
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [
            g.select_one('div.JheGif.nDgy9d').text
            for g in soup.select('div.dbsr')[:num_headlines]
            if g.select_one('div.JheGif.nDgy9d')
        ]
        return headlines
    except:
        return []

def analizar_sentimiento_vader(titulares):
    if not titulares:
        return "Sin noticias recientes."
    analizador = cargar_analizador_personalizado()
    textos = " ".join(titulares)
    scores = analizador.polarity_scores(textos)
    compound = scores["compound"]
    sentimiento = (
        "Positivo" if compound >= 0.05
        else "Negativo" if compound <= -0.05
        else "Neutro"
    )
    resumen = "; ".join(titulares[:2])
    return f"{resumen}\nSentimiento general: {sentimiento}"

def detectar_tendencia(close):
    if len(close) < 10:
        return "Insuficiente"
    actual = float(close.iloc[-1])
    pasado = float(close.iloc[-10])
    return "Alcista" if actual > pasado else "Bajista" if actual < pasado else "Lateral"

def encontrar_soporte_resistencia(close, periodo=14):
    soporte = min(close[-periodo:])
    resistencia = max(close[-periodo:])
    return round(soporte, 2), round(resistencia, 2)

def analizar_ticker(ticker):
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < 20:
            return None
    except:
        return None

    if any(col not in df.columns for col in ["High", "Low", "Close"]):
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    try:
        rsi = ta.momentum.RSIIndicator(close=close).rsi().squeeze()
        macd_obj = ta.trend.MACD(close=close)
        macd = macd_obj.macd().squeeze()
        macd_signal = macd_obj.macd_signal().squeeze()
        sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator().squeeze()
        sma_200 = ta.trend.SMAIndicator(close=close, window=200).sma_indicator().squeeze()
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range().squeeze()
    except:
        return None

    try:
        precio = float(close.iloc[-1])
        cierre_ant = float(close.iloc[-2])
        rsi_val = float(rsi.iloc[-1])
        macd_val = float(macd.iloc[-1])
        macd_sig_val = float(macd_signal.iloc[-1])
        sma_50_val = float(sma_50.iloc[-1])
        sma_200_val = float(sma_200.iloc[-1])
        atr_val = float(atr.iloc[-1])
    except:
        return None

    señales_bajistas = []
    señales_alcistas = []
    score_bajista = 0
    score_alcista = 0

    if rsi_val > RSI_SOBRECOMPRA:
        señales_bajistas.append("RSI en sobrecompra → posible venta")
        score_bajista += 1
    if macd_val < macd_sig_val:
        señales_bajistas.append("MACD cruzando a la baja")
        score_bajista += 1
    if precio < sma_50_val:
        señales_bajistas.append("Precio por debajo de la SMA 50")
        score_bajista += 1
    if sma_50_val < sma_200_val:
        señales_bajistas.append("SMA 50 por debajo de SMA 200")
        score_bajista += 1
    if precio < cierre_ant:
        señales_bajistas.append("Última vela cerró en rojo")
        score_bajista += 1

    if rsi_val < RSI_SOBREVENTA:
        señales_alcistas.append("RSI en sobreventa → posible compra")
        score_alcista += 1
    if macd_val > macd_sig_val:
        señales_alcistas.append("MACD cruzando al alza")
        score_alcista += 1
    if precio < sma_50_val and sma_50_val < sma_200_val:
        señales_alcistas.append("Precio bajo con posible recuperación")
        score_alcista += 1
    if precio > cierre_ant:
        señales_alcistas.append("Última vela cerró en verde")
        score_alcista += 1

    tendencia = detectar_tendencia(close)
    retorno_7d = ((close.iloc[-1] / close.iloc[-7]) - 1) * 100
    volatilidad = np.std(close[-14:]) / close.iloc[-1] * 100
    soporte, resistencia = encontrar_soporte_resistencia(close)

    return {
        "ticker": ticker,
        "precio": round(precio, 2),
        "rsi": round(rsi_val, 2),
        "score_bajista": score_bajista,
        "score_alcista": score_alcista,
        "señales_bajistas": señales_bajistas,
        "señales_alcistas": señales_alcistas,
        "tendencia": tendencia,
        "retorno_7d": round(retorno_7d, 2),
        "volatilidad": round(volatilidad, 2),
        "atr": round(atr_val, 2),
        "soporte": soporte,
        "resistencia": resistencia
    }

def enviar_mensaje_telegram(token, chat_id, mensaje):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "Markdown"
    }

    print("📤 Enviando mensaje a Telegram:")
    print("BOT_TOKEN:", token[:10] + "..." if token else "No definido")
    print("CHAT_ID:", chat_id)
    print("Mensaje (primeros 300 caracteres):")
    print(mensaje[:300])

    try:
        resp = requests.post(url, data=payload, timeout=10)
        print("Código de respuesta:", resp.status_code)
        if resp.status_code == 200:
            print("✅ Mensaje enviado correctamente.")
        else:
            print("❌ Error al enviar mensaje:", resp.text)
    except Exception as e:
        print("⚠️ Excepción al intentar enviar mensaje:", e)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": mensaje,
        "parse_mode": "Markdown"
    }
    try:
        print("Enviando mensaje a Telegram...")
        resp = requests.post(url, data=payload, timeout=10)
        print("Código de respuesta:", resp.status_code)
        if resp.status_code == 200:
            print("Mensaje enviado correctamente.")
        else:
            print("Error al enviar mensaje:", resp.text)
    except Exception as e:
        print("Excepción al intentar enviar mensaje:", e)

# Ejecución principal
print("🔍 Analizando símbolos:", symbols)
resultados = []
for sym in symbols:
    print(f"⏳ Descargando datos para {sym}...")
    res = analizar_ticker(sym)
    if res:
        print(f"✅ {sym} analizado correctamente")
        resultados.append(res)
    else:
        print(f"⚠️ {sym} no tiene datos suficientes o falló el análisis")
resultados = [r for r in resultados if r]

print(f"📊 Total de símbolos con resultados válidos: {len(resultados)}")
if not resultados:
    mensaje = "No se pudo analizar ningún activo."
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

    print("🧠 Generando mensaje de alerta...")
mensaje = f'''
OPORTUNIDAD DESTACADA: {mejor['ticker']}
Precio: {mejor['precio']} USD
RSI: {mejor['rsi']}

Entrada sugerida: en {tipo}

Análisis técnico cualitativo:
- Tendencia general: {mejor['tendencia']}

Datos cuantitativos:
- Volatilidad 14d: {mejor['volatilidad']}%
- Retorno 7d: {mejor['retorno_7d']}%
- ATR: {mejor['atr']}

Señales detectadas:
''' + "\n".join(f"- {s}" for s in señales) + f'''

Soporte: {mejor['soporte']} | Resistencia: {mejor['resistencia']}

Noticias recientes:
{resumen_noticia}

Pregunta para ChatGPT:
Con base en los siguientes datos técnicos para {mejor['ticker']}:
- RSI: {mejor['rsi']}
- Tendencia: {mejor['tendencia']}
- Señales: {', '.join(señales)}
- Sentimiento: {sentimiento}
- Soporte: {mejor['soporte']}, Resistencia: {mejor['resistencia']}
¿Consideras que es una buena oportunidad para entrar en {'compra' if tipo == 'largo' else 'venta'}? Justifica brevemente.
'''

    print("🚀 Enviando mensaje a Telegram...")
    enviar_mensaje_telegram(BOT_TOKEN, CHAT_ID, mensaje)
