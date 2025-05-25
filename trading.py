import os
import yfinance as yf
import ta
import requests
from datetime import datetime
import numpy as np

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))

hora_utc = datetime.utcnow().hour
if hora_utc < 14 or hora_utc >= 21:
    print("⏱️ Fuera del horario de envío. No se envía mensaje.")
    exit()

symbols = ["TSLA", "AAPL", "NVDA", "AMD", "BTC-USD", "^IXIC"]
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

def detectar_tendencia(close):
    try:
        val_actual = close.iloc[-1].item()
        val_antiguo = close.iloc[-10].item()
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
        rsi = data["rsi"].iloc[-1]
        macd_val = data["macd"].iloc[-1]
        macd_sig = data["macd_signal"].iloc[-1]
        sma_50 = data["sma_50"].iloc[-1]
        sma_200 = data["sma_200"].iloc[-1]
        precio = data["Close"].iloc[-1]
        cierre_anterior = data["Close"].iloc[-2]
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

resultados = [analizar_ticker(sym) for sym in symbols]
resultados = [r for r in resultados if r]

if not resultados:
    mensaje = "❌ No se pudo analizar ningún activo."
else:
    mejor = max(resultados, key=lambda r: max(r["score_bajista"], r["score_alcista"]))
    tipo = "corto 🔻" if mejor["score_bajista"] >= mejor["score_alcista"] else "largo 🚀"
    señales = mejor["señales_bajistas"] if tipo.startswith("corto") else mejor["señales_alcistas"]

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
""" + "\n".join(f"- {s}" for s in señales)

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
