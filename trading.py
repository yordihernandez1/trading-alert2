import os
import yfinance as yf
import ta
import requests

BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = int(os.environ.get("CHAT_ID"))
TICKER = "AAPL"
RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30

try:
    data = yf.download(TICKER, period="3mo", interval="1d")
except Exception as e:
    print("❌ Error al descargar datos:", e)
    exit()

if data.empty:
    print(f"❌ No se encontraron datos para {TICKER}")
    exit()

close = data["Close"].squeeze()
data["rsi"] = ta.momentum.RSIIndicator(close=close).rsi()
macd = ta.trend.MACD(close=close)
data["macd"] = macd.macd()
data["macd_signal"] = macd.macd_signal()
data["sma_50"] = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
data["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()

try:
    rsi = data["rsi"].iloc[-1].item()
    macd_val = data["macd"].iloc[-1].item()
    macd_sig = data["macd_signal"].iloc[-1].item()
    sma_50 = data["sma_50"].iloc[-1].item()
    sma_200 = data["sma_200"].iloc[-1].item()
    precio = data["Close"].iloc[-1].item()
    cierre_anterior = data["Close"].iloc[-2].item()
except Exception as e:
    print("❌ Error al leer indicadores:", e)
    exit()

señales = []

if rsi > RSI_SOBRECOMPRA:
    señales.append("RSI elevado → posible sobrecompra")
elif rsi < RSI_SOBREVENTA:
    señales.append("RSI bajo → posible sobreventa")
else:
    señales.append("RSI normal → No está ni en sobreventa ni en sobrecompra")

if macd_val < macd_sig:
    señales.append("MACD cruzando a la baja")
if precio < sma_50:
    señales.append("Precio por debajo de la SMA 50")
if sma_50 < sma_200:
    señales.append("SMA 50 por debajo de SMA 200 → tendencia bajista")
if precio < cierre_anterior:
    señales.append("Última vela cerró en rojo")

mensaje = (
    f"📉 ANÁLISIS TÉCNICO – {TICKER}\n\n"
    f"💰 Precio: {round(precio, 2)} USD\n"
    f"📊 RSI: {round(rsi, 2)}\n"
    f"📈 MACD: {round(macd_val, 2)} / Señal: {round(macd_sig, 2)}\n"
    f"📉 SMA 50: {round(sma_50, 2)}\n"
    f"📉 SMA 200: {round(sma_200, 2)}\n\n"
    f"📌 Señales detectadas:\n" + "\n".join("- " + s for s in señales) + "\n\n"
    "¿Es buen momento para una entrada en corto?"
)

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": mensaje
}
response = requests.post(url, data=payload)

if response.status_code == 200:
    print("✅ Mensaje enviado correctamente")
else:
    print("❌ Error al enviar mensaje:", response.text)
