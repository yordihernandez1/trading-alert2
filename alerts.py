import os
import time
import io
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta
from bs4 import BeautifulSoup
from telegram import Bot
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.data import path as nltk_path

nltk_path.append('./')
sid = SentimentIntensityAnalyzer(lexicon_file='vader_lexicon.txt')

API_KEY = os.getenv("FINNHUB_API_KEY")
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SUMMARY_FILE = "last_alert.txt"

SYMBOLS = {
    "AAPL": "stock",
    "MSFT": "stock",
    "NVDA": "stock",
    "TSLA": "stock",
    "AMZN": "stock",
    "SPY": "etf",
    "QQQ": "etf",
    "IWM": "etf",
    "BINANCE:BTCUSDT": "crypto",
    "BINANCE:ETHUSDT": "crypto",
    "OANDA:WTICOUSD": "forex",
    "OANDA:XAUUSD": "forex"
}

def get_candles(symbol, resolution, count=150):
    url = f"https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "count": count,
        "token": API_KEY
    }
    res = requests.get(url, params=params).json()
    if res.get("s") != "ok":
        raise Exception(f"Error al obtener velas para {symbol}")
    return pd.DataFrame({
        "t": pd.to_datetime(res["t"], unit="s"),
        "o": res["o"],
        "h": res["h"],
        "l": res["l"],
        "c": res["c"],
        "v": res["v"]
    })

def analyze_daily(symbol):
    df = get_candles(symbol, resolution="D")
    df.set_index("t", inplace=True)
    df["RSI"] = np.ravel(ta.momentum.RSIIndicator(df["c"]).rsi())
    macd = ta.trend.MACD(df["c"])
    df["MACD"] = np.ravel(macd.macd())
    df["MACD_SIGNAL"] = np.ravel(macd.macd_signal())
    df["SMA50"] = df["c"].rolling(window=50).mean()
    df["SMA200"] = df["c"].rolling(window=200).mean()
    df["ATR"] = np.ravel(ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range())
    last = df.iloc[-1]
    return df, {
        "price": last["c"],
        "rsi": last["RSI"],
        "macd": last["MACD"],
        "macd_signal": last["MACD_SIGNAL"],
        "sma50": last["SMA50"],
        "sma200": last["SMA200"],
        "atr": last["ATR"],
        "trend": "up" if last["SMA50"] > last["SMA200"] else "down",
        "overbought": last["RSI"] > 70,
        "oversold": last["RSI"] < 30
    }

def analyze_intraday(symbol):
    df = get_candles(symbol, resolution="5")
    df.set_index("t", inplace=True)
    df["EMA9"] = df["c"].ewm(span=9).mean()
    df["EMA21"] = df["c"].ewm(span=21).mean()
    df["RSI"] = np.ravel(ta.momentum.RSIIndicator(df["c"]).rsi())
    df["VOLUME_MEAN"] = df["v"].rolling(window=20).mean()
    last = df.iloc[-1]
    prev = df.iloc[-2]
    return df, {
        "crossover": prev["EMA9"] < prev["EMA21"] and last["EMA9"] > last["EMA21"],
        "rsi_extreme": last["RSI"] < 30 or last["RSI"] > 70,
        "volume_spike": last["v"] > 1.5 * last["VOLUME_MEAN"]
    }

def analyze_sentiment(symbol):
    q = symbol.replace("BINANCE:", "").replace("OANDA:", "")
    url = f"https://www.bing.com/news/search?q={q}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        headlines = [a.get_text() for a in soup.find_all('a') if a.get_text()]
        scores = [sid.polarity_scores(h)['compound'] for h in headlines[:5]]
        avg = np.mean(scores) if scores else 0
        return 'positivo' if avg > 0.2 else 'negativo' if avg < -0.2 else 'neutro'
    except:
        return 'desconocido'

def plot_chart(df, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(df["c"], label="Close")
    plt.plot(df["EMA9"], label="EMA9")
    plt.plot(df["EMA21"], label="EMA21")
    plt.title(f"{symbol} Intradía")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def send_telegram_alert(text, image=None):
    bot = Bot(token=BOT_TOKEN)
    if image:
        bot.send_photo(chat_id=CHAT_ID, photo=image, caption=text)
    else:
        bot.send_message(chat_id=CHAT_ID, text=text)

def time_since_last_alert():
    if not os.path.exists(SUMMARY_FILE):
        return float("inf")
    with open(SUMMARY_FILE, "r") as f:
        last = float(f.read().strip())
    return (datetime.datetime.now().timestamp() - last) / 60

def update_last_alert_time():
    with open(SUMMARY_FILE, "w") as f:
        f.write(str(datetime.datetime.now().timestamp()))

def run_alert_bot():
    for symbol in SYMBOLS:
        try:
            df_daily, daily = analyze_daily(symbol)
            atr_pct = daily["atr"] / daily["price"]
            if atr_pct > 0.015:
                continue
            entry = daily["price"]
            stop = entry * 0.98
            target = entry + abs(entry - stop) * 1.33
            rr = (target - entry) / (entry - stop)
            df_intra, intra = analyze_intraday(symbol)
            if not (intra["crossover"] or intra["rsi_extreme"] or intra["volume_spike"]):
                continue
            sentiment = analyze_sentiment(symbol)
            chart = plot_chart(df_intra, symbol)
            text = (
                f"📈 Alerta de Trading: {symbol}\n"
                f"Precio: {entry:.2f}\n"
                f"SL: {stop:.2f} | TP: {target:.2f}\n"
                f"Riesgo/Recompensa: {rr:.2f}\n"
                f"RSI: {daily['rsi']:.1f} | MACD: {daily['macd']:.2f}\n"
                f"Tendencia: {daily['trend']}\n"
                f"Sentimiento de noticias: {sentiment}"
            )
            send_telegram_alert(text, chart)
            update_last_alert_time()
            return
        except Exception as e:
            print(f"Error con {symbol}: {e}")
        time.sleep(2)

    if time_since_last_alert() >= 30:
        send_telegram_alert(f"⏳ No se detectaron señales en los últimos 30 minutos. ({datetime.datetime.now().strftime('%H:%M')})")
        update_last_alert_time()
