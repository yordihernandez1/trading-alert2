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
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN no está definido. Asegúrate de configurarlo como variable de entorno.")
if not CHAT_ID:
    raise ValueError("❌ CHAT_ID no está definido. Asegúrate de configurarlo como variable de entorno.")

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

RSI_SOBRECOMPRA = 70
RSI_SOBREVENTA = 30
UMBRAL_ALERTA = 30
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
        ultima_raw = data.get("ultima")
        if not ultima_raw:
            raise ValueError("Campo 'ultima' no encontrado")
        ultima = datetime.fromisoformat(ultima_raw)
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
        ultimo_raw = data.get("ultimo")
        if not ultimo_raw:
            raise ValueError("Campo 'ultimo' no encontrado")
        ultimo = datetime.fromisoformat(ultimo_raw)
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

from datetime import datetime, time

def es_mercado_abierto():
    ahora = datetime.utcnow()
    # Días de semana: 0 = lunes, ..., 4 = viernes
    if ahora.weekday() >= 5:
        return False

    hora_actual = ahora.time()
    apertura = time(14, 30)  # 14:30 UTC
    cierre = time(21, 0)     # 21:00 UTC

    return apertura <= hora_actual <= cierre

def get_news_headlines(ticker, num_headlines=3):
    url = f"https://www.google.com/search?q={ticker}+stock&tbm=nws"
    headers = { 
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        res = requests.get(url, headers=headers, timeout=7)
        if res.status_code != 200:
            print(f"⚠️ Error al buscar noticias de {ticker} - status {res.status_code}")
            return []

        soup = BeautifulSoup(res.text, 'html.parser')
        containers = soup.select('div.dbsr')
        if not containers:
            print(f"⚠️ No se encontraron titulares para {ticker}.")
            return []

        headlines = []
        for el in containers[:num_headlines]:
            title_div = el.select_one('div.JheGif.nDgy9d')
            if title_div:
                headlines.append(title_div.text.strip())

        return headlines

    except Exception as e:
        print(f"⚠️ Error al obtener noticias para {ticker}: {e}")
        return []

def get_news_headlines_bing(ticker, num_headlines=3):
    url = f"https://www.bing.com/news/search?q={ticker}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=7)
        soup = BeautifulSoup(res.text, 'html.parser')
        items = soup.select("a.title")[:num_headlines]
        headlines = [item.text.strip() for item in items if item.text.strip()]
        if not headlines:
            print(f"⚠️ No se encontraron titulares en Bing para {ticker}")
        return headlines
    except Exception as e:
        print(f"⚠️ Error en Bing News ({ticker}): {e}")
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
        print(f"❌ Error en análisis diario de {ticker}: {e}")
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
        return None, None

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, data=payload, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ Error al enviar mensaje Telegram: {res.status_code} - {res.text}")
        else:
            print("✅ Mensaje enviado correctamente a Telegram.")
    except Exception as e:
        print("⚠️ Error enviando mensaje a Telegram:", e)

def enviar_imagen(path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    with open(path, "rb") as photo:
        requests.post(url, files={"photo": photo}, data={"chat_id": CHAT_ID})

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

# Ejecución principal
candidatos = []
CRIPTOS = ["ETH-USD", "SOL-USD"]
OTROS = [s for s in SYMBOLS if s not in CRIPTOS]

if es_mercado_abierto():
    ahora = datetime.utcnow()
    print(f"🕒 Hora actual UTC: {ahora.strftime('%H:%M:%S')} | Día de la semana (0=lunes): {ahora.weekday()}")

    if not es_mercado_abierto():
        print("🚫 El mercado está cerrado. El análisis no se ejecutará.")
    else:
        print("✅ El mercado está abierto. Se iniciará el análisis.")
        print(f"📋 Tickers a evaluar: {len(SYMBOLS)} → {SYMBOLS}")

    for ticker in SYMBOLS:
        print(f"🔍 Evaluando {ticker}...")
        diario = analizar_tecnico_diario(ticker)
        intradia, df = analizar_intradía(ticker)
        print(f"🔎 Procesando {ticker}...")
        
        if diario and intradia:
            entrada = diario["precio"]
            atr = diario["atr"]

            # Validación y corrección si el precio está desfasado (>1%)
            try:
                historial = yf.Ticker(ticker).history(period="1d", interval="1m")
                if historial.empty or "Close" not in historial.columns:
                    print(f"⚠️ {ticker} sin datos recientes para validación de precio.")
                    continue

                precio_actual = historial["Close"].iloc[-1]
                diff_pct = abs(entrada - precio_actual) / precio_actual * 100

                if diff_pct > 1:
                    print(f"⚠️ {ticker}: entrada ajustada de {entrada} → {round(precio_actual, 2)} (desviación de {round(diff_pct, 2)}%)")
                    entrada = round(precio_actual, 2)
            except Exception as e:
                print(f"❌ Error obteniendo precio actual de {ticker}: {e}")
                continue

            # SL y TP fijos adaptados a cuenta fondeada
            sl_pct = 0.002  # 0.2%
            tp_pct = 0.003  # 0.3%

            # Medidas absolutas
            stop_dist = entrada * sl_pct
            tp_dist = entrada * tp_pct

            # ❌ Filtro de volatilidad: si el ATR > 1.5% del precio, descartar
            if atr > entrada * 0.015:
                print(f"⚠️ {ticker} descartado por alta volatilidad: ATR {atr} > 1.5% del precio ({entrada})")
                continue

            # Dirección
            if intradia["direccion"] == "subida":
                stop = entrada - stop_dist
                take_profit = entrada + tp_dist
            else:
                stop = entrada + stop_dist
                take_profit = entrada - tp_dist

            # Porcentajes
            stop_pct = abs(entrada - stop) / entrada * 100
            tp_pct = abs(take_profit - entrada) / entrada * 100

            # ❌ Filtro 2: riesgo real (SL) superior al 2%
            if stop_pct > 2:
                print(f"⚠️ {ticker} descartado por SL elevado: {round(stop_pct, 2)}%")
                continue

            # Validar que la recompensa sea mayor al riesgo
            if tp_pct < stop_pct * 1.33:
                print(f"⚠️ {ticker} descartado: TP ({round(tp_pct, 2)}%) < 1.33x SL ({round(stop_pct, 2)}%)")
                continue

            # Señales adicionales
            if diario["tendencia"] == "Alcista" and intradia["direccion"] == "subida":
                intradia["prob_sube"] += 10
            elif diario["tendencia"] == "Bajista" and intradia["direccion"] == "bajada":
                intradia["prob_baja"] += 10

            if intradia["direccion"] == "subida" and abs(diario["precio"] - diario["soporte"]) / diario["precio"] < 0.01:
                intradia["prob_sube"] += 10
            elif intradia["direccion"] == "bajada" and abs(diario["precio"] - diario["resistencia"]) / diario["precio"] < 0.01:
                intradia["prob_baja"] += 10

            if "Precio sobre SMA50" in diario["señales"] and "SMA50 sobre SMA200" in diario["señales"] and intradia["direccion"] == "subida":
                intradia["prob_sube"] += 10
            elif "Precio sobre SMA50" not in diario["señales"] and "SMA50 sobre SMA200" not in diario["señales"] and intradia["direccion"] == "bajada":
                intradia["prob_baja"] += 10

            # Probabilidad total
            raw_prob_total = max(intradia["prob_sube"], intradia["prob_baja"])
            prob_total = int((raw_prob_total / 40) * 100)
            print(f"📊 {ticker} - Prob sube: {intradia['prob_sube']} | Prob baja: {intradia['prob_baja']} | Total: {prob_total}%")

            candidatos.append({
                "ticker": ticker,
                "diario": diario,
                "intradia": intradia,
                "df": df,
                "prob_total": prob_total,
                "entrada": round(entrada, 2),
                "stop": round(stop, 2),
                "take_profit": round(take_profit, 2),
                "stop_pct": round(stop_pct, 2),
                "tp_pct": round(tp_pct, 2)
            })

            print(f"✅ Análisis completo para {ticker}")
            print(f"🧮 Candidatos acumulados: {len(candidatos)}")

    minutos_alerta = tiempo_desde_ultima_alerta()
    minutos_resumen = tiempo_desde_ultimo_resumen()
    print(f"⏱ Min desde última alerta: {minutos_alerta}")
    print(f"📊 Min desde último resumen: {minutos_resumen}")

    if minutos_alerta >= TIEMPO_RESUMEN_MINUTOS and minutos_resumen >= minutos_alerta and candidatos:
        resumen = "\n".join([
            f"{c['ticker']} | Prob: {c['prob_total']}% | Dirección: {'↑' if c['intradia']['direccion'] == 'subida' else '↓'} | TP: {c['tp_pct']}% | SL: {c['stop_pct']}%"
            for c in sorted(candidatos, key=lambda x: x["prob_total"], reverse=True)
            if c.get("intradia") is not None and c["intradia"].get("direccion") is not None
        ])
        enviar_telegram(f"📊 *Resumen de oportunidades*\n\n{resumen}")
        print("📤 Enviando resumen de oportunidades a Telegram...")
        registrar_resumen()

    if candidatos:
        mejor = max(candidatos, key=lambda r: r["prob_total"])

        if mejor["prob_total"] >= UMBRAL_ALERTA:
            entrada = mejor["entrada"]
            stop = mejor["stop"]
            take_profit = mejor["take_profit"]
            stop_pct = mejor["stop_pct"]
            tp_pct = mejor["tp_pct"]
        
            titulares = get_news_headlines(mejor["ticker"])
            if not titulares:
                print("🔁 Usando Bing como respaldo para titulares.")
                titulares = get_news_headlines_bing(mejor["ticker"])

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

💵 *Entrada sugerida:* {entrada}  
🔻 *Stop Loss:* {stop} ({stop_pct}%)  
🎯 *Take Profit:* {take_profit} ({tp_pct}%)  
📊 *Soporte:* {mejor['diario']['soporte']} | 📈 *Resistencia:* {mejor['diario']['resistencia']}  

📰 *Noticias recientes:*
{resumen_noticia}
"""

            enviar_telegram(mensaje)
            print("🚨 Enviando mejor oportunidad a Telegram...")
            registrar_alerta()
            img_path = generar_grafico(mejor["df"], mejor["ticker"])
            enviar_imagen(img_path)

            enviar_telegram
