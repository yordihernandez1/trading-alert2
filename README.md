# Sistema de Alertas de Trading

Este repositorio contiene un script que analiza acciones y criptomonedas, detecta señales técnicas, evalúa noticias mediante análisis de sentimiento y envía alertas a Telegram.

## 📦 Requisitos

- Python 3.8 o superior
- Tener el archivo `vader_lexicon.txt` en la raíz del proyecto (es el diccionario de VADER)
- Crear un bot de Telegram y obtener el `BOT_TOKEN`
- Obtener tu `CHAT_ID` (puede ser personal o de grupo)

## 📁 Estructura del proyecto

```
trading_alert_system/
├── main.py
├── requirements.txt
├── .env.example
└── vader_lexicon.txt
```

## ⚙️ Configuración

1. Copia `.env.example` a `.env` y rellena con tus credenciales.
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecuta el script:

```bash
python main.py
```

## 🕑 Uso en Render / Railway

- Configura tus variables `BOT_TOKEN` y `CHAT_ID` en el panel de entorno.
- Usa un cronjob entre las 14:00 y 21:00 UTC para llamar a `python main.py`.

## 📤 Salida esperada

El bot de Telegram enviará un mensaje con:

- El activo con mayor potencial
- Datos técnicos (RSI, MACD, SMA, ATR...)
- Análisis de sentimiento de noticias
- Pregunta final para revisión manual o ChatGPT

---

Este sistema es autónomo y no depende de conexiones externas para descargar modelos de NLP.