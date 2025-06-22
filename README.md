# Bot de Alertas de Trading

Este bot evalúa acciones, ETFs, criptomonedas, índices y materias primas para generar alertas técnicas y enviarlas por Telegram.

## Características

- Análisis técnico diario (RSI, MACD, SMA50, SMA200, ATR)
- Análisis intradía (EMA9/21, RSI, volumen)
- Filtro por riesgo/recompensa y volatilidad
- Análisis de sentimiento de noticias
- Envío automático a Telegram
- Resumen cada 30 minutos si no hay alertas

## Variables de entorno

- `BOT_TOKEN`: Token de tu bot de Telegram
- `CHAT_ID`: ID del chat donde enviar las alertas

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

Puedes programarlo en Render con un cronjob (ej. cada 5 minutos).

