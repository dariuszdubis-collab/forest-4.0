# FOREST 4.0

Modularny framework do backtestów strategii (przykład EMA cross), z dashboardem (Streamlit) i
papierowym brokerem do trybu live.

## Szybki start (Poetry)

```bash
# 1) Zainstaluj Poetry (polecamy pipx)
pipx install poetry

# 2) Zainicjalizuj środowisko i zainstaluj zależności
poetry install

# 3) Uruchom testy
poetry run pytest -q

# 4) Uruchom dashboard
poetry run forest-dashboard
```

## Struktura
- `src/forest/config.py` – Pydantic 2.0: konfiguracja backtestu (risk/strategy/timeframe).
- `src/forest/utils/` – normalizacja timeframe, walidacja danych, logger.
- `src/forest/core/indicators.py` – EMA i ATR (pandas).
- `src/forest/backtest/` – engine, risk, tradebook, trace, grid.
- `src/forest/dashboard/app.py` – Streamlit UI: backtest, grid, heatmap.
- `src/forest/live/router.py` – PaperBroker i interfejs OrderRouter.