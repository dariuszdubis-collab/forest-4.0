# FOREST 4.0

[![CI](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/ci.yml)
[![Docker](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/docker-build.yml/badge.svg?branch=main)](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/docker-build.yml)
[![CodeQL](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/dariuszdubis-collab/forest-4.0/actions/workflows/codeql.yml)

# FOREST 4.0

Modularny framework do backtestów strategii (przykład EMA cross), z dashboardem (Streamlit) i
papierowym brokerem do trybu live.

## Szybki start (Poetry)

```bash
pipx install poetry

poetry install
poetry run pytest -q

poetry run forest4-dashboard

## Docker (dashboard)

Aby uruchomić dashboard bez instalacji Pythona:

```bash
docker build -t forest4:latest .
feat/demo-ohlc
docker run --rm -p 8501:8501 forest4:latest

## Docker (GHCR) – Quickstart

Najprościej przez Docker Compose:

```bash
docker compose up -d
# -> http://localhost:8501

### Demo danych (syntetyczny OHLC)
```bash
poetry run python -m forest4.examples.synthetic --out demo.csv --periods 365 --freq D
# a następnie w Dashboardzie wgraj demo.csv (zakładka Back-test)
=======
docker run --rm -p 8501:8501 forest4:latest

