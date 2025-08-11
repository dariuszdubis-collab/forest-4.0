# ---------- FOREST 4.0 Makefile ----------
# Uwaga: używamy .RECIPEPREFIX, więc każda linia komend zaczyna się znakiem '>'
# (nie musisz pilnować tabulatorów).
.RECIPEPREFIX := >

# Zmienne (możesz zmienić na własne)
IMAGE := forest4:latest

.PHONY: help setup fmt lint test build-image run-dash run-cli clean

help:
> @echo "Targets:"
> @echo "  setup        - poetry install + pre-commit install"
> @echo "  fmt          - format (ruff)"
> @echo "  lint         - lint (ruff)"
> @echo "  test         - pytest -q"
> @echo "  build-image  - docker build -t $(IMAGE) ."
> @echo "  run-dash     - docker run --rm -p 8501:8501 $(IMAGE)"
> @echo "  run-cli      - uruchom backtest w kontenerze (prices.csv -> equity.csv)"
> @echo "  clean        - sprząta cache/artefakty"

setup:
> poetry install --no-interaction
> poetry run pre-commit install

fmt:
> poetry run ruff format .

lint:
> poetry run ruff check .

test:
> poetry run pytest -q

build-image:
> docker build -t $(IMAGE) .

run-dash:
> docker run --rm -p 8501:8501 $(IMAGE)

# Przykład: połóż prices.csv w katalogu repo i uruchom:
#   make run-cli
# Powstanie equity.csv obok pliku wejściowego.
run-cli:
> docker run --rm -v "$(PWD)":/work $(IMAGE) \
>   sh -lc "forest4-backtest --input /work/prices.csv --fast 12 --slow 26 --out /work/equity.csv"

clean:
> rm -rf .ruff_cache .pytest_cache .mypy_cache dist build *.egg-info
