FROM python:3.11-slim

# Ustawienia runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Kopiujemy minimalny zestaw do zbudowania pakietu i odpalenia dashboardu
COPY pyproject.toml README.md /app/
COPY src /app/src

# Instalacja zależności i samego pakietu (pip użyje poetry-core z pyproject)
RUN pip install --upgrade pip && pip install .

# Streamlit
EXPOSE 8501

# Uruchom dashboard
CMD ["streamlit", "run", "src/forest4/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
