FROM python:3.10-slim

WORKDIR /app

# 1) Zainstaluj systemowe biblioteki potrzebne do kompilacji pakietów Python
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libffi-dev \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2) Skopiuj requirements i zainstaluj zależności Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# 3) Skopiuj pozostały kod / modele
COPY . .

EXPOSE 8501 5000

CMD ["streamlit", "run", "Streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
