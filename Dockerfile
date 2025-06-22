FROM python:3.9.13-slim

#WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      python3-dev \
      libffi-dev \
      libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501 5000

CMD ["streamlit", "run", "Streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
