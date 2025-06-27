FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit mlflow onnx onnxruntime pillow scikit-learn

EXPOSE 8501

CMD ["streamlit", "run", "Streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
