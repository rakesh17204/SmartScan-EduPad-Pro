# Dockerfile
FROM python:3.11-slim

# System dependencies for OpenCV and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Helpful envs
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Render (and many PaaS) provide $PORT.
# Use shell form so $PORT is expanded at runtime.
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
