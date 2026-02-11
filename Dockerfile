# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps (runtime + tesseract). Keep lean with --no-install-recommends
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip first, then install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Example entrypoint (adjust to your app)
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
