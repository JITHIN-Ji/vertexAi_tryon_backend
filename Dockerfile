FROM python:3.13-slim

# System libraries required by opencv-python-headless and ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class gthread --threads 4 --timeout 120 --keep-alive 75 --max-requests 100 --max-requests-jitter 50 --log-level info app:app