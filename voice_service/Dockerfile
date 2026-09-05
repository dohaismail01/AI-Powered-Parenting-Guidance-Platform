FROM python:3.12-slim

# ffmpeg is required by faster-whisper/ctranslate2 to decode most audio formats
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py transcriber.py stt_service.py ./

# Model downloads on first run and gets cached in this layer's volume;
# mount a volume at /root/.cache/huggingface in production so you don't
# re-download on every container restart.
EXPOSE 8001

CMD ["uvicorn", "stt_service:app", "--host", "0.0.0.0", "--port", "8001"]
