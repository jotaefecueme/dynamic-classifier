FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc6-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "app:app", "-k", "uvicorn.workers.UvicornWorker", \
    "-b", "0.0.0.0:8080", \
    "--workers", "1", "--threads", "1", \
    "--timeout", "30", "--graceful-timeout", "15", \
    "--keep-alive", "60"]
