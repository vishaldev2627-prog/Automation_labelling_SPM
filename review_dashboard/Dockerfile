FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8600

CMD ["gunicorn", "--bind", "0.0.0.0:8600", "--workers", "2", "--threads", "4", "app:app"]
