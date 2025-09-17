FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --upgrade pip &&     pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]
