FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN uv pip install --system -r /app/requirements.txt

COPY . /app
