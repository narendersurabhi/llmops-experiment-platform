FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf

WORKDIR /app
COPY docker/qwen-service/requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

COPY components /app/components

CMD ["uvicorn", "components.inference.qwen_service:app", "--host", "0.0.0.0", "--port", "8000"]
