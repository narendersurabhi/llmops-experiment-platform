FROM ghcr.io/mlflow/mlflow:v2.16.2
COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

RUN uv pip install --system boto3 psycopg2-binary
