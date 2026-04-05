ARG BASE_IMAGE=llmops/base:dev
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.9.22 /uv /uvx /bin/

COPY docker/trainer/requirements.txt /tmp/trainer-requirements.txt
RUN uv pip install --system -r /tmp/trainer-requirements.txt

ENTRYPOINT ["python", "-m", "components.trainer.run_pretraining"]
