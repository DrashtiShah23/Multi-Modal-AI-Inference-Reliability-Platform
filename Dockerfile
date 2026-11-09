# Python images: `--target analyzer` or `--target modelserver`.
# The model server image bakes the three model releases in at build time.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[postgres,otlp]" && useradd --uid 10001 --no-create-home bellwether

FROM base AS models
RUN python -m bellwether.models.build --out /models

FROM base AS modelserver
COPY --from=models /models /models
ENV BELLWETHER_MODEL_REPOSITORY=/models/v1 BELLWETHER_MODEL_VERSION=v1 BELLWETHER_PORT=8000
USER 10001
EXPOSE 8000
ENTRYPOINT ["bellwether-modelserver"]

FROM base AS analyzer
COPY --from=models /models /models
ENV BELLWETHER_PORT=8010 BELLWETHER_GRPC_PORT=50051 BELLWETHER_MODEL_REPOSITORY=/models/v1
USER 10001
EXPOSE 8010 50051
ENTRYPOINT ["bellwether-analyzer"]
CMD ["serve"]
