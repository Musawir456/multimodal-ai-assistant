FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# Build with audio support: docker build --build-arg EXTRAS="[audio]" -t multimodal-ai-assistant .
ARG EXTRAS=""
RUN pip install ".${EXTRAS}"

EXPOSE 8000
CMD ["multimodal-assistant", "serve", "--host", "0.0.0.0"]
