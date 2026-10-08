# Nakış Atölyesi — profesyonel motor (Ink/Stitch v3.3.0, GPL-3.0)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    INKSTITCH_DIR=/opt/inkstitch INKSTITCH_PYTHON=python
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG INKSTITCH_VERSION=v3.3.0
RUN GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch ${INKSTITCH_VERSION} \
      https://github.com/inkstitch/inkstitch.git ${INKSTITCH_DIR} \
    && rm -rf ${INKSTITCH_DIR}/.git

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps inkex==1.4.1 \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
