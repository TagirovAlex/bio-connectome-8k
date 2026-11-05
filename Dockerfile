# Build stage: создаёт образ, в котором уже стоит torch(cu128)+код+8K-графы.
# На сервере:  docker build -t run8k:latest .
# Запуск с GPU: docker run --rm --gpus all -v $HOST/results:/opt/run8k/results run8k:latest
FROM python:3.11-slim

# Системные зависимости: build-essential для колес с компиляцией, libgl для matplotlib/env
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

WORKDIR /opt/run8k

# torch ставится отдельно с cu128-индекса (Blackwell/Ada и новее), затем остальное
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url ${TORCH_INDEX_URL} && \
    pip install -r requirements.txt

# Код, конфиги, 8K-графы, метрики 2K (нужны для итоговых фигур 'all')
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY data/processed ./data/processed
COPY results/metrics ./results/metrics

# Точка входа: обучает 8K, анализирует, строит фигуры, пишет маркер готовности
ENTRYPOINT ["bash", "scripts/run_8k_inside.sh"]