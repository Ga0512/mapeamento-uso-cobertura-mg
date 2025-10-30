# Base NVIDIA otimizada (menor que a runtime)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CI_MODE="false"

WORKDIR /app

# Instala apenas o essencial em uma única camada
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev gdal-bin && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    pip3 cache purge

# Copia requirements antes do código (melhor cache)
COPY requirements.txt .

# Instala dependências do projeto
RUN pip3 install --no-cache-dir -r requirements.txt

# PyTorch CUDA 11.8 (usa CUDA runtime da base)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# TensorFlow GPU (essa versão já detecta a GPU via CUDA 11.8)
RUN pip3 install --no-cache-dir tensorflow==2.15.0

# Copia o restante do app
COPY . .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
