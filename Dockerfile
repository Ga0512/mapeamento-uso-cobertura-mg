# Use uma imagem NVIDIA com CUDA e cuDNN já incluídos
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CI_MODE="false"

WORKDIR /app

# Corrigido: adicionado '&& \' antes do rm -rf
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip python3-dev \
        gdal-bin && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip3 install --upgrade pip

# Corrigido: havia 'install install' duplicado
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Corrigido: havia 'install install' duplicado também aqui
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Instala TensorFlow GPU
RUN pip3 install tensorflow

# Copia app
COPY . .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
