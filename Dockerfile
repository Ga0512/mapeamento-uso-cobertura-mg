# Use uma imagem NVIDIA com CUDA e cuDNN já incluídos
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Ambiente básico
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Instala Python3.10 (já provavelmente incluído ou pode instalar via apt),
# mas se quiser exatamente 3.10 talvez precise de algum ppa ou escolher outra base
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN python3.10 -m pip install --upgrade pip

# Copia requirements e instala
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Instala PyTorch + torchvision + torchaudio para CUDA 11.8
RUN python3.10 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Instala TensorFlow GPU
RUN python3.10 -m pip install tensorflow

# Copia app
COPY . .

# Expondo porta
EXPOSE 8000

# Comando de entrada
CMD ["python3.10", "-m", "uvicorn", "pipeline.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug", "--reload"]
