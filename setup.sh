#!/usr/bin/env bash
set -e

echo "=== 🧩 Configurando ambiente do CI/CD ==="

# Atualiza pacotes e instala dependências básicas
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y \
    software-properties-common \
    git curl wget unzip build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev

echo "=== 🌍 Instalando GDAL ==="
# Instala o GDAL e bindings Python
sudo apt-get install -y gdal-bin libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

# Exibe versão do GDAL para checagem
gdalinfo --version || echo "GDAL não encontrado!"

echo "=== 🐍 Instalando dependências Python ==="
python3 -m venv venv
source venv/bin/activate

# Garante pip atualizado
pip install --upgrade pip wheel setuptools

# Instala dependências do projeto
pip install -r requirements.txt

# Instala GDAL com a mesma versão do sistema
GDAL_VERSION=$(gdal-config --version)
pip install "GDAL==${GDAL_VERSION}"

echo "=== ✅ Ambiente pronto! ==="
python --version
gdalinfo --version
