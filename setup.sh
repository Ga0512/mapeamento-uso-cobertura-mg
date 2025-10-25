echo "=== Conteúdo visível no container ==="
ls -R | head -n 50


echo "=== 🧩 Configurando ambiente do CI/CD ==="

# 1) Básico + PPA com GDAL recente
sudo apt-get update -y && sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update -y
sudo apt-get install -y python3-dev python3-numpy

# 2) GDAL do sistema (3.11.x) + headers
sudo apt-get install -y gdal-bin libgdal-dev
gdalinfo --version
gdal-config --version

# 3) Python venv
python3 -m venv venv
source venv/bin/activate
python -V
pip install --upgrade pip wheel setuptools

# 4) Exporta paths (compilação nativa, se necessário)
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
export GDAL_CONFIG=/usr/bin/gdal-config

# 5) Instala **PRIMEIRO** o pacote GDAL do Python, compatível com o sistema
GDAL_VERSION="$(gdal-config --version)"
echo "Instalando GDAL Python ${GDAL_VERSION}…"
# Tenta wheel; se cair para source, já está tudo alinhado com o sistema
pip install "GDAL==${GDAL_VERSION}"

# 6) Agora, as demais deps do projeto (Torch etc.)
#    Obs: mantenha GDAL FORA do requirements.txt para evitar conflito
pip install -r requirements.txt

echo "=== ✅ Ambiente pronto! ==="
python --version
gdalinfo --version

pytest -p no:faulthandler