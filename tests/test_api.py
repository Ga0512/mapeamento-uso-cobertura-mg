import io
from fastapi.testclient import TestClient
from pipeline.api.main import app

client = TestClient(app)

def test_predict_route():
    """
    Testa se o endpoint /predict/ responde corretamente.
    Usa uma imagem fake gerada em memória.
    O teste não depende de arquivos nem de modelo real.
    """

    # Gera uma imagem fake em memória (sem precisar de .tif)
    fake_image = io.BytesIO(b"fake_image_data")

    # Faz o POST simulando o upload
    response = client.post(
        "/predict/",
        files={"images": ("fake.jpg", fake_image, "image/jpeg")},
        data={"model_name": "segformer"},
    )

    # Verifica se a API respondeu corretamente
    assert response.status_code == 200, f"Status inválido: {response.status_code}"
    data = response.json()

    # Verifica estrutura da resposta
    assert isinstance(data, dict), "A resposta deve ser um dicionário JSON"
    assert "masks" in data, "A resposta deve conter o campo 'masks'"
    assert isinstance(data["masks"], list), "O campo 'masks' deve ser uma lista"
