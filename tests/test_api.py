from fastapi.testclient import TestClient
from pipeline.api.main import app

client = TestClient(app)

def test_predict_route():
    with open("./dataset/Augmented/Images/Amostra_piloto_6_2_aug_cls4_002.tif", "rb") as img:
        response = client.post(
            "/predict/",
            files={"images": ("sample.jpg", img, "image/jpeg")},
            data={"model_name": "segformer"}
        )
    assert response.status_code == 200
    assert "masks" in response.json()
