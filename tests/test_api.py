from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction():
    response = client.post(
        "/predict",
        json={"value": 5}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input"] == 5
    assert abs(data["prediction"] - 10) < 0.001
