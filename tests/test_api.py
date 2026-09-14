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

def test_drift_endpoint():
    response = client.post(
        "/drift",
        json={
            "training_data": [
                20, 22, 24, 25, 27,
                30, 31, 32, 35, 36
            ],
            "production_data": [
                20, 22, 24, 25, 27,
                30, 31, 32, 35, 36
            ]
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "psi" in data
    assert "drift_detected" in data
