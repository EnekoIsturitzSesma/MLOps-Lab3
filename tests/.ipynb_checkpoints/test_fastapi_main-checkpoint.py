import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
from api.fastapi_main import app


client = TestClient(app)


def create_test_image():
    """Create an in-memory RGB image."""
    img = Image.new("RGB", (100, 100), color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    return img_bytes


def test_predict():
    img_bytes = create_test_image()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_resize():
    img_bytes = create_test_image()
    response = client.post(
        "/resize",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"width": "50", "height": "60"},
    )
    assert response.status_code == 200
    assert response.json() == {"width": 50, "height": 60}


def test_grayscale():
    img_bytes = create_test_image()
    response = client.post(
        "/grayscale",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "L"


def test_normalize():
    img_bytes = create_test_image()
    response = client.post(
        "/normalize",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    result = response.json()
    assert result["min"] >= 0
    assert result["max"] <= 1
