import io
from fastapi.testclient import TestClient
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_invalid_image():
    bad_bytes = io.BytesIO(b"not-an-image")
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", bad_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "error" in response.json()


def test_resize():
    img_bytes = create_test_image()
    response = client.post(
        "/resize",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
        data={"width": 50, "height": 60},
    )
    assert response.status_code == 200
    assert response.json() == {"width": 50, "height": 60}


def test_resize_missing_params():
    img_bytes = create_test_image()
    response = client.post(
        "/resize",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
        data={},
    )
    assert response.status_code == 422


def test_resize_invalid_dimensions():
    img_bytes = create_test_image()
    response = client.post(
        "/resize",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
        data={"width": -10, "height": 0},
    )
    assert response.status_code == 200
    assert "error" in response.json()


def test_grayscale():
    img_bytes = create_test_image()
    response = client.post(
        "/grayscale",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "L"


def test_grayscale_invalid_image():
    bad_bytes = io.BytesIO(b"invalid")
    response = client.post(
        "/grayscale",
        files={"file": ("img.jpg", bad_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "error" in response.json()


def test_normalize():
    img_bytes = create_test_image()
    response = client.post(
        "/normalize",
        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
    )
    result = response.json()
    assert result["min"] >= 0
    assert result["max"] <= 1


def test_normalize_invalid_image():
    bad_bytes = io.BytesIO(b"invalid")
    response = client.post(
        "/normalize",
        files={"file": ("img.jpg", bad_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    assert "error" in response.json()
