import io
import json
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image
import pytest

from mylib.model import PetClassifierONNX


def create_test_image():
    img = Image.new("RGB", (224, 224), (100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Image.open(buf)


@pytest.fixture
def fake_labels(tmp_path):
    labels_file = tmp_path / "labels.json"
    labels = ["cat", "dog", "bird"]
    labels_file.write_text(json.dumps(labels), encoding="utf-8")
    return labels_file


@pytest.fixture
def mock_onnx_session():
    """Mock ONNX Runtime to avoid loading a real model."""
    dummy_session = MagicMock()
    dummy_session.get_inputs.return_value = [MagicMock(name="input")]
    dummy_session.run.return_value = [np.array([[0.1, 0.9, 0.0]])]  # dog
    return dummy_session


@patch("mylib.model.ort.InferenceSession")
def test_model_initialization(mock_inference, fake_labels, tmp_path, mock_onnx_session):
    mock_inference.return_value = mock_onnx_session

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-model")

    model = PetClassifierONNX(model_path=str(model_path), labels_path=str(fake_labels))

    assert model.class_labels == ["cat", "dog", "bird"]
    assert model.input_name == "input"


@patch("mylib.model.ort.InferenceSession")
def test_preprocess_output_shape(mock_inference, fake_labels, tmp_path, mock_onnx_session):
    mock_inference.return_value = mock_onnx_session

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-model")
    model = PetClassifierONNX(model_path=str(model_path), labels_path=str(fake_labels))

    img = create_test_image()
    arr = model.preprocess(img)

    assert arr.shape == (1, 3, 224, 224)  # batch, channels, H, W
    assert arr.dtype == np.float32


@patch("mylib.model.ort.InferenceSession")
def test_predict(mock_inference, fake_labels, tmp_path, mock_onnx_session):
    mock_inference.return_value = mock_onnx_session

    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake-model")
    model = PetClassifierONNX(model_path=str(model_path), labels_path=str(fake_labels))

    img = create_test_image()
    pred = model.predict(img)

    assert pred == "dog"  # because model.run returns [[0.1, 0.9, 0.0]]
