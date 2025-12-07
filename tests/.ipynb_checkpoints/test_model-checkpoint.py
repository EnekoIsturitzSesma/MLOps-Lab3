import pytest
from PIL import Image
import numpy as np
from mylib.model import (
    predict_class,
    resize_image,
    convert_to_grayscale,
    normalize_image,
)



@pytest.fixture
def sample_image():
    return Image.new("RGB", (100, 100), color="white")


def test_predict_class_returns_valid_label(sample_image):
    classes = ["dog", "cat", "horse", "bear", "pig"]
    pred = predict_class(sample_image)
    assert pred in classes, "The predicted class is not among the established ones."


def test_resize_image(sample_image):
    new_size = (50, 50)
    resized = resize_image(sample_image, new_size)
    assert resized.size == new_size, "The image was not correctly resized."


def test_convert_to_grayscale(sample_image):
    gray = convert_to_grayscale(sample_image)
    assert gray.mode == "L", "The image is not in grayscale."


def test_normalize_image(sample_image):
    normalized = normalize_image(sample_image)

    assert isinstance(normalized, np.ndarray), "Output is not a NumPy array."
    assert normalized.min() >= 0.0, "Existing values below 0."
    assert normalized.max() <= 1.0, "Existing values over 1."
    assert normalized.shape == (100, 100, 3), "Shape does not match."
