import os
import json

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def test_best_model_exists():
    """Check that the ONNX model exists before running inference tests."""
    onnx_path = os.path.join(RESULTS_DIR, "best_model.onnx")
    assert os.path.exists(onnx_path), "Serialized ONNX model (best_model.onnx) is missing."


def test_class_labels_exists():
    """Check that the class label JSON file exists."""
    labels_path = os.path.join(RESULTS_DIR, "class_labels.json")
    assert os.path.exists(labels_path), "Class label file (class_labels.json) is missing."


def test_class_labels_is_valid_json():
    """Ensure the class label JSON is valid and not empty."""
    labels_path = os.path.join(RESULTS_DIR, "class_labels.json")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    assert isinstance(labels, list), "Class labels file must contain a list."
    assert len(labels) > 0, "Class label list must not be empty."
    assert all(isinstance(label, str) for label in labels), "Each class label must be a string."
