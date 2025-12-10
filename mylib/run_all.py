import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_all  # pylint: disable=import-error
from serialize_best_model import main as serialize_model  # pylint: disable=import-error
from inference import PetClassifierONNX  # pylint: disable=import-error
from PIL import Image


def main():
    print("\n=== TRAINING EXPERIMENTS ===")
    train_all()

    print("\n=== SERIALIZING BEST MODEL ===")
    serialize_model()

    print("\n=== TESTING ONNX INFERENCE ===")
    onnx_model_path = "results/best_model.onnx"
    labels_path = "results/class_labels.json"

    classifier = PetClassifierONNX(onnx_model_path, labels_path)

    img = Image.open("image.png")
    pred = classifier.predict(img)

    print(f"Predicted class: {pred}")


if __name__ == "__main__":
    main()
