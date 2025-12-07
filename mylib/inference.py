"""
Load an ONNX model with onnxruntime and run inference on an input image.
Outputs predicted class name (requires class_labels.json in same folder).
"""

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort
import json

from torchvision import transforms


def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    t = transform(img)
    # ONNX runtime expects numpy arrays (batch, c, h, w) as float32
    return t.unsqueeze(0).numpy().astype(np.float32)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)


def main(args):
    # load labels
    with open(args.labels, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # load ONNX model
    sess = ort.InferenceSession(args.onnx_model)
    input_name = sess.get_inputs()[0].name

    x = preprocess(args.image)
    preds = sess.run(None, {input_name: x})
    logits = preds[0]
    probs = softmax(logits)
    pred_idx = int(probs.argmax(axis=1)[0])
    pred_label = classes[pred_idx]
    confidence = float(probs[0, pred_idx])
    print(
        f"Predicted: {pred_label} (index={pred_idx}) with confidence {confidence:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-model", default="best_model.onnx")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="class_labels.json")
    inf_args = parser.parse_args()
    main(inf_args)
