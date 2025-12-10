import json
import numpy as np
from PIL import Image
import onnxruntime as ort


class PetClassifierONNX:
    """
    Wrapper to load an ONNX model and perform inference.
    """

    def __init__(self, model_path: str, labels_path: str):
        with open(labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

        # --- ONNX Runtime session ---
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

        print(f"Loaded ONNX model: {model_path}")
        print(f"Loaded labels: {labels_path}")

    def preprocess(self, img: Image.Image) -> np.ndarray:
        """
        Preprocesses the image to the format expected by MobileNetV2:
        - Convert to RGB
        - Resize to 224x224
        - Normalize with ImageNet coefficients
        - Add batch dimension
        """
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img).astype("float32") / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        return img.astype("float32")

    def predict(self, img: Image.Image) -> str:
        """
        Returns the predicted label.
        """

        x = self.preprocess(img)

        inputs = {self.input_name: x}

        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]

        class_idx = int(np.argmax(logits))
        class_label = self.labels[class_idx]

        return class_label
