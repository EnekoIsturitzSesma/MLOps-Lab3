import json
import numpy as np
from PIL import Image
import onnxruntime as ort


class PetClassifierONNX:
    """
    Wrapper para cargar un modelo ONNX y hacer inferencias.
    """

    def __init__(self, model_path: str, labels_path: str):
        # --- Load labels ---
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

        # Nombre esperado por el modelo
        self.input_name = self.session.get_inputs()[0].name

        print(f"Loaded ONNX model: {model_path}")
        print(f"Loaded labels: {labels_path}")

    # -------------------------------------------------------------
    # Image preprocessing (same as training)
    # -------------------------------------------------------------
    def preprocess(self, img: Image.Image) -> np.ndarray:
        """
        Preprocesa la imagen al formato esperado por MobileNetV2:
        - Convertir a RGB
        - Redimensionar a 224x224
        - Normalizar con coeficientes de ImageNet
        - Añadir dimensión batch
        """
        img = img.convert("RGB")
        img = img.resize((224, 224))

        img = np.array(img).astype("float32") / 255.0

        # Normalización ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)  # añadir batch

        return img.astype("float32")

    # -------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------
    def predict(self, img: Image.Image) -> str:
        """
        Devuelve el label predicho.
        """

        x = self.preprocess(img)

        inputs = {self.input_name: x}

        outputs = self.session.run(None, inputs)
        logits = outputs[0][0]  # shape: [num_classes]

        class_idx = int(np.argmax(logits))
        class_label = self.labels[class_idx]

        return class_label
