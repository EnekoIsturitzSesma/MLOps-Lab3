import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms


class PetClassifierONNX:
    def __init__(self, model_path="best_model.onnx", labels_path="class_labels.json"):
        # Load class labels
        with open(labels_path, "r", encoding="utf-8") as f:
            self.class_labels = json.load(f)

        # ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        # Preprocessing (same as training)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def preprocess(self, image: Image.Image):
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # batch dimension
        return tensor.numpy()

    def predict(self, image: Image.Image):
        inp = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: inp})
        logits = outputs[0][0]  # first batch element
        pred_idx = int(np.argmax(logits))
        return self.class_labels[pred_idx]


model = PetClassifierONNX("results/best_model.onnx", "results/class_labels.json")


def predict_class(image):
    return model.predict(image)


def resize_image(image, size):
    """
    Resizes an image to the specified dimensions.

    Parameters:
        image (PIL.Image): Input image.
        size (tuple): Target size as (width, height).

    Returns:
        PIL.Image: The resized image.
    """
    return image.resize(size)


def convert_to_grayscale(image):
    """
    Converts an image to grayscale.

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: The grayscale image in mode 'L'.
    """
    return image.convert("L")


def normalize_image(image):
    """
    Normalizes image pixel values to the range [0, 1].

    Parameters:
        image (PIL.Image): Input image.

    Returns:
        numpy.ndarray: The normalized image as a NumPy array.
    """
    image_array = np.array(image) / 255.0
    return image_array
