import os
import pytest
import numpy as np
from PIL import Image
import onnxruntime as ort

def test_inference_runs():
    onnx_file = "best_model.onnx"
    class_file = "class_labels.json"
    
    assert os.path.exists(onnx_file)
    assert os.path.exists(class_file)

    # Carga ONNX
    session = ort.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Preparamos una imagen dummy de prueba
    img = np.random.rand(1, 3, 224, 224).astype("float32")  # batch_size=1, RGB 224x224
    outputs = session.run(None, {input_name: img})
    
    # Comprobamos salida
    assert outputs[0].shape[0] == 1  # 1 imagen
    assert outputs[0].shape[1] > 0   # al menos una clase
