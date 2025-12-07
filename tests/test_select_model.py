import os
import subprocess
import pytest

def test_serialize_runs_and_onnx_exists():
    # Ejecuta el script de serialización
    result = subprocess.run(
        ["uv", "run", "python", "mylib/select_best_model.py"],
        capture_output=True, text=True
    )
    print(result.stdout)
    assert result.returncode == 0

    # Comprobar que el modelo ONNX se ha creado
    onnx_file = "best_model.onnx"
    assert os.path.exists(onnx_file)
    
    # Comprobar que el JSON con clases existe
    class_file = "class_labels.json"
    assert os.path.exists(class_file)
