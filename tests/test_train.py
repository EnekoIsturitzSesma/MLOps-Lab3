import os
import subprocess
import json
import pytest

def test_train_runs_and_artifacts_exist():
    # Ejecuta el script de entrenamiento
    result = subprocess.run(
        ["uv", "run", "python", "mylib/train.py", "--epochs", "1"],  # 1 epoch para test rápido
        capture_output=True, text=True
    )
    print(result.stdout)
    assert result.returncode == 0

    # Comprobamos que algunos artifacts de MLflow se generaron
    artifacts = ["class_labels_16_0.001.json", "class_labels_32_0.001.json"]
    for artifact in artifacts:
        assert os.path.exists(artifact) or os.path.exists(os.path.join("mlruns", artifact))
