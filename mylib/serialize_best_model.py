import json
import os
import torch
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch


MODEL_NAME = "pet_classifier"  # nombre con el que registraste el modelo
OUTPUT_DIR = "serialized"  # carpeta donde guardamos el onnx + labels
LABELS_FILENAME = "class_labels.json"  # nombre del archivo con labels
ONNX_FILENAME = "model.onnx"  # nombre del modelo serializado


def get_best_model_version(client):
    """
    Selecciona el mejor modelo registrado según la métrica 'final_val_accuracy'.
    """
    print("Searching registered models...")

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError("No registered models found.")

    best_version = None
    best_acc = -1

    for mv in versions:
        run_id = mv.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics

        val_acc = metrics.get("final_val_accuracy", None)
        if val_acc is None:
            continue

        print(f"Version {mv.version} | Run {run_id} | Val Acc = {val_acc}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_version = mv

    if best_version is None:
        raise ValueError("No valid models with 'final_val_accuracy' found.")

    print(
        f"\nBest model: Version {best_version.version} | "
        f"Run {best_version.run_id} | Acc = {best_acc:.4f}"
    )

    return best_version


def serialize_to_onnx(model, example_input, output_path):
    """
    Serializa el modelo PyTorch a ONNX.
    """
    torch.onnx.export(
        model,
        example_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    print(f"ONNX model saved → {output_path}")


def main():
    client = MlflowClient()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------
    # 1. Seleccionar mejor versión del modelo registrado
    # ---------------------------------------------------
    best_version = get_best_model_version(client)
    run_id = best_version.run_id

    # ---------------------------------------------------
    # 2. Descargar labels.json del run
    # ---------------------------------------------------
    print("\nDownloading class labels...")
    local_path = client.download_artifacts(run_id, LABELS_FILENAME)
    labels_output_path = os.path.join(OUTPUT_DIR, LABELS_FILENAME)

    with open(local_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    with open(labels_output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"Labels saved → {labels_output_path}")

    # ---------------------------------------------------
    # 3. Cargar modelo PyTorch desde MLflow
    # ---------------------------------------------------
    model_uri = f"runs:/{run_id}/model"
    print(f"\nLoading model from {model_uri} ...")

    model = mlflow.pytorch.load_model(model_uri)
    model.to("cpu")
    model.eval()

    # ---------------------------------------------------
    # 4. Crear input dummy para exportar a ONNX
    # ---------------------------------------------------
    example_input = torch.randn(1, 3, 224, 224, device="cpu")

    # ---------------------------------------------------
    # 5. Exportar modelo a ONNX
    # ---------------------------------------------------
    onnx_output_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)
    serialize_to_onnx(model, example_input, onnx_output_path)

    print("\nSerialization completed successfully.")


if __name__ == "__main__":
    main()
