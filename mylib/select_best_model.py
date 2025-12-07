"""
Query MLflow experiment to pick the best run by final_val_accuracy, load the logged model,
serialize it to ONNX and save locally for deployment.
"""

import argparse
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient


def main(args):
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(args.experiment_name)
    if exp is None:
        raise ValueError(f"Experiment {args.experiment_name} not found")

    exp_id = exp.experiment_id
    # Search runs, order by final_val_accuracy desc
    runs = mlflow.search_runs(
        [exp_id], order_by=["metrics.final_val_accuracy DESC"], max_results=10
    )
    if runs.shape[0] == 0:
        raise RuntimeError("No runs found in experiment")

    best_run = runs.iloc[0]
    best_run_id = best_run.run_id
    best_val = best_run.get("metrics.final_val_accuracy", None)
    print(f"Best run id: {best_run_id}, final_val_accuracy={best_val}")

    # Load model artifact (logged with artifact_path="model")
    model_uri = f"runs:/{best_run_id}/model"
    print("Loading model from:", model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    # Export to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_out_path = args.output or "best_model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("ONNX model exported to", onnx_out_path)

    # Also download the class labels artifact (if exists) to the same folder
    try:
        artifacts = client.list_artifacts(best_run_id, path="")

        # recursive find simple approach
        labels_file = None
        for it in artifacts:
            if it.is_dir:
                for sub in client.list_artifacts(best_run_id, path=it.path):
                    if sub.path.endswith("class_labels.json"):
                        labels_file = sub.path
                        client.download_artifacts(best_run_id, sub.path, dst_path=".")
            else:
                if it.path.endswith("class_labels.json"):
                    labels_file = it.path
                    client.download_artifacts(best_run_id, it.path, dst_path=".")
        if labels_file:
            print("Downloaded class labels artifact.")
        else:
            print("No class_labels.json artifact found in run.")
    except FileNotFoundError as e:
        print("Artifact file not found:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="MLOps-Lab3-experiment")
    parser.add_argument("--output", type=str, default=None)
    sbm_args = parser.parse_args()
    main(sbm_args)
