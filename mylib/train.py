"""
Train a small transfer-learning model on Oxford-IIIT Pet, track experiments with MLflow,
and register each trained model under the same registered model name.
"""

import json
import random
from datetime import datetime
import itertools

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets import OxfordIIITPet

import mlflow
import mlflow.pytorch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return running_loss / total, correct / total


def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    return running_loss / total, correct / total


def run_experiment(config, args, dataset, n_classes):
    set_seed(args.seed)
    device = torch.device("cpu")

    # Split dataset
    n = len(dataset)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=2
    )

    # Prepare model
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in backbone.features.parameters():
        param.requires_grad = False

    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2), nn.Linear(in_features, n_classes)
    )
    model = backbone.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"]
    )

    # MLflow logging
    run_name = f"{args.model_name}_bs{config['batch_size']}_lr{config['lr']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "seed": args.seed,
                "model_name": args.model_name,
                "dataset": "OxfordIIITPet(trainval)",
                "n_classes": n_classes,
                "batch_size": config["batch_size"],
                "lr": config["lr"],
                "optimizer": "Adam",
                "epochs": args.epochs,
                "val_ratio": args.val_ratio,
            }
        )

        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"[{run_name}] Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}"
            )

        # final metrics
        mlflow.log_metric("final_train_accuracy", train_accs[-1])
        mlflow.log_metric("final_val_accuracy", val_accs[-1])
        mlflow.log_metric("final_train_loss", train_losses[-1])
        mlflow.log_metric("final_val_loss", val_losses[-1])

        # log loss curves
        plt.figure()
        plt.plot(range(1, args.epochs + 1), train_losses, label="train_loss")
        plt.plot(range(1, args.epochs + 1), val_losses, label="val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        loss_plot = f"loss_curve_{config['batch_size']}_{config['lr']}.png"
        plt.savefig(loss_plot)
        mlflow.log_artifact(loss_plot)

        # log class labels
        labels_json = f"class_labels_{config['batch_size']}_{config['lr']}.json"
        with open(labels_json, "w", encoding="utf-8") as f:
            json.dump(dataset.classes, f)
        mlflow.log_artifact(labels_json)

        # log and register model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=args.registered_model_name,
        )
        print(
            f"Model logged and registered as: {args.registered_model_name} (run_id={run_id})"
        )


def main(args):
    mlflow.set_experiment(args.experiment_name)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = OxfordIIITPet(
        root=args.data_dir,
        split="trainval",
        target_types="category",
        transform=transform,
        download=True,
    )
    n_classes = len(dataset.classes)
    print("Detected classes:", n_classes)

    # define experiments
    batch_sizes = [16, 32]
    learning_rates = [1e-3, 5e-4]
    configs = [
        {"batch_size": bs, "lr": lr}
        for bs, lr in itertools.product(batch_sizes, learning_rates)
    ]

    for config in configs:
        run_experiment(config, args, dataset, n_classes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="root to download dataset"
    )
    parser.add_argument("--experiment-name", type=str, default="MLOps-Lab3-experiment")
    parser.add_argument("--registered-model-name", type=str, default="MLOps-Lab3-Model")
    parser.add_argument("--model-name", type=str, default="mobilenet_v2")
    parser.add_argument(
        "--epochs", type=int, default=5, help="train 5 epochs for all experiments"
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    train_args = parser.parse_args()
    main(train_args)
