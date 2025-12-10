import json
import os
import random
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Reproducibilidad
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Training of a single experiment
# -----------------------------
def run_experiment(model_name, batch_size, lr, num_epochs, experiment_name):
    set_seed(42)

    mlflow.set_experiment(experiment_name)
    run_name = f"{model_name}_bs{batch_size}_lr{lr}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": model_name,
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": num_epochs,
                "seed": 42,
                "dataset": "Oxford-IIIT Pet",
            }
        )

        # -----------------------------
        # Dataset & transforms
        # -----------------------------
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        dataset = datasets.OxfordIIITPet(
            root="./data",
            download=True,
            transform=transform,
            target_types="category",
        )

        num_classes = len(dataset.classes)

        # Save class labels for inference later
        os.makedirs("results", exist_ok=True)
        class_path = "results/class_labels.json"
        with open(class_path, "w", encoding="utf-8") as f:
            json.dump(dataset.classes, f)
        mlflow.log_artifact(class_path)

        # Split dataset
        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # -----------------------------
        # Load pretrained model
        # -----------------------------
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

        device = "cpu"
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses = []
        val_losses = []
        val_accuracies = []

        # -----------------------------
        # Training Loop
        # -----------------------------
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            accuracy = correct / total

            val_losses.append(avg_val_loss)
            val_accuracies.append(accuracy)

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", accuracy, step=epoch)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Val Acc: {accuracy:.4f}"
            )

        final_acc = val_accuracies[-1]
        mlflow.log_metric("final_val_accuracy", final_acc)

        # -----------------------------
        # Save curve plot
        # -----------------------------
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Loss Curve")

        plot_path = f"results/loss_curve_bs{batch_size}_lr{lr}.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # -----------------------------
        # Log PyTorch model
        # -----------------------------
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="pet_classifier",
        )

        return final_acc


# -----------------------------
# Grid search + choose best model
# -----------------------------
def train_all():
    experiment_name = "pet_experiments"

    # 4 experiments
    configs = [
        {"batch_size": 8, "lr": 1e-3},
        {"batch_size": 16, "lr": 1e-3},
        {"batch_size": 8, "lr": 5e-4},
        {"batch_size": 16, "lr": 5e-4},
    ]

    best_acc = -1

    for cfg in configs:
        print(f"\n=== Training experiment: bs={cfg['batch_size']} lr={cfg['lr']} ===")

        acc = run_experiment(
            model_name="mobilenetV2",
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            num_epochs=3,
            experiment_name=experiment_name,
        )

        if acc > best_acc:
            best_acc = acc

    print(f"\nBest model accuracy = {best_acc:.4f}")


if __name__ == "__main__":
    train_all()
