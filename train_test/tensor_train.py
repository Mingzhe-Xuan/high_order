import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import Model
from data import get_tensor_dataloader


def plot_loss(train_losses: list, save_path: str, property_name: str):
    loss_dir = os.path.join(os.path.dirname(save_path), "loss")
    os.makedirs(loss_dir, exist_ok=True)
    path = os.path.join(loss_dir, f"{property_name}_loss.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss",
        color="blue",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_mae(train_mae: list, save_path: str, property_name: str):
    mae_dir = os.path.join(os.path.dirname(save_path), "mae")
    os.makedirs(mae_dir, exist_ok=True)
    path = os.path.join(mae_dir, f"{property_name}_mae.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_mae) + 1),
        train_mae,
        label="Training MAE",
        color="red",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training MAE Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_pointwise_mae(train_pointwise_mae: list, save_path: str, property_name: str):
    pointwise_mae_dir = os.path.join(os.path.dirname(save_path), "pointwise_mae")
    os.makedirs(pointwise_mae_dir, exist_ok=True)
    path = os.path.join(pointwise_mae_dir, f"{property_name}_pointwise_mae.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_pointwise_mae) + 1),
        train_pointwise_mae,
        label="Training Pointwise MAE",
        color="orange",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Pointwise MAE")
    plt.title("Training Pointwise MAE Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_mse(train_mse: list, save_path: str, property_name: str):
    mse_dir = os.path.join(os.path.dirname(save_path), "mse")
    os.makedirs(mse_dir, exist_ok=True)
    path = os.path.join(mse_dir, f"{property_name}_mse.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_mse) + 1),
        train_mse,
        label="Training MSE",
        color="green",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training MSE Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_mean_fnorm_percent_error(train_mean_fnorm_percent_error: list, save_path: str, property_name: str):
    mean_fnorm_percent_error_dir = os.path.join(os.path.dirname(save_path), "mean_fnorm_percent_error")
    os.makedirs(mean_fnorm_percent_error_dir, exist_ok=True)
    path = os.path.join(mean_fnorm_percent_error_dir, f"{property_name}_mean_fnorm_percent_error.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_mean_fnorm_percent_error) + 1),
        train_mean_fnorm_percent_error,
        label="Mean FNORM Percent Error",
        color="purple",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Mean FNORM Percent Error")
    plt.title("Training Mean FNORM Percent Error Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_batchwise_percent_error(train_batchwise_percent_error: list, save_path: str, property_name: str):
    batchwise_percent_error_dir = os.path.join(os.path.dirname(save_path), "batchwise_percent_error")
    os.makedirs(batchwise_percent_error_dir, exist_ok=True)
    path = os.path.join(batchwise_percent_error_dir, f"{property_name}_batchwise_percent_error.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_batchwise_percent_error) + 1),
        train_batchwise_percent_error,
        label="Batchwise Percent Error",
        color="brown",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Batchwise Percent Error")
    plt.title("Training Batchwise Percent Error Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def tensor_train(
    property_name: str,
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    tensor_trainset,
    tensor_valset,
    # cutoff: float,
    # batch_size: int,
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_from: str = None,
    save_interval: int = 5,
    clip_grad_norm: float = 1.0,
    # pin_memory: bool = True,
    # num_workers: int = 0,
    # shuffle: bool = True,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    loss_func: str = "huber",
    limit: int = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(
        embedding_layer,
        invariant_layers,
        middle_mlp,
        equivariant_layers,
        final_mlp,
        readout_layer,
    )
    model = model.to(device)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    train_losses = []
    train_mae = []
    train_pointwise_mae = []
    train_mse = []
    train_mean_fnorm_percent_error = []
    train_batchwise_percent_error = []

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        train_mae = checkpoint["train_mae"]
        train_pointwise_mae = checkpoint.get("train_pointwise_mae", [])
        train_mse = checkpoint.get("train_mse", [])
        train_mean_fnorm_percent_error = checkpoint.get("train_mean_fnorm_percent_error", [])
        train_batchwise_percent_error = checkpoint.get("train_batchwise_percent_error", [])
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    # with open("data/dataloaders/path_name.json") as f:
    #     path_name_dict = json.load(f)
    # assert (
    #     property_name in path_name_dict.keys()
    # ), f"property_name {property_name} is not supported"
    # path = path_name_dict[property_name][0]
    # batches = get_tensor_dataloader(
    #     path,
    #     property_name,
    #     cutoff,
    #     batch_size,
    #     pin_memory,
    #     num_workers,
    #     shuffle,
    # )
    batches = tensor_trainset

    # Loss function
    if loss_func == "huber":
        loss_func = nn.HuberLoss()
    elif loss_func == "mse":
        loss_func = nn.MSELoss()
    elif loss_func == "l1":
        loss_func = nn.L1Loss()
    else:
        raise NotImplementedError(f"loss_func {loss_func} is not implemented")

    # Optimizer
    if optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"optimizer {optimizer} is not implemented")

    # Scheduler
    if scheduler == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise NotImplementedError(f"scheduler {scheduler} is not implemented")

    os.makedirs(checkpoint_path / property_name, exist_ok=True)

    model.train()
    for epoch in range(start_epoch, min(num_epochs, start_epoch + limit)):
        epoch_loss = 0.0
        epoch_mae_sum = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            tensor_property = batch.tensor_property.to(device)

            optimizer.zero_grad()
            pred_tensor_property = model(atom_type, edge_vec, edge_index, batch_index)
            loss = loss_func(pred_tensor_property, tensor_property)
            pointwise_mae = (
                (pred_tensor_property - tensor_property).view(-1).abs().mean()
            )
            mse = (pred_tensor_property - tensor_property).view(-1).pow(2).mean()
            fnorm_error = abs(
                torch.norm(pred_tensor_property, dim=-1)
                - torch.norm(tensor_property, dim=-1)
            )
            fnorm = torch.norm(tensor_property, dim=-1)
            mean_fnorm_percent_error = (fnorm_error / fnorm).mean()
            # # Indicators in GMTNet are bullshit!!!
            # # Calculate the both proper indicators and the fake indicators.
            # Here are the weirdo indicators in GMTNet
            # Batchwise RSSE: fnorm error in GMTNet
            batchwise_rsse = (
                (pred_tensor_property - tensor_property).view(-1).pow(2).sum().sqrt()
            )
            # Batchwise sum of fnorm: fnorm in GMTNet
            batchwise_sum_fnorm = tensor_property.view(-1).pow(2).sum().sqrt()
            # Batchwise percent error: percent error in GMTNet
            batchwise_percent_error = batchwise_rsse / batchwise_sum_fnorm

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            # Accumulate MAE for the epoch
            epoch_mae_sum += pointwise_mae.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}", 
                "mae": f"{pointwise_mae.item():.6f}",
                "mse": f"{mse.item():.6f}",
                "fnorm_err%": f"{mean_fnorm_percent_error.item():.6f}",
                "batch_err%": f"{batchwise_percent_error.item():.6f}"
            })

        avg_loss = epoch_loss / num_batches
        
        # Calculate average MAE for this epoch from accumulated batch MAEs
        avg_mae = epoch_mae_sum / num_batches
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        
        # Store additional metrics
        train_pointwise_mae.append(pointwise_mae.item())
        train_mse.append(mse.item())
        train_mean_fnorm_percent_error.append(mean_fnorm_percent_error.item())
        train_batchwise_percent_error.append(batchwise_percent_error.item())
        
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}, Avg MAE: {avg_mae:.6f}, "
            f"Pointwise MAE: {train_pointwise_mae[-1]:.6f}, MSE: {train_mse[-1]:.6f}, "
            f"Mean FNORM % Error: {train_mean_fnorm_percent_error[-1]:.6f}, "
            f"Batchwise % Error: {train_batchwise_percent_error[-1]:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = checkpoint_path / property_name / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "train_pointwise_mae": train_pointwise_mae,
                    "train_mse": train_mse,
                    "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
                    "train_batchwise_percent_error": train_batchwise_percent_error,
                },
                best_checkpoint_path,
            )
            print(f"Saved best model with loss: {best_loss:.6f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_file = (
                checkpoint_path / property_name / f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "train_mae": train_mae,
                    "train_pointwise_mae": train_pointwise_mae,
                    "train_mse": train_mse,
                    "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
                    "train_batchwise_percent_error": train_batchwise_percent_error,
                },
                checkpoint_file,
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    print(
        f"Training completed. Best loss: {best_loss:.6f}, Best MAE: {train_mae[-1]:.6f}\n"
        f"Final Pointwise MAE: {train_pointwise_mae[-1]:.6f}, Final MSE: {train_mse[-1]:.6f}\n"
        f"Final Mean FNORM % Error: {train_mean_fnorm_percent_error[-1]:.6f}, "
        f"Final Batchwise % Error: {train_batchwise_percent_error[-1]:.6f}"
    )
    plot_loss(train_losses, pic_dir, property_name)
    plot_mae(train_mae, pic_dir, property_name)
    plot_pointwise_mae(train_pointwise_mae, pic_dir, property_name)
    plot_mse(train_mse, pic_dir, property_name)
    plot_mean_fnorm_percent_error(train_mean_fnorm_percent_error, pic_dir, property_name)
    plot_batchwise_percent_error(train_batchwise_percent_error, pic_dir, property_name)
    return model, train_losses, train_mae, train_pointwise_mae, train_mse, train_mean_fnorm_percent_error, train_batchwise_percent_error
