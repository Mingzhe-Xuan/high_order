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

from src.model import Model
from data import get_scalar_dataloader


def plot_loss(train_losses: list, save_path: str, property_name: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    path = os.path.join(os.path.dirname(save_path), f"{property_name}_loss.png")

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    path = os.path.join(os.path.dirname(save_path), f"{property_name}_mae.png")

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


def plot_val_loss(train_losses: list, val_losses: list, save_path: str, property_name: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    path = os.path.join(os.path.dirname(save_path), f"{property_name}_val_loss.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Training Loss",
        color="blue",
    )
    plt.plot(
        range(1, len(val_losses) + 1),
        val_losses,
        label="Validation Loss",
        color="orange",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_val_mae(train_mae: list, val_mae_scores: list, save_path: str, property_name: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    path = os.path.join(os.path.dirname(save_path), f"{property_name}_val_mae.png")

    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(train_mae) + 1),
        train_mae,
        label="Training MAE",
        color="red",
    )
    plt.plot(
        range(1, len(val_mae_scores) + 1),
        val_mae_scores,
        label="Validation MAE",
        color="green",
    )
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Training and Validation MAE Over Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def validate_model(model, val_loader, device, loss_fn):
    """
    Validate the model on the validation dataset
    """
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch in val_loader:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            scalar_property = batch.scalar_property.to(device)

            pred_scalar_property = model(atom_type, edge_vec, edge_index, batch_index)
            mae = (pred_scalar_property - scalar_property).abs().mean()
            loss = loss_fn(pred_scalar_property, scalar_property)
            
            val_loss += loss.item()
            val_mae += mae.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    avg_val_mae = val_mae / num_batches
    
    return avg_val_loss, avg_val_mae


def scalar_train(
    property_name: str,
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    scalar_trainset,
    scalar_valset,
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
    """
    Train a scalar property prediction model with validation.

    Args:
        property_name: Name of the property to predict
        embedding_layer: Embedding layer of the model
        invariant_layers: Invariant layers of the model
        middle_mlp: Middle MLP layers
        equivariant_layers: Equivariant layers of the model
        final_mlp: Final MLP layers
        readout_layer: Readout layer
        scalar_trainset: Training dataset
        scalar_valset: Validation dataset
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints
        pic_dir: Directory to save plots
        start_epoch: Starting epoch (for resuming)
        resume_from: Path to resume from checkpoint
        save_interval: Interval to save checkpoints
        clip_grad_norm: Gradient clipping norm
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        optimizer: Type of optimizer ('adamw', 'adam', 'sgd')
        scheduler: Type of scheduler ('cosine_annealing', 'step')
        loss_func: Type of loss function ('huber', 'mse', 'l1')
        limit: Limit number of epochs (optional)

    Returns:
        tuple: (model, train_losses, train_mae, val_losses, val_mae_scores)
    """
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

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        train_mae = checkpoint["train_mae"]
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    # supported_scalar_properties = [
    #     "formation_energy",
    #     "opt_band_gap",
    #     "total_energy",
    #     "ehull",
    #     "mbj_bandgap",
    #     "bandgap",
    #     "e_form",
    #     "bulk_modulus",
    #     "shear_modulus",
    # ]

    # with open("data/dataloaders/name_path.json") as f:
    #     name_path_dict = json.load(f)
    # assert (
    #     property_name in name_path_dict.keys()
    # ), f"property_name {property_name} is not supported"
    # path = name_path_dict[property_name][0]
    # batches = get_scalar_dataloader(
    #     path,
    #     property_name,
    #     cutoff,
    #     batch_size,
    #     pin_memory,
    #     num_workers,
    #     shuffle,
    # )
    batches = scalar_trainset

    # Loss function
    if loss_func == "huber":
        loss_fn = nn.HuberLoss()
    elif loss_func == "mse":
        loss_fn = nn.MSELoss()
    elif loss_func == "l1":
        loss_fn = nn.L1Loss()
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

    # Initialize validation losses list
    val_losses = []
    val_mae_scores = []

    model.train()
    for epoch in range(start_epoch, min(num_epochs, start_epoch + limit)):
        # Training phase
        epoch_loss = 0.0
        epoch_mae = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            scalar_property = batch.scalar_property.to(device)

            optimizer.zero_grad()
            pred_scalar_property = model(atom_type, edge_vec, edge_index, batch_index)
            mae = (pred_scalar_property - scalar_property).abs().mean()
            loss = loss_fn(pred_scalar_property, scalar_property)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mae += mae.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}", "mae": f"{mae.item():.6f}"})

        avg_loss = epoch_loss / num_batches
        avg_mae = epoch_mae / num_batches
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        
        # Validation phase
        val_loss, val_mae = validate_model(model, scalar_valset, device, loss_fn)
        val_losses.append(val_loss)
        val_mae_scores.append(val_mae)
        
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Train MAE: {avg_mae:.6f}, Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")

        # Save best model based on validation loss instead of training loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_checkpoint_path = checkpoint_path / property_name / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "train_mae": train_mae,
                    "val_losses": val_losses,
                    "val_mae_scores": val_mae_scores,
                },
                best_checkpoint_path,
            )
            print(f"Saved best model with val loss: {best_loss:.6f}, val MAE: {val_mae:.6f}")

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
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                    "train_mae": train_mae,
                    "val_losses": val_losses,
                    "val_mae_scores": val_mae_scores,
                },
                checkpoint_file,
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    print(f"Training completed. Best val loss: {best_loss:.6f}, Final val MAE: {val_mae:.6f}")
    plot_loss(train_losses, pic_dir, property_name)
    plot_mae(train_mae, pic_dir, property_name)
    plot_val_loss(train_losses, val_losses, pic_dir, property_name)
    plot_val_mae(train_mae, val_mae_scores, pic_dir, property_name)
    return model
