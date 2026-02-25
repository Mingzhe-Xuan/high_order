import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model import Model
from data import get_mp_dataloader


def plot_loss(train_losses: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    path = os.path.join(os.path.dirname(save_path), "self_loss.png")

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


def self_train(
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    dataloader,
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
    loss_func: str = "huber",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
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

    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    # batches = get_mp_dataloader(
    #     cutoff=cutoff,
    #     batch_size=batch_size,
    #     pin_memory=pin_memory,
    #     num_workers=num_workers,
    #     shuffle=shuffle,
    # )

    batches = dataloader

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

    os.makedirs(checkpoint_path / "self_train", exist_ok=True)

    model.train()
    for epoch in range(start_epoch, min(num_epochs, start_epoch + limit)):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            unstable_edge_vec = batch.unstable_edge_vec.to(device)
            batch_index = batch.batch.to(device)
            force = batch.force.to(device)

            optimizer.zero_grad()
            pred_force = model(atom_type, unstable_edge_vec, edge_index, batch_index)
            loss = loss_fn(pred_force, force)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = checkpoint_path / "self_train" / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "best_loss": best_loss,
                    "train_losses": train_losses,
                },
                best_checkpoint_path,
            )
            print(f"Saved best model with loss: {best_loss:.6f}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_file = (
                checkpoint_path / "self_train" / f"checkpoint_epoch_{epoch+1}.pth"
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
                },
                checkpoint_file,
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    plot_loss(train_losses, pic_dir)
    return model
