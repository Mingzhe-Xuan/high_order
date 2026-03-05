import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path

from src.model import Model
from data import get_mp_dataloader
from src.train_test.utils.visualization import get_visualization_dir, plot_train_val_metrics
from src.train_test.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def self_train(
    embedding_layer,
    invariant_layers,
    middle_mlp,
    equivariant_layers,
    final_mlp,
    readout_layer,
    dataloader,
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_from: str = None,
    save_interval: int = 5,
    clip_grad_norm: float = 1.0,
    loss_func: str = "huber",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    limit: int = None,
    use_amp: bool = False,
):
    """
    Self-supervised training for the model using force prediction.
    
    Args:
        embedding_layer: Embedding layer of the model
        invariant_layers: Invariant layers of the model
        middle_mlp: Middle MLP layers
        equivariant_layers: Equivariant layers of the model
        final_mlp: Final MLP layers
        readout_layer: Readout layer
        dataloader: Training dataloader
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints (a subfolder with timestamp will be created)
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
        start_epoch: Starting epoch (for resuming)
        resume_from: Path to resume from checkpoint
        save_interval: Interval to save checkpoints
        clip_grad_norm: Gradient clipping norm
        loss_func: Type of loss function ('huber', 'mse', 'l1')
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        optimizer: Type of optimizer ('adamw', 'adam', 'sgd')
        scheduler: Type of scheduler ('cosine_annealing', 'step')
        limit: Limit number of epochs (optional)
        use_amp: Whether to use automatic mixed precision

    Returns:
        model: The trained model
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

    best_loss = float("inf")
    train_losses = []

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        if use_amp and device.type == "cuda" and scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    batches = dataloader

    if loss_func == "huber":
        loss_fn = nn.HuberLoss()
    elif loss_func == "mse":
        loss_fn = nn.MSELoss()
    elif loss_func == "l1":
        loss_fn = nn.L1Loss()
    else:
        raise NotImplementedError(f"loss_func {loss_func} is not implemented")

    if optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"optimizer {optimizer} is not implemented")

    if scheduler == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise NotImplementedError(f"scheduler {scheduler} is not implemented")

    if limit is None:
        limit = num_epochs
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
            
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred_force = model(atom_type, unstable_edge_vec, edge_index, batch_index)
                    loss = loss_fn(pred_force, force)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
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

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "train_losses": train_losses,
        }
        if use_amp and device.type == "cuda" and scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name="self_train",
                num_epochs=num_epochs,
                is_best=True,
            )
            print(f"Saved best model with loss: {best_loss:.6f} to {checkpoint_path}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name="self_train",
                num_epochs=num_epochs,
                epoch=epoch,
            )
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    
    vis_dir = get_visualization_dir(pic_dir)
    self_train_vis_dir = os.path.join(vis_dir, "self_train")
    
    plot_train_val_metrics(
        train_values=train_losses,
        val_values=[],
        save_dir=self_train_vis_dir,
        property_name="self_train",
        metric_name="loss",
        train_color="blue",
        val_color="orange",
        title="Self-Training Loss Over Epochs",
        filename="self_train_loss.png",
    )
    
    return model
