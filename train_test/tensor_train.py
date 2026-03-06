import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
from pathlib import Path

from src.model import Model
from data import get_tensor_dataloader
from src.train_test.utils.visualization import (
    get_visualization_dir,
    plot_all_train_val_metrics,
)
from src.train_test.utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
)


def validate_tensor_model(model, val_loader, device, loss_fn):
    """
    Validate the tensor model on the validation dataset.
    
    Returns:
        tuple: (avg_val_loss, avg_val_mae, avg_val_pointwise_mae, avg_val_mse, 
                avg_val_fnorm_err, avg_val_batch_err)
    """
    model.eval()
    val_loss = 0.0
    val_mae_sum = 0.0
    val_pointwise_mae_sum = 0.0
    val_mse_sum = 0.0
    val_fnorm_err_sum = 0.0
    val_batch_err_sum = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            tensor_property = batch.tensor_property.to(device)

            pred_tensor_property = model(atom_type, edge_vec, edge_index, batch_index)
            loss = loss_fn(pred_tensor_property, tensor_property)
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
            batchwise_rsse = (
                (pred_tensor_property - tensor_property).view(-1).pow(2).sum().sqrt()
            )
            batchwise_sum_fnorm = tensor_property.view(-1).pow(2).sum().sqrt()
            batchwise_percent_error = batchwise_rsse / batchwise_sum_fnorm
            
            val_loss += loss.item()
            val_mae_sum += pointwise_mae.item()
            val_pointwise_mae_sum += pointwise_mae.item()
            val_mse_sum += mse.item()
            val_fnorm_err_sum += mean_fnorm_percent_error.item()
            val_batch_err_sum += batchwise_percent_error.item()
            num_batches += 1

    avg_val_loss = val_loss / num_batches
    avg_val_mae = val_mae_sum / num_batches
    avg_val_pointwise_mae = val_pointwise_mae_sum / num_batches
    avg_val_mse = val_mse_sum / num_batches
    avg_val_fnorm_err = val_fnorm_err_sum / num_batches
    avg_val_batch_err = val_batch_err_sum / num_batches
    
    return avg_val_loss, avg_val_mae, avg_val_pointwise_mae, avg_val_mse, avg_val_fnorm_err, avg_val_batch_err


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
    num_epochs: int,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
    resume_from: str = None,
    save_interval: int = 5,
    clip_grad_norm: float = 1.0,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    loss_func: str = "huber",
    limit: int = None,
    use_amp: bool = False,
):
    """
    Train a tensor property prediction model with validation.

    Args:
        property_name: Name of the property to predict
        embedding_layer: Embedding layer of the model
        invariant_layers: Invariant layers of the model
        middle_mlp: Middle MLP layers
        equivariant_layers: Equivariant layers of the model
        final_mlp: Final MLP layers
        readout_layer: Readout layer
        tensor_trainset: Training dataset
        tensor_valset: Validation dataset
        num_epochs: Number of training epochs
        checkpoint_dir: Directory to save checkpoints (a subfolder with timestamp will be created)
        pic_dir: Directory to save plots (a subfolder with timestamp will be created)
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
        use_amp: Whether to use automatic mixed precision

    Returns:
        tuple: (model, training_history)
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
    train_mae = []
    train_pointwise_mae = []
    train_mse = []
    train_mean_fnorm_percent_error = []
    train_batchwise_percent_error = []

    val_losses = []
    val_mae_scores = []
    val_pointwise_mae = []
    val_mse_scores = []
    val_mean_fnorm_percent_error = []
    val_batchwise_percent_error = []

    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == "cuda" else None

    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, device)
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
        val_losses = checkpoint.get("val_losses", [])
        val_mae_scores = checkpoint.get("val_mae_scores", [])
        val_pointwise_mae = checkpoint.get("val_pointwise_mae", [])
        val_mse_scores = checkpoint.get("val_mse_scores", [])
        val_mean_fnorm_percent_error = checkpoint.get("val_mean_fnorm_percent_error", [])
        val_batchwise_percent_error = checkpoint.get("val_batchwise_percent_error", [])
        if use_amp and device.type == "cuda" and scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        print(f"Resumed from checkpoint: {resume_from}, epoch {start_epoch}")

    batches = tensor_trainset

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
        epoch_mae_sum = 0.0
        epoch_pointwise_mae_sum = 0.0
        epoch_mse_sum = 0.0
        epoch_fnorm_err_sum = 0.0
        epoch_batch_err_sum = 0.0
        num_batches = 0

        pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            atom_type = batch.atom_type.to(device)
            edge_index = batch.edge_index.to(device)
            edge_vec = batch.edge_vec.to(device)
            batch_index = batch.batch.to(device)
            tensor_property = batch.tensor_property.to(device)

            optimizer.zero_grad()
            
            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    pred_tensor_property = model(atom_type, edge_vec, edge_index, batch_index)
                    loss = loss_fn(pred_tensor_property, tensor_property)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_tensor_property = model(atom_type, edge_vec, edge_index, batch_index)
                loss = loss_fn(pred_tensor_property, tensor_property)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

                optimizer.step()
            
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
            batchwise_rsse = (
                (pred_tensor_property - tensor_property).view(-1).pow(2).sum().sqrt()
            )
            batchwise_sum_fnorm = tensor_property.view(-1).pow(2).sum().sqrt()
            batchwise_percent_error = batchwise_rsse / batchwise_sum_fnorm

            epoch_loss += loss.item()
            epoch_mae_sum += pointwise_mae.item()
            epoch_pointwise_mae_sum += pointwise_mae.item()
            epoch_mse_sum += mse.item()
            epoch_fnorm_err_sum += mean_fnorm_percent_error.item()
            epoch_batch_err_sum += batchwise_percent_error.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.6f}", 
                "mae": f"{pointwise_mae.item():.6f}",
                "mse": f"{mse.item():.6f}",
                "fnorm_err%": f"{mean_fnorm_percent_error.item():.6f}",
                "batch_err%": f"{batchwise_percent_error.item():.6f}"
            })

        avg_loss = epoch_loss / num_batches
        avg_mae = epoch_mae_sum / num_batches
        avg_pointwise_mae = epoch_pointwise_mae_sum / num_batches
        avg_mse = epoch_mse_sum / num_batches
        avg_fnorm_err = epoch_fnorm_err_sum / num_batches
        avg_batch_err = epoch_batch_err_sum / num_batches
        
        train_losses.append(avg_loss)
        train_mae.append(avg_mae)
        train_pointwise_mae.append(avg_pointwise_mae)
        train_mse.append(avg_mse)
        train_mean_fnorm_percent_error.append(avg_fnorm_err)
        train_batchwise_percent_error.append(avg_batch_err)
        
        val_loss, val_avg_mae, val_p_mae, val_mse, val_fnorm_err, val_batch_err = validate_tensor_model(
            model, tensor_valset, device, loss_fn
        )
        val_losses.append(val_loss)
        val_mae_scores.append(val_avg_mae)
        val_pointwise_mae.append(val_p_mae)
        val_mse_scores.append(val_mse)
        val_mean_fnorm_percent_error.append(val_fnorm_err)
        val_batchwise_percent_error.append(val_batch_err)
        
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Train MAE: {avg_mae:.6f}, "
            f"Train Pointwise MAE: {train_pointwise_mae[-1]:.6f}, Train MSE: {train_mse[-1]:.6f}, "
            f"Val Loss: {val_loss:.6f}, Val MAE: {val_avg_mae:.6f}, "
            f"Val Pointwise MAE: {val_p_mae:.6f}, Val MSE: {val_mse:.6f}, "
            f"Val Mean FNORM % Error: {val_fnorm_err:.6f}, "
            f"Val Batchwise % Error: {val_batch_err:.6f}"
        )

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_mae": val_avg_mae,
            "val_pointwise_mae": val_p_mae,
            "val_mse": val_mse,
            "val_mean_fnorm_percent_error": val_fnorm_err,
            "val_batchwise_percent_error": val_batch_err,
            "best_loss": best_loss,
            "train_losses": train_losses,
            "train_mae": train_mae,
            "train_pointwise_mae": train_pointwise_mae,
            "train_mse": train_mse,
            "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
            "train_batchwise_percent_error": train_batchwise_percent_error,
            "val_losses": val_losses,
            "val_mae_scores": val_mae_scores,
            "val_pointwise_mae": val_pointwise_mae,
            "val_mse_scores": val_mse_scores,
            "val_mean_fnorm_percent_error": val_mean_fnorm_percent_error,
            "val_batchwise_percent_error": val_batchwise_percent_error,
        }
        if use_amp and device.type == "cuda" and scaler is not None:
            checkpoint_data["scaler_state_dict"] = scaler.state_dict()

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name=property_name,
                num_epochs=num_epochs,
                is_best=True,
            )
            print(f"Saved best model with val loss: {best_loss:.6f} to {checkpoint_path}")

        if (epoch + 1) % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_data=checkpoint_data,
                checkpoint_base_dir=checkpoint_dir,
                property_name=property_name,
                num_epochs=num_epochs,
                epoch=epoch,
            )
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")

    print(f"Training completed. Best val loss: {best_loss:.6f}")

    vis_dir = get_visualization_dir(pic_dir)
    property_vis_dir = os.path.join(vis_dir, property_name)
    
    train_metrics = {
        "train_loss": train_losses,
        "train_mae": train_mae,
        "train_pointwise_mae": train_pointwise_mae,
        "train_mse": train_mse,
        "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
        "train_batchwise_percent_error": train_batchwise_percent_error,
    }
    
    val_metrics = {
        "val_loss": val_losses,
        "val_mae": val_mae_scores,
        "val_pointwise_mae": val_pointwise_mae,
        "val_mse": val_mse_scores,
        "val_mean_fnorm_percent_error": val_mean_fnorm_percent_error,
        "val_batchwise_percent_error": val_batchwise_percent_error,
    }
    
    plot_all_train_val_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        save_dir=property_vis_dir,
        property_name=property_name,
    )
    
    training_history = {
        "train_losses": train_losses,
        "train_mae": train_mae,
        "train_pointwise_mae": train_pointwise_mae,
        "train_mse": train_mse,
        "train_mean_fnorm_percent_error": train_mean_fnorm_percent_error,
        "train_batchwise_percent_error": train_batchwise_percent_error,
        "val_losses": val_losses,
        "val_mae_scores": val_mae_scores,
        "val_pointwise_mae": val_pointwise_mae,
        "val_mse_scores": val_mse_scores,
        "val_mean_fnorm_percent_error": val_mean_fnorm_percent_error,
        "val_batchwise_percent_error": val_batchwise_percent_error,
    }
    
    return model, training_history
