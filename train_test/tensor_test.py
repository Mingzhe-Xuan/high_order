import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader

from data import tensor_properties


def plot_tensor_test_results(y_true, y_pred, save_path: str, property_name: str):
    """
    Plot tensor test results comparing true vs predicted values
    """
    # Create subdirectories for different metrics
    results_dir = os.path.join(os.path.dirname(save_path), "tensor_test_results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{property_name}_test_results.png")

    plt.figure(figsize=(10, 8))
    
    # Convert tensors to numpy arrays if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Flatten tensors to compare individual elements
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Scatter plot
    plt.scatter(y_true_flat, y_pred_flat, alpha=0.6)
    
    # Perfect prediction line
    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Tensor Test Results: {property_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def calculate_tensor_metrics(y_true, y_pred):
    """
    Calculate various tensor-specific metrics for model evaluation
    """
    # Convert to tensors if they're not already
    if not torch.is_tensor(y_true):
        y_true = torch.tensor(y_true)
    if not torch.is_tensor(y_pred):
        y_pred = torch.tensor(y_pred)
    
    # Mean Absolute Error
    mae = F.l1_loss(y_pred, y_true)
    
    # Mean Squared Error
    mse = F.mse_loss(y_pred, y_true)
    
    # Root Mean Squared Error
    rmse = torch.sqrt(mse)
    
    # Pointwise MAE (flattened view)
    pointwise_mae = (y_pred - y_true).view(-1).abs().mean()
    
    # FNORM Error (Frobenius norm error)
    fnorm_error = torch.abs(
        torch.norm(y_pred, dim=-1) - torch.norm(y_true, dim=-1)
    )
    fnorm = torch.norm(y_true, dim=-1)
    mean_fnorm_percent_error = (fnorm_error / (fnorm + 1e-8)).mean()
    
    # Batchwise RSSE (Root Sum of Squared Errors)
    batchwise_rsse = (y_pred - y_true).view(-1).pow(2).sum().sqrt()
    batchwise_sum_fnorm = y_true.view(-1).pow(2).sum().sqrt()
    batchwise_percent_error = batchwise_rsse / (batchwise_sum_fnorm + 1e-8)
    
    # Mean Absolute Percentage Error (with epsilon to avoid division by zero)
    epsilon = 1e-8
    mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # R² Score (coefficient of determination)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': rmse.item(),
        'pointwise_mae': pointwise_mae.item(),
        'mean_fnorm_percent_error': mean_fnorm_percent_error.item(),
        'batchwise_percent_error': batchwise_percent_error.item(),
        'mape': mape.item(),
        'r2_score': r2_score.item()
    }


def tensor_test(
    tensor_models: dict[str, nn.Module],
    tensor_dataloaders: dict[str, DataLoader],
    # limit: int,
    pic_dir: str,
    metric_dir: str = "metrics",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move all models to the correct device and set to evaluation mode
    for key in tensor_models:
        tensor_models[key] = tensor_models[key].to(device)
        tensor_models[key].eval()
    
    results = {}

    with torch.no_grad():
        for prop in tensor_properties:
            testset = tensor_dataloaders[f"{prop}_testset"]
            
            all_true_values = []
            all_pred_values = []
            total_loss = 0.0
            num_batches = 0

            for batch in testset:
                atom_type = batch.atom_type.to(device)
                edge_index = batch.edge_index.to(device)
                edge_vec = batch.edge_vec.to(device)
                batch_index = batch.batch.to(device)
                tensor_property = batch.tensor_property.to(device)
                
                pred_tensor_property = tensor_models[prop](atom_type, edge_vec, edge_index, batch_index)
                
                # Store predictions and true values for metric calculation
                all_true_values.append(tensor_property.cpu())
                all_pred_values.append(pred_tensor_property.cpu())
                
                # Calculate batch loss for overall statistics
                batch_loss = F.mse_loss(pred_tensor_property, tensor_property)
                total_loss += batch_loss.item()
                num_batches += 1
            
            # Concatenate all true and predicted values
            all_true_values = torch.cat(all_true_values, dim=0)
            all_pred_values = torch.cat(all_pred_values, dim=0)
            
            # Calculate metrics
            metrics = calculate_tensor_metrics(all_true_values, all_pred_values)
            avg_loss = total_loss / num_batches
            
            # Print metrics
            print(f"\nTensor Test Results for {prop}:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  Pointwise MAE: {metrics['pointwise_mae']:.6f}")
            print(f"  Mean FNORM % Error: {metrics['mean_fnorm_percent_error']:.6f}%")
            print(f"  Batchwise % Error: {metrics['batchwise_percent_error']:.6f}%")
            print(f"  MAPE: {metrics['mape']:.6f}%")
            print(f"  R² Score: {metrics['r2_score']:.6f}")
            
            # Store results
            results[prop] = {
                'avg_loss': avg_loss,
                'metrics': metrics,
                'true_values': all_true_values,
                'predicted_values': all_pred_values
            }
            
            # Plot test results
            plot_tensor_test_results(all_true_values, all_pred_values, pic_dir, prop)

    # # Save metrics to JSON file
    # os.makedirs(metric_dir, exist_ok=True)
    # metrics_file_path = os.path.join(metric_dir, "tensor_test_metrics.json")
    # with open(metrics_file_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    # print(f"\nTensor test metrics saved to {metrics_file_path}")

    return results
