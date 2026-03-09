import torch
import json
from typing import Dict, Any, List, Optional
from pathlib import Path


def count_parameters(module: torch.nn.Module) -> int:
    """Count the total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def count_parameters_by_layer(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Count parameters for each layer in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with layer names and their parameter counts
    """
    layer_info = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if param_count > 0:
                layer_info[name] = {
                    "layer_type": module.__class__.__name__,
                    "parameters": param_count,
                    "trainable": True
                }
    
    return layer_info


def get_model_state_summary(
    model: torch.nn.Module,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Get a comprehensive summary of the model state.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        
    Returns:
        Dictionary containing model state information
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    
    total_params = count_parameters(model)
    layer_params = count_parameters_by_layer(model)
    
    summary = {
        "model_name": model_name,
        "total_parameters": total_params,
        "trainable_parameters": total_params,
        "device": str(device),
        "layers": layer_params
    }
    
    return summary


def save_num_params_markdown(
    model_summaries: List[Dict[str, Any]],
    output_path: str,
    params_dict: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model parameter summaries to a markdown file.
    
    Args:
        model_summaries: List of model summary dictionaries
        output_path: Path to save the markdown file
        params_dict: Optional dictionary of training parameters to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Model Parameters Summary\n\n")
        f.write(f"Generated on: {torch.utils.data.get_worker_info()}\n\n")
        
        if params_dict:
            f.write("## Training Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(params_dict, indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
        
        f.write("## Model Architecture and Parameters\n\n")
        
        for summary in model_summaries:
            model_name = summary.get("model_name", "Model")
            total_params = summary.get("total_parameters", 0)
            device = summary.get("device", "unknown")
            layers = summary.get("layers", {})
            
            f.write(f"### {model_name}\n\n")
            f.write(f"- **Total Parameters**: {total_params:,}\n")
            f.write(f"- **Device**: {device}\n\n")
            
            if layers:
                f.write("| Layer Name | Layer Type | Parameters |\n")
                f.write("|------------|------------|------------|\n")
                
                for layer_name, layer_info in layers.items():
                    layer_type = layer_info.get("layer_type", "Unknown")
                    param_count = layer_info.get("parameters", 0)
                    f.write(f"| {layer_name} | {layer_type} | {param_count:,} |\n")
                
                f.write("\n")
        
        f.write("---\n\n")
        f.write("*This file was automatically generated before training.*\n")


def analyze_model_components(
    embedding_layer: torch.nn.Module,
    invariant_layers: torch.nn.ModuleList,
    middle_mlp: torch.nn.Module,
    equivariant_layers: torch.nn.ModuleList,
    final_mlp: torch.nn.Module,
    readout_layer: Optional[torch.nn.Module] = None
) -> List[Dict[str, Any]]:
    """
    Analyze all model components and return their parameter summaries.
    
    Args:
        embedding_layer: Embedding layer
        invariant_layers: Invariant layers
        middle_mlp: Middle MLP
        equivariant_layers: Equivariant layers
        final_mlp: Final MLP
        readout_layer: Readout layer (optional)
        
    Returns:
        List of model summary dictionaries
    """
    summaries = []
    
    summaries.append(get_model_state_summary(embedding_layer, "EmbeddingLayer"))
    
    for i, inv_layer in enumerate(invariant_layers):
        summaries.append(get_model_state_summary(inv_layer, f"InvariantLayer_{i}"))
    
    summaries.append(get_model_state_summary(middle_mlp, "MiddleMLP"))
    
    for i, equiv_layer in enumerate(equivariant_layers):
        summaries.append(get_model_state_summary(equiv_layer, f"EquivariantLayer_{i}"))
    
    summaries.append(get_model_state_summary(final_mlp, "FinalMLP"))
    
    if readout_layer is not None:
        summaries.append(get_model_state_summary(readout_layer, "ReadoutLayer"))
    
    return summaries
