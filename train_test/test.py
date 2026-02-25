import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import scalar_test, tensor_test


def test(
    scalar_models: dict[str, nn.Module],
    tensor_models: dict[str, nn.Module],
    scalar_dataloaders: dict[str, DataLoader],
    tensor_dataloaders: dict[str, DataLoader],
    pic_dir: str,
    metric_dir: str,
):
    scalar_results = scalar_test(
        scalar_models=scalar_models,
        scalar_dataloaders=scalar_dataloaders,
        pic_dir=pic_dir,
        metric_dir=metric_dir,
    )
    tensor_results = tensor_test(
        tensor_models=tensor_models,
        tensor_dataloaders=tensor_dataloaders,
        pic_dir=pic_dir,
        metric_dir=metric_dir,
    )
    return scalar_results, tensor_results
    
