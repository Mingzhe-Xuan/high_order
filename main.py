import torch
import torch.nn as nn
import json
import os
from typing import Union
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader
import argparse


from ..data import (
    get_mp_dataloader,
    get_scalar_dataloaders_split,
    get_tensor_dataloaders_split,
    scalar_properties,
    tensor_properties,
    name_path_dict,
    readout_configs,
)
from .train_test import (
    # self_train,
    # scalar_train,
    # tensor_train,
    # scalar_test,
    # tensor_test,
    train,
    test,
)
from .train_test.utils.save_metrics import save_results_to_markdown

def _create_scalar_dataloaders(
    name_path_dict,
    scalar_properties,
    cutoff,
    train_val_test,
    seed,
    batch_size,
    pin_memory,
    num_workers,
):
    """Create dataloaders for scalar properties."""
    dataloaders = {}
    for prop in scalar_properties:
        trainset, valset, testset = get_scalar_dataloaders_split(
            path=name_path_dict[prop],
            property_name=prop,
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
        dataloaders[f"{prop}_trainset"] = trainset
        dataloaders[f"{prop}_valset"] = valset
        dataloaders[f"{prop}_testset"] = testset
    return dataloaders


def _create_tensor_dataloaders(
    name_path_dict,
    tensor_properties,
    cutoff,
    train_val_test,
    seed,
    batch_size,
    pin_memory,
    num_workers,
):
    """Create dataloaders for tensor properties."""
    dataloaders = {}
    for prop in tensor_properties:
        trainset, valset, testset = get_tensor_dataloaders_split(
            path=name_path_dict[prop],
            property_name=prop,
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
        dataloaders[f"{prop}_trainset"] = trainset
        dataloaders[f"{prop}_valset"] = valset
        dataloaders[f"{prop}_testset"] = testset
    return dataloaders


def main(
    cutoff: float = 5.0,
    batch_size: int = 32,
    pin_memory: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    # model
    # embedding layer
    dist_emb_func: str = "gaussian",
    embed_dim: int = 64,
    max_atom_type: int = 118,
    # invariant layers
    inv_update_method: str = "comformer",
    num_inv_layers: int = 3,
    # middle_mlp
    middle_scalar_hidden_dim: int = 128,  # hidden dim should be 2 times of the input by convention
    num_middle_hidden_layers: int = 1,
    # equivariant layers
    equi_update_method: str = "tpconv_with_edge",
    num_equi_layers: int = 3,
    tp_method: str = "so2",
    scalar_dim: int = 16,
    vec_dim: int = 8,
    # final mlp
    num_final_hidden_layers: int = 1,
    final_scalar_hidden_dim: int = 64,
    final_vec_hidden_dim: int = 16,
    final_scalar_out_dim: int = 16,
    final_vec_out_dim: int = 8,
    # train
    need_self_train: bool = True,
    need_scalar_train: bool = True,
    need_tensor_train: bool = True,
    # self_trainset: DataLoader = None,
    # scalar_dataloaders: dict[str, DataLoader] = None,
    # tensor_dataloaders: dict[str, DataLoader] = None,
    final_pooling: bool = False,
    # train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    # batch_size: int = 32,
    # num_workers: int = 0,
    # pin_memory: bool = True,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    clip_grad_norm: float = 1.0,
    save_interval: int = 5,
    optimizer: str = "adamw",
    scheduler: str = "cosine_annealing",
    self_loss_func: str = "huber",
    scalar_loss_func: str = "huber",
    tensor_loss_func: str = "huber",
    self_train_limit: int = None,
    scalar_train_limit: int = None,
    tensor_train_limit: int = None,
    # test
    # scalar_test_limit: int = None,
    # tensor_test_limit: int = None,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    metric_dir: str = "metrics",
    start_epoch: int = 0,
):
    # scalar_properties = [
    #     "formation_energy",
    #     "opt_bandgap",
    #     "total_energy",
    #     "ehull",
    #     "mbj_bandgap",
    #     "bandgap",
    #     "e_form",
    #     "bulk_modulus",
    #     "shear_modulus",
    # ]
    # tensor_properties = [
    #     "dielectric",
    #     "dielectric_ionic",
    #     "elastic_sym_kbar",
    #     "elastic_total_kbar",
    #     "piezoelectric_C_m2",
    #     "piezoelectric_e_Angst",
    # ]
    self_trainset = get_mp_dataloader(
        cutoff=cutoff,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    scalar_dataloaders = _create_scalar_dataloaders(
        name_path_dict,
        scalar_properties,
        cutoff,
        train_val_test,
        seed,
        batch_size,
        pin_memory,
        num_workers,
    )
    tensor_dataloaders = _create_tensor_dataloaders(
        name_path_dict,
        tensor_properties,
        cutoff,
        train_val_test,
        seed,
        batch_size,
        pin_memory,
        num_workers,
    )
    # Train
    print("Start training...")
    (
        scalar_models,
        tensor_models,
        embedding_layer,
        invariant_layers,
        equivariant_layers,
    ) = train(
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        cutoff=cutoff,
        # invariant layers
        inv_update_method=inv_update_method,
        # inv_dim: int = 64,
        num_inv_layers=num_inv_layers,
        # middle_mlp
        middle_scalar_hidden_dim=middle_scalar_hidden_dim,  # hidden dim should be 2 times of the input by convention
        num_middle_hidden_layers=num_middle_hidden_layers,
        # equivariant layers
        equi_update_method=equi_update_method,
        num_equi_layers=num_equi_layers,
        tp_method=tp_method,
        scalar_dim=scalar_dim,
        vec_dim=vec_dim,
        # irreps_list: Union[list[str], list[Irreps]] = ["128x0e", ""],
        # final mlp
        num_final_hidden_layers=num_final_hidden_layers,
        final_scalar_hidden_dim=final_scalar_hidden_dim,
        final_vec_hidden_dim=final_vec_hidden_dim,
        final_scalar_out_dim=final_scalar_out_dim,
        final_vec_out_dim=final_vec_out_dim,
        # train
        need_self_train=need_self_train,
        need_scalar_train=need_scalar_train,
        need_tensor_train=need_tensor_train,
        self_trainset=self_trainset,
        scalar_dataloaders=scalar_dataloaders,
        tensor_dataloaders=tensor_dataloaders,
        final_pooling=final_pooling,
        # train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
        # batch_size: int = 32,
        # num_workers: int = 0,
        # pin_memory: bool = True,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
        save_interval=save_interval,
        optimizer=optimizer,
        scheduler=scheduler,
        self_loss_func=self_loss_func,
        scalar_loss_func=scalar_loss_func,
        tensor_loss_func=tensor_loss_func,
        self_limit=self_train_limit,
        scalar_limit=scalar_train_limit,
        tensor_limit=tensor_train_limit,
        checkpoint_dir=checkpoint_dir,
        pic_dir=pic_dir,
        start_epoch=start_epoch,
    )
    # Test
    print("Start testing...")
    scalar_results, tensor_results = test(
        scalar_models=scalar_models,
        tensor_models=tensor_models,
        scalar_dataloaders=scalar_dataloaders,
        tensor_dataloaders=tensor_dataloaders,
        # scalar_limit=scalar_test_limit,
        # tensor_limit=tensor_test_limit,
        pic_dir=pic_dir,
        # metric_dir=metric_dir,
    )
    # Save parameter information, scalar_results, tensor_results 
    # in a markdown file in a clear and well-organized format
    save_results_to_markdown(
        params=locals(),
        scalar_results=scalar_results,
        tensor_results=tensor_results,
        metric_dir=metric_dir,
    )

    print("Done.")

if __name__ == "__main__":
    # Use argparse here to parse all parameters of main from command lines.
    parser = argparse.ArgumentParser(description='High-order Equivariant Network Training')
    
    # Data parameters
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance for graph construction (default: 5.0)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for dataloaders (default: 32)')
    parser.add_argument('--pin-memory', action='store_true', default=True, help='Pin memory for dataloaders (default: True)')
    parser.add_argument('--no-pin-memory', dest='pin_memory', action='store_false', help='Disable pin memory for dataloaders')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers for dataloaders (default: 0)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--train-val-test', type=float, nargs=3, default=[0.8, 0.1, 0.1], 
                        help='Train/validation/test split ratios (default: 0.8 0.1 0.1)')
    
    # Model parameters - embedding layer
    parser.add_argument('--dist-emb-func', type=str, default='gaussian', 
                        choices=['gaussian', 'bessel', 'polynomial'], 
                        help='Distance embedding function (default: gaussian)')
    parser.add_argument('--embed-dim', type=int, default=64, help='Embedding dimension (default: 64)')
    parser.add_argument('--max-atom-type', type=int, default=118, help='Maximum atom type (default: 118)')
    
    # Model parameters - invariant layers
    parser.add_argument('--inv-update-method', type=str, default='comformer',
                        choices=['comformer', 'mlp', 'attention'],
                        help='Invariant update method (default: conformer)')
    parser.add_argument('--num-inv-layers', type=int, default=3, help='Number of invariant layers (default: 3)')
    
    # Model parameters - middle MLP
    parser.add_argument('--middle-scalar-hidden-dim', type=int, default=128, 
                        help='Hidden dimension for middle scalar layers (default: 128)')
    parser.add_argument('--num-middle-hidden-layers', type=int, default=1, 
                        help='Number of middle hidden layers (default: 1)')
    
    # Model parameters - equivariant layers
    parser.add_argument('--equi-update-method', type=str, default='tpconv_with_edge',
                        choices=['tpconv_with_edge', 'linear', 'conv'],
                        help='Equivariant update method (default: tpconv_with_edge)')
    parser.add_argument('--num-equi-layers', type=int, default=3, help='Number of equivariant layers (default: 3)')
    parser.add_argument('--tp-method', type=str, default='so2', choices=['so2', 'so3'],
                        help='Tensor product method (default: so2)')
    parser.add_argument('--scalar-dim', type=int, default=16, help='Scalar dimension (default: 16)')
    parser.add_argument('--vec-dim', type=int, default=8, help='Vector dimension (default: 8)')
    
    # Model parameters - final MLP
    parser.add_argument('--num-final-hidden-layers', type=int, default=1, 
                        help='Number of final hidden layers (default: 1)')
    parser.add_argument('--final-scalar-hidden-dim', type=int, default=64, 
                        help='Final scalar hidden dimension (default: 64)')
    parser.add_argument('--final-vec-hidden-dim', type=int, default=16, 
                        help='Final vector hidden dimension (default: 16)')
    parser.add_argument('--final-scalar-out-dim', type=int, default=16, 
                        help='Final scalar output dimension (default: 16)')
    parser.add_argument('--final-vec-out-dim', type=int, default=8, 
                        help='Final vector output dimension (default: 8)')
    
    # Training parameters
    parser.add_argument('--need-self-train', action='store_true', default=True, 
                        help='Enable self training (default: True)')
    parser.add_argument('--no-need-self-train', dest='need_self_train', action='store_false',
                        help='Disable self training')
    parser.add_argument('--need-scalar-train', action='store_true', default=True, 
                        help='Enable scalar training (default: True)')
    parser.add_argument('--no-need-scalar-train', dest='need_scalar_train', action='store_false',
                        help='Disable scalar training')
    parser.add_argument('--need-tensor-train', action='store_true', default=True, 
                        help='Enable tensor training (default: True)')
    parser.add_argument('--no-need-tensor-train', dest='need_tensor_train', action='store_false',
                        help='Disable tensor training')
    parser.add_argument('--final-pooling', action='store_true', default=False, 
                        help='Enable final pooling (default: False)')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (default: 1e-5)')
    parser.add_argument('--clip-grad-norm', type=float, default=1.0, help='Gradient clipping norm (default: 1.0)')
    parser.add_argument('--save-interval', type=int, default=5, help='Model save interval (default: 5)')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='cosine_annealing', 
                        choices=['cosine_annealing', 'step', 'exponential'],
                        help='Learning rate scheduler (default: cosine_annealing)')
    parser.add_argument('--self-loss-func', type=str, default='huber', 
                        choices=['mse', 'mae', 'huber'],
                        help='Self loss function (default: huber)')
    parser.add_argument('--scalar-loss-func', type=str, default='huber', 
                        choices=['mse', 'mae', 'huber'],
                        help='Scalar loss function (default: huber)')
    parser.add_argument('--tensor-loss-func', type=str, default='huber', 
                        choices=['mse', 'mae', 'huber'],
                        help='Tensor loss function (default: huber)')
    parser.add_argument('--self-train-limit', type=int, default=None, 
                        help='Limit for self training samples (default: None)')
    parser.add_argument('--scalar-train-limit', type=int, default=None, 
                        help='Limit for scalar training samples (default: None)')
    parser.add_argument('--tensor-train-limit', type=int, default=None, 
                        help='Limit for tensor training samples (default: None)')
    
    # Directory parameters
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
                        help='Checkpoint directory (default: checkpoints)')
    parser.add_argument('--pic-dir', type=str, default='pics', 
                        help='Picture output directory (default: pics)')
    parser.add_argument('--metric-dir', type=str, default='metrics', 
                        help='Metrics output directory (default: metrics)')
    parser.add_argument('--start-epoch', type=int, default=0, 
                        help='Starting epoch for training (default: 0)')
    
    args = parser.parse_args()
    main(**vars(args))