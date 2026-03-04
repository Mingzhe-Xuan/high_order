import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import json
import os
from typing import Union
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader
import argparse


from data import (
    get_mp_dataloader,
    get_scalar_dataloaders_split,
    get_tensor_dataloaders_split,
    scalar_properties,
    tensor_properties,
    name_path_dict,
    readout_configs,
)
from src.train_test import (
    # self_train,
    # scalar_train,
    # tensor_train,
    # scalar_test,
    # tensor_test,
    train,
    test,
)
from src.train_test.utils.save_metrics import save_results_to_markdown

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
    print("Start loading data...")
    if need_self_train:
        self_trainset = get_mp_dataloader(
            cutoff=cutoff,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    else:
        self_trainset = None
    if need_scalar_train:
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
    else:
        scalar_dataloaders = None
    if need_tensor_train:
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
    else:
        tensor_dataloaders = None
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