import torch
import torch.nn as nn
import json
import os
from typing import Union
from e3nn.o3 import Irreps

from ...data import (
    get_mp_dataloader,
    get_scalar_dataloaders_split,
    get_tensor_dataloaders_split,
)
from .self_train import self_train
from .scalar_train import scalar_train
from .tensor_train import tensor_train
from ..model import (
    EmbeddingLayer,
    InvariantLayer,
    MiddleMLP,
    EquivariantLayer,
    FinalMLP,
    ReadoutLayer,
    Model,
)


def train(
    # random seed
    seed: int = 42,
    # model
    # embedding layer
    dist_emb_func: str = "gaussian",
    embed_dim: int = 64,
    max_atom_type: int = 118,
    cutoff: float = 5.0,
    # invariant layers
    inv_update_method: str = "comformer",
    # inv_dim: int = 64,
    num_inv_layers: int = 3,
    # middle_mlp
    middle_scalar_hidden_dim: int = 128, # hidden dim should be 2 times of the input by convention
    num_middle_hidden_layers: int = 1,
    # equivariant layers
    equi_update_method: str = "tpconv_with_edge",
    num_equi_layers: int = 3,
    tp_method: str = "so2",
    scalar_dim: int = 16,
    vec_dim: int = 8,
    # irreps_list: Union[list[str], list[Irreps]] = ["128x0e", ""],
    # final mlp
    num_final_hidden_layers: int = 1,
    final_scalar_hidden_dim: int = 64,
    final_vec_hidden_dim: int = 16,
    final_scalar_out_dim: int = 16,
    final_vec_out_dim: int = 8,
    # train
    train_val_test: tuple[float, float, float] = (0.8, 0.1, 0.1),
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
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
    self_limit: int = None,
    scalar_limit: int = None,
    tensor_limit: int = None,
    checkpoint_dir: str = "checkpoints",
    pic_dir: str = "pics",
    start_epoch: int = 0,
):
    """
    1. self train: emb, inv, eqv, deco, readout - scalar train: emb, inv, deco - tensor train: emb, inv, eqv, deco, readout
    2. different readout layers, different final mlps, shared embedding, invariant and equivariant layers
    3. use comformer as invariant layer, tpconv_with_edge by default
    4. different scalar properties: train on different scalar properties one batch each time with different models, and so on.
    """
    # datasets
    # 1. self trainset
    # 2. scalar trainset, valset, testset
    # 3. tensor trainset, valset, testset
    self_trainset = get_mp_dataloader(
        cutoff=cutoff,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )

    with open("data/dataloaders/path_name.json") as f:
        path_name_dict = json.load(f)

    scalar_properties = [
        "formation_energy",
        "opt_bandgap",
        "total_energy",
        "ehull",
        "mbj_bandgap",
        "bandgap",
        "e_form",
        "bulk_modulus",
        "shear_modulus",
    ]
    formation_energy_trainset, formation_energy_valset, formation_energy_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["formation_energy"],
            property_name="formation_energy",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    opt_bandgap_trainset, opt_bandgap_valset, opt_bandgap_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["opt_bandgap"],
            property_name="opt_bandgap",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    total_energy_trainset, total_energy_valset, total_energy_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["total_energy"],
            property_name="total_energy",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    ehull_trainset, ehull_valset, ehull_testset = get_scalar_dataloaders_split(
        path=path_name_dict["ehull"],
        property_name="ehull",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    mbj_bandgap_trainset, mbj_bandgap_valset, mbj_bandgap_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["mbj_bandgap"],
            property_name="mbj_bandgap",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    bandgap_trainset, bandgap_valset, bandgap_testset = get_scalar_dataloaders_split(
        path=path_name_dict["bandgap"],
        property_name="bandgap",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    e_form_trainset, e_form_valset, e_form_testset = get_scalar_dataloaders_split(
        path=path_name_dict["e_form"],
        property_name="e_form",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    bulk_modulus_trainset, bulk_modulus_valset, bulk_modulus_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["bulk_modulus"],
            property_name="bulk_modulus",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    shear_modulus_trainset, shear_modulus_valset, shear_modulus_testset = (
        get_scalar_dataloaders_split(
            path=path_name_dict["shear_modulus"],
            property_name="shear_modulus",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )

    tensor_properties = [
        "dielectric",
        "dielectric_ionic",
        "elastic_sym_kbar",
        "elastic_total_kbar",
        "piezoelectric_C_m2",
        "piezoelectric_e_Angst",
    ]
    dielectric_trainset, dielectric_valset, dielectric_testset = (
        get_tensor_dataloaders_split(
            path=path_name_dict["dielectric"],
            property_name="dielectric",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    dielectric_ionic_trainset, dielectric_ionic_valset, dielectric_ionic_testset = (
        get_tensor_dataloaders_split(
            path=path_name_dict["dielectric_ionic"],
            property_name="dielectric_ionic",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    elastic_sym_kbar_trainset, elastic_sym_kbar_valset, elastic_sym_kbar_testset = (
        get_tensor_dataloaders_split(
            path=path_name_dict["elastic_sym_kbar"],
            property_name="elastic_sym_kbar",
            cutoff=cutoff,
            train_val_test=train_val_test,
            seed=seed,
            batch_size=batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=True,
        )
    )
    (
        elastic_total_kbar_trainset,
        elastic_total_kbar_valset,
        elastic_total_kbar_testset,
    ) = get_tensor_dataloaders_split(
        path=path_name_dict["elastic_total_kbar"],
        property_name="elastic_total_kbar",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    (
        piezoelectric_C_m2_trainset,
        piezoelectric_C_m2_valset,
        piezoelectric_C_m2_testset,
    ) = get_tensor_dataloaders_split(
        path=path_name_dict["piezoelectric_C_m2"],
        property_name="piezoelectric_C_m2",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )
    (
        piezoelectric_e_Angst_trainset,
        piezoelectric_e_Angst_valset,
        piezoelectric_e_Angst_testset,
    ) = get_tensor_dataloaders_split(
        path=path_name_dict["piezoelectric_e_Angst"],
        property_name="piezoelectric_e_Angst",
        cutoff=cutoff,
        train_val_test=train_val_test,
        seed=seed,
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        shuffle=True,
    )

    # models
    # 1. shared embedding layer
    # 2. shared invariant layers
    # 3. shared middle_mlp, equivariant layers for self train
    # 4. shared equivariant for scalar train
    # 4. different middle_mlps, final mlps, readout layers for each property in scalar train and tensor train
    embedding_layer = EmbeddingLayer(
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        cutoff=cutoff,
    )
    invariant_layers = nn.ModuleList(
        [
            InvariantLayer(update_method=inv_update_method, scalar_dim=embed_dim)
            for _ in range(num_inv_layers)
        ]
    )
    # middle_mlp = MiddleMLP(
    #     scalar_dim_in=embed_dim,
    #     scalar_dim_hidden=middle_scalar_hidden_dim,
    #     num_hidden_layers=num_middle_hidden_layers,
    # )

    irreps_list = [f"{scalar_dim}x0e"]
    l_max = num_equi_layers
    for l in range(1, l_max + 1):
        p = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"{scalar_dim}x0e+{vec_dim}x{l}{p}")

    equivariant_layers = nn.ModuleList(
        [
            EquivariantLayer(
                update_method=equi_update_method,
                irreps_in=irreps_list[i],
                irreps_out=irreps_list[i+1],
                tp_method=tp_method,
                residual=True,
            )
            for i in range(num_equi_layers)
        ]
    )

    self_middle_mlp = MiddleMLP(
        scalar_dim_in=embed_dim,
        scalar_dim_hidden=middle_scalar_hidden_dim,
        scalar_dim_out=scalar_dim,
        num_hidden_layers=num_middle_hidden_layers,
    )
    self_final_mlp = FinalMLP(
        irreps_in=irreps_list[-1],
        irreps_out=
        num_hidden_layers=num_final_hidden_layers,
    )
