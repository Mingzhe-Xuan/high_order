import torch
import torch.nn as nn
import json
import os
from typing import Union
from e3nn.o3 import Irreps
from torch_geometric.loader import DataLoader

from ..model.utils import get_irreps_from_ns_nv_lmax

from .self_train import self_train
from .scalar_train import scalar_train
from .tensor_train import tensor_train
from ...data import scalar_properties, tensor_properties, readout_configs
from ..model import (
    EmbeddingLayer,
    InvariantLayer,
    MiddleMLP,
    EquivariantLayer,
    FinalMLP,
    ReadoutLayer,
    Model,
)


def _create_shared_components(
    dist_emb_func,
    embed_dim,
    max_atom_type,
    cutoff,
    inv_update_method,
    num_inv_layers,
    num_equi_layers,
    equi_update_method,
    tp_method,
    scalar_dim,
    vec_dim,
    num_final_hidden_layers,
    final_scalar_hidden_dim,
    final_vec_hidden_dim,
    final_scalar_out_dim,
    final_vec_out_dim,
):
    """Create shared model components."""
    # Shared embedding layer
    embedding_layer = EmbeddingLayer(
        dist_emb_func=dist_emb_func,
        embed_dim=embed_dim,
        max_atom_type=max_atom_type,
        cutoff=cutoff,
    )

    # Shared invariant layers
    invariant_layers = nn.ModuleList(
        [
            InvariantLayer(update_method=inv_update_method, scalar_dim=embed_dim)
            for _ in range(num_inv_layers)
        ]
    )

    # Build irreps list
    irreps_list = [f"{scalar_dim}x0e"]
    l_max = num_equi_layers
    for l in range(1, l_max + 1):
        p = "e" if l % 2 == 0 else "o"
        irreps_list.append(f"{scalar_dim}x0e+{vec_dim}x{l}{p}")

    # Shared equivariant layers
    equivariant_layers = nn.ModuleList(
        [
            EquivariantLayer(
                update_method=equi_update_method,
                irreps_in=irreps_list[i],
                irreps_out=irreps_list[i + 1],
                tp_method=tp_method,
                residual=True,
            )
            for i in range(num_equi_layers)
        ]
    )

    final_irreps_hidden = get_irreps_from_ns_nv_lmax(
        ns=final_scalar_hidden_dim, nv=final_vec_hidden_dim, l_max=l_max
    )
    final_irreps_out = get_irreps_from_ns_nv_lmax(
        ns=final_scalar_out_dim, nv=final_vec_out_dim, l_max=l_max
    )

    return (
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        irreps_list,
        final_irreps_hidden,
        final_irreps_out,
        l_max,
    )


def _create_scalar_models(
    scalar_properties,
    embedding_layer,
    invariant_layers,
    equivariant_layers,
    irreps_list,
    final_irreps_hidden,
    final_irreps_out,
    final_pooling,
    embed_dim,
    scalar_dim,
    middle_scalar_hidden_dim,
    num_middle_hidden_layers,
    num_final_hidden_layers,
):
    """Create models for scalar properties."""
    models = {}

    for prop in scalar_properties:
        # Create model components for each property
        middle_mlp = MiddleMLP(
            scalar_dim_in=embed_dim,
            scalar_dim_hidden=middle_scalar_hidden_dim,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=num_middle_hidden_layers,
        )

        final_mlp = FinalMLP(
            irreps_in=irreps_list[-1],
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=num_final_hidden_layers,
        )

        readout_layer = ReadoutLayer(
            l_max=0,
            symmetry=None,
        )

        model = Model(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            middle_mlp=middle_mlp,
            equivariant_layers=equivariant_layers,
            final_mlp=final_mlp,
            readout_layer=readout_layer,
            self_train=False,
            final_pooling=final_pooling,
        )

        models[f"{prop}_model"] = model

    return models


def _create_tensor_models(
    tensor_properties,
    embedding_layer,
    invariant_layers,
    equivariant_layers,
    irreps_list,
    final_irreps_hidden,
    final_irreps_out,
    final_pooling,
    scalar_dim,
    vec_dim,
    middle_scalar_hidden_dim,
    num_middle_hidden_layers,
    num_final_hidden_layers,
):
    """Create models for tensor properties."""
    models = {}

    # Define readout layer configurations for different tensor properties
    # readout_configs = {
    #     "dielectric": {"l_max": 2, "symmetry": "ij=ji"},
    #     "dielectric_ionic": {"l_max": 2, "symmetry": "ij=ji"},
    #     "elastic_sym_kbar": {"l_max": 4, "symmetry": "ijkl=jikl=ijlk=klij"},
    #     "elastic_total_kbar": {"l_max": 4, "symmetry": "ijkl=jikl=ijlk=klij"},
    #     "piezoelectric_C_m2": {"l_max": 2, "symmetry": "i,jk=kj"},
    #     "piezoelectric_e_Angst": {"l_max": 2, "symmetry": "i,jk=kj"},
    # }

    for prop in tensor_properties:
        # Create model components for each property
        middle_mlp = MiddleMLP(
            scalar_dim_in=irreps_list[-1],  # Different from scalar models
            scalar_dim_hidden=middle_scalar_hidden_dim,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=num_middle_hidden_layers,
        )

        final_mlp = FinalMLP(
            irreps_in=irreps_list[-1],
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=num_final_hidden_layers,
        )

        readout_config = readout_configs.get(prop, {"l_max": 2, "symmetry": None})
        readout_layer = ReadoutLayer(
            l_max=readout_config["l_max"],
            symmetry=readout_config["symmetry"],
        )

        model = Model(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            middle_mlp=middle_mlp,
            equivariant_layers=equivariant_layers,
            final_mlp=final_mlp,
            readout_layer=readout_layer,
            self_train=False,
            final_pooling=final_pooling,
        )

        models[f"{prop}_model"] = model

    return models


def train(
    # # random seed
    # seed: int = 42,
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
    middle_scalar_hidden_dim: int = 128,  # hidden dim should be 2 times of the input by convention
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
    need_self_train: bool = True,
    need_scalar_train: bool = True,
    need_tensor_train: bool = True,
    self_trainset: DataLoader = None,
    scalar_dataloaders: dict[str, DataLoader] = None,
    tensor_dataloaders: dict[str, DataLoader] = None,
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
    assert (
        self_trainset is not None or not need_self_train
    ), "self_trainset is required when need_self_train is True"
    assert (
        scalar_dataloaders is not None or not need_scalar_train
    ), "scalar_dataloaders is required when need_scalar_train is True"
    assert (
        tensor_dataloaders is not None or not need_tensor_train
    ), "tensor_dataloaders is required when need_tensor_train is True"
    ################################ This part should be in main.py ##################################
    # # datasets
    # # 1. self trainset
    # # 2. scalar trainset, valset, testset
    # # 3. tensor trainset, valset, testset
    # with open("data/dataloaders/name_path.json") as f:
    #     name_path_dict = json.load(f)

    # if need_self_train:
    #     self_trainset = get_mp_dataloader(
    #         cutoff=cutoff,
    #         batch_size=batch_size,
    #         pin_memory=pin_memory,
    #         num_workers=num_workers,
    #         shuffle=True,
    #     )

    # # Create scalar dataloaders
    # if need_scalar_train:
    #     scalar_dataloaders = _create_scalar_dataloaders(
    #         name_path_dict,
    #         scalar_properties,
    #         cutoff,
    #         train_val_test,
    #         seed,
    #         batch_size,
    #         pin_memory,
    #         num_workers,
    #     )

    # # Create tensor dataloaders
    # if need_tensor_train:
    #     tensor_dataloaders = _create_tensor_dataloaders(
    #         name_path_dict,
    #         tensor_properties,
    #         cutoff,
    #         train_val_test,
    #         seed,
    #         batch_size,
    #         pin_memory,
    #         num_workers,
    #     )
    ##################################################################################################

    # Create shared model components
    (
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        irreps_list,
        final_irreps_hidden,
        final_irreps_out,
        l_max,
    ) = _create_shared_components(
        dist_emb_func,
        embed_dim,
        max_atom_type,
        cutoff,
        inv_update_method,
        num_inv_layers,
        num_equi_layers,
        equi_update_method,
        tp_method,
        scalar_dim,
        vec_dim,
        num_final_hidden_layers,
        final_scalar_hidden_dim,
        final_vec_hidden_dim,
        final_scalar_out_dim,
        final_vec_out_dim,
    )

    # Self train
    if need_self_train:
        self_middle_mlp = MiddleMLP(
            scalar_dim_in=embed_dim,
            scalar_dim_hidden=middle_scalar_hidden_dim,
            scalar_dim_out=scalar_dim,
            num_hidden_layers=num_middle_hidden_layers,
        )
        self_final_mlp = FinalMLP(
            irreps_in=irreps_list[-1],
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=num_final_hidden_layers,
        )
        self_readout_layer = ReadoutLayer(
            l_max=1,
            symmetry=None,
        )
        self_model = Model(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            equivariant_layers=equivariant_layers,
            middle_mlp=self_middle_mlp,
            final_mlp=self_final_mlp,
            readout_layer=self_readout_layer,
            self_train=True,
            final_pooling=final_pooling,
            irreps_list=irreps_list,
        )

        # Execute self training
        self_trained_model = self_train(
            embedding_layer=embedding_layer,
            invariant_layers=invariant_layers,
            middle_mlp=self_middle_mlp,
            equivariant_layers=equivariant_layers,
            final_mlp=self_final_mlp,
            readout_layer=self_readout_layer,
            dataloader=self_trainset,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
            pic_dir=pic_dir,
            start_epoch=start_epoch,
            resume_from=None,
            save_interval=save_interval,
            clip_grad_norm=clip_grad_norm,
            loss_func=self_loss_func,
            learning_rate=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            scheduler=scheduler,
            limit=self_limit,
        )

        # save the shared layers
        embedding_layer = self_trained_model.embedding_layer
        invariant_layers = self_trained_model.invariant_layers
        equivariant_layers = self_trained_model.equivariant_layers

    ######## Here the training strategy is train the model property by property,  ########
    ######## which can be improved to train one batch for each property one time. ########
    ######## Or simply repeat the training process multiple times.                ########
    # Scalar train - create models for all scalar properties
    if need_scalar_train:
        scalar_models = _create_scalar_models(
            scalar_properties,
            embedding_layer,
            invariant_layers,
            equivariant_layers,
            irreps_list,
            final_irreps_hidden,
            final_irreps_out,
            final_pooling,
            embed_dim,
            scalar_dim,
            middle_scalar_hidden_dim,
            num_middle_hidden_layers,
            num_final_hidden_layers,
        )

        # Execute scalar training for each property
        for prop in scalar_properties:
            scalar_models[f"{prop}_model"] = scalar_train(
                property_name=prop,
                embedding_layer=embedding_layer,
                invariant_layers=invariant_layers,
                middle_mlp=scalar_models[f"{prop}_model"].middle_mlp,
                equivariant_layers=equivariant_layers,
                final_mlp=scalar_models[f"{prop}_model"].final_mlp,
                readout_layer=scalar_models[f"{prop}_model"].readout_layer,
                scalar_trainset=scalar_dataloaders[f"{prop}_trainset"],
                scalar_valset=scalar_dataloaders[f"{prop}_valset"],
                num_epochs=num_epochs,
                checkpoint_dir=checkpoint_dir,
                pic_dir=pic_dir,
                start_epoch=start_epoch,
                resume_from=None,
                save_interval=save_interval,
                clip_grad_norm=clip_grad_norm,
                learning_rate=lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_func=scalar_loss_func,
                limit=scalar_limit,
            )

            # save the shared layers
            embedding_layer = scalar_models[f"{prop}_model"].embedding_layer
            invariant_layers = scalar_models[f"{prop}_model"].invariant_layers
            equivariant_layers = scalar_models[f"{prop}_model"].equivariant_layers

    # Tensor train - create models for all tensor properties
    if need_tensor_train:
        tensor_models = _create_tensor_models(
            tensor_properties,
            embedding_layer,
            invariant_layers,
            equivariant_layers,
            irreps_list,
            final_irreps_hidden,
            final_irreps_out,
            final_pooling,
            scalar_dim,
            vec_dim,
            middle_scalar_hidden_dim,
            num_middle_hidden_layers,
            num_final_hidden_layers,
        )

        # Execute tensor training for each property
        for prop in tensor_properties:
            tensor_models[f"{prop}_model"] = tensor_train(
                property_name=prop,
                embedding_layer=embedding_layer,
                invariant_layers=invariant_layers,
                middle_mlp=tensor_models[f"{prop}_model"].middle_mlp,
                equivariant_layers=equivariant_layers,
                final_mlp=tensor_models[f"{prop}_model"].final_mlp,
                readout_layer=tensor_models[f"{prop}_model"].readout_layer,
                tensor_trainset=tensor_dataloaders[f"{prop}_trainset"],
                tensor_valset=tensor_dataloaders[f"{prop}_valset"],
                num_epochs=num_epochs,
                checkpoint_dir=checkpoint_dir,
                pic_dir=pic_dir,
                start_epoch=start_epoch,
                resume_from=None,
                save_interval=save_interval,
                clip_grad_norm=clip_grad_norm,
                learning_rate=lr,
                weight_decay=weight_decay,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_func=tensor_loss_func,
                limit=tensor_limit,
            )

            # save the shared layers
            equi_shared = True
            if equi_shared:
                embedding_layer = tensor_models[f"{prop}_model"].embedding_layer
                invariant_layers = tensor_models[f"{prop}_model"].invariant_layers
                equivariant_layers = tensor_models[f"{prop}_model"].equivariant_layers

    return scalar_models, tensor_models, embedding_layer, invariant_layers, equivariant_layers
