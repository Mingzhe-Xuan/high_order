import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from typing import List, Union, Dict

from .train_test.train import train
from .train_test.test import test
from .model import Model


def load_model(model_paths: Dict[str, str]) -> nn.Module:
    # Embed weight, invariant weight, equivariant weight
    ...


def main(
    need_train: bool,
    invariant_layer: str,
    equivariant_layer: str,
    train_test_split: float,
    scalar_dim: int,
    max_atom_type: int,
    cutoff: float,
    emb_func: str,
    num_invariant_layers: int,
    irreps_list: Union[List[str], List[Irreps]],
    l_max: int,
    symmetry: str,
    model_paths: Union[Dict[str, str], None] = None,
    hidden_irreps_list: Union[List[str], List[Irreps], None] = None,
    tp_method: str = "fully_connected",
):

    if need_train:
        # Create model
        model = Model(
            l_max=l_max,
            symmetry=symmetry,
            max_atom_type=max_atom_type,
            cutoff=cutoff,
            emb_func=emb_func,
            num_invariant_layers=num_invariant_layers,
            invariant_layer=invariant_layer,
            equivariant_layer=equivariant_layer,
            scalar_dim=scalar_dim,
            irreps_list=irreps_list,
            tp_method=tp_method,
            hidden_irreps_list=hidden_irreps_list,
        )
        print("Training...")
        train(model, train_test_split)
    else:
        assert (
            model_paths is not None
        ), "model_paths must be provided when need_train is False"
        model = load_model(model_paths)
    print("Testing...")
    test(model, train_test_split)
    print("Done.")
