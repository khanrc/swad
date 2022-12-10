"""Make Loss surface plane ((w1, w2, w3)-plane) and infer the grids.
https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py
"""
import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm
from munch import Munch

from domainbed.algorithms import get_algorithm_class
from domainbed.lib import swa_utils
from domainbed.datasets import get_dataset, split_dataset
from domainbed import evaluator
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.lib.logger import Logger
from infer_utils import load_algorithms, setup_eval_meta

logger = Logger.get()


def params_to_vector(parameters):
    return torch.cat(list(map(lambda x: x.detach().flatten(), parameters)))


def get_xy(point, origin, vector_x, vector_y):
    return torch.as_tensor(
        [
            torch.dot(point - origin, vector_x),
            torch.dot(point - origin, vector_y)
        ]
    )


def get_basis(w1, w2, w3):
    """https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py#L105

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
    """
    u = w2 - w1
    du = u.norm()
    u /= du

    v = w3 - w1
    v -= u.dot(v) * u
    dv = v.norm()
    v /= dv

    return u, v, du, dv


def copy_flat_params_(flat_params, model):
    offset = 0
    for p in model.parameters():
        size = p.numel()
        ip = flat_params[offset:offset+size].view(p.shape)
        with torch.no_grad():
            p.copy_(ip)
        offset += size


def infer_grid(w1, w2, w3, base_model, test_loader, G, margin=0.2):
    """Make a grid by (w1, w2, w3)-plane and infer for each grid point.
    https://github.com/timgaripov/dnn-mode-connectivity/blob/master/plane.py

    Args:
        w1, w2, w3: 1-dim torch tensor (vector)
        base_model: model for architecture.
        test_loader: dataloader for test env.
        G: n_grid_points (per axis); total points = G * G.
        margin
    """
    u, v, du, dv = get_basis(w1, w2, w3)

    alphas = np.linspace(0. - margin, 1. + margin, G)
    betas = np.linspace(0. - margin, 1. + margin, G)

    results = []

    for i, w in enumerate([w1, w2, w3]):
        c = get_xy(w, w1, u, v)

        results.append({
            "ij": f"w{i+1}",
            "grid_xy": c
        })

    tk = tqdm(total=G*G)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            tk.set_description(f"i={i+1}/{G}, j={j+1}/{G}")
            interpolated = w1 + alpha * du * u + beta * dv * v
            copy_flat_params_(interpolated, base_model)
            # update_bn -> skip

            acc, loss, _sam = evaluator.accuracy(base_model, test_loader, None, compute_sam=False)
            #  c = get_xy(interpolated, w1, u, v)
            #  c == [alpha * dx, beta * dy] -> it has a little residual < 0.01.

            results.append({
                "ij": [i, j],
                "grid_xy": torch.as_tensor([alpha * du, beta * dv]),
                "error": 1. - acc,
                "loss": loss
            })
            tk.update(1)

    return results


def run(ckpt_dir, test_env, G=None, margin=0.6, mode='all', first=5, middle=7, last=9, max_dset=3000):
    """
    Args:
        G: # of ticks for each axis
        margin: horizontal & vertical margin
        mode: 'test' (test-in only) / 'train' (train in/out) / 'all'
        first, middle, last: checkpoint index
    """
    if G is None:
        G = int((1.0 + margin*2) * 15)
        logger.info(f"G = {G}")
    assert mode in ['test', 'train', 'all']

    ckpt_dir = Path(ckpt_dir)
    test_envs = [test_env]
    ckpt_paths = list(ckpt_dir.glob(f"TE{test_env}_*.pth"))

    # load checkpoints
    logger.info("# Load checkpoints ...")
    load_indices = [first, middle, last]
    algos, dataset, in_splits, out_splits, margs, mhparams = load_algorithms(
        ckpt_paths, test_envs, load_indices=load_indices
    )

    # setup eval loaders
    eval_meta_all = setup_eval_meta(dataset, in_splits, out_splits, margs, mhparams)

    # filter
    eval_meta = []
    if mode in ['test', 'all']:
        eval_meta += [
            (*row, 'test') for row in eval_meta_all
            if row[0] == f"env{test_env}_in"
        ]
    if mode in ['train', 'all']:
        eval_meta += [
            (*row, 'train') for row in eval_meta_all
            if not row[0].startswith(f"env{test_env}")
        ]

    logger.info(f"First/Middle/Last: {first}/{middle}/{last}")
    w1 = params_to_vector(algos[0].parameters())
    w2 = params_to_vector(algos[1].parameters())
    w3 = params_to_vector(algos[2].parameters())

    base_model = copy.deepcopy(algos[0]).cuda()

    # Build test dataloader
    for i, (name, loader_kwargs, _weights, typ) in enumerate(eval_meta):
        dset = loader_kwargs["dataset"]
        if max_dset and len(dset) > max_dset:
            limited_dset, _ = split_dataset(dset, max_dset)
            logger.info("Dataset is sampled: #%d -> #%d" % (len(dset), len(limited_dset)))
            loader_kwargs["dataset"] = limited_dset
        loader_kwargs["batch_size"] = 128
        test_loader = FastDataLoader(**loader_kwargs)

        logger.info(f"[{i+1}/{len(eval_meta)}] Start inference for {name} ...")
        results = infer_grid(w1, w2, w3, base_model, test_loader, G=G, margin=margin)

        unique_name = ckpt_dir.parent.stem
        runoption = f"G{G}_margin{margin}_[{first}-{middle}-{last}]"
        out_dir = Path("loss_surface_results", margs.dataset, unique_name, runoption, typ)
        out_dir.mkdir(exist_ok=True, parents=True)
        filename = f"TE{test_env}_{name}.pth"
        torch.save(results, out_dir / filename)


if __name__ == "__main__":
    import fire
    fire.Fire(run)
