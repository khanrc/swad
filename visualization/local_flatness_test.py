from pathlib import Path
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from munch import Munch
from domainbed.algorithms import get_algorithm_class
from domainbed.lib import swa_utils
from domainbed.datasets import get_dataset
#  from domainbed import evaluator
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed.evaluator import Evaluator
from domainbed.lib.logger import Logger
from domainbed.lib import misc
from infer_utils import load_algorithms, setup_eval_meta

logger = Logger.get()


def params_to_vector(parameters):
    return torch.cat(list(map(lambda x: x.detach().flatten(), parameters)))


def rand_unit(N):
    random_vector = torch.randn(N)
    random_unit_vector = random_vector / torch.norm(random_vector)
    return random_unit_vector


def compute_avg_model(models):
    avg_model = copy.deepcopy(models[0])
    model_params = zip(*(model.parameters() for model in models))
    for avg_param, model_param_list in zip(avg_model.parameters(), model_params):
        model_param_stack = torch.stack(model_param_list)
        model_param = model_param_stack.mean(0)

        with torch.no_grad():
            avg_param.copy_(model_param)

    # merge range
    ranges = sorted([model.step_range for model in models], key=lambda x: (min(x), max(x)))
    out_ranges = []
    m_range = ranges[0]
    for rng in ranges[1:]:
        if max(m_range) + 1 == min(rng):
            m_range = range(min(m_range), max(rng) + 1)
        else:
            out_ranges.append(m_range)
            m_range = rng
    out_ranges.append(m_range)

    if len(out_ranges) == 1:
        out_ranges = out_ranges[0]

    avg_model.step_range = out_ranges
    return avg_model


def add_flat_params_(flat_params, model):
    offset = 0
    for n, p in model.named_parameters():
        size = p.numel()
        ip = flat_params[offset:offset+size].view(p.shape)
        with torch.no_grad():
            p.add_(ip)
        offset += size


def eval_with_move(algorithm, direction, evaluator, step_size=1., max_dist=50.):
    algo = copy.deepcopy(algorithm)
    # make a unit vector
    direction = direction / torch.norm(direction)
    direction = direction.cuda()
    algo = algo.cuda()

    results = []
    distance = 0.
    while distance <= max_dist:
        accs, summaries, losses, _ = evaluator.evaluate(algo)
        accs = {k + "_acc": v for k, v in accs.items()}
        losses = {k + "_loss": v for k, v in losses.items()}

        results.append({
            **accs,
            **losses,
            "distance": distance
        })

        target_acc = summaries["test_in"]
        logger.info(f"[D {distance:.0f}] Target Acc {target_acc:.2%}")
        distance += step_size
        add_flat_params_(direction*step_size, algo)

    return results


def run(ckpt_dir, test_env, step_size=5., max_dist=60., n_repeat=100):
    work_dir = Path('.')
    data_dir = Path('./data/')

    ckpt_dir = Path(ckpt_dir)
    test_envs = [test_env]
    ckpt_paths = list(ckpt_dir.glob(f"TE{test_env}_*.pth"))
    ckpt_paths = [path for path in ckpt_paths if "iidbest" not in str(path)]
    # load checkpoints
    logger.info("# Load checkpoints ...")
    # TODO setup dataset load phase ...
    algos, dataset, in_splits, out_splits, margs, mhparams = load_algorithms(ckpt_paths, test_envs, data_dir=data_dir)

    n_params = sum([
        p.numel()
        for n, p in algos[0].named_parameters()
    ])

    # setup eval loaders
    eval_meta = setup_eval_meta(dataset, in_splits, out_splits, margs, mhparams)

    evaluator = Evaluator(
        test_envs, eval_meta, len(dataset), logger, evalmode='normal', target_env=test_env
    )

    algorithm = get_algorithm(algos)
    logger.info("Algorithm step range = %s" % algorithm.step_range)

    results = []
    org_algo = copy.deepcopy(algorithm)
    for i in range(n_repeat):
        logger.info("Repeat [%d/%d]" % (i+1, n_repeat))
        direction = rand_unit(n_params)
        res = eval_with_move(
            algorithm, direction, evaluator, step_size=step_size, max_dist=max_dist,
        )
        results.append(res)

        for op, p in zip(org_algo.parameters(), algorithm.parameters()):
            if not torch.allclose(op, p):
                raise ValueError("Sanity check failed")

    # save
    dset_name = margs.dataset
    filename = f"{dset_name}_TE{test_env}_S{step_size:.0f}_MD{max_dist:.0f}_N{n_repeat}"
    outpath = work_dir / "perturb_results" / "move" / filename
    outpath.parent.mkdir(exist_ok=True, parents=True)
    torch.save(results, outpath)
    logger.info(f"Results are saved to `{outpath.absolute()}`")


if __name__ == "__main__":
    import fire
    fire.Fire(run)
