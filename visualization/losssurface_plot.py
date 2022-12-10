from pathlib import  Path
import collections
import copy
import re

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import seaborn as sns

DEBUG = False
SHAPE = False

def setup_matplotlib():
    #  matplotlib.rc('text', usetex=True)
    #  matplotlib.rc('text.latex', preamble=[r'\usepackage{sansmath}', r'\sansmath'])
    #  matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['DejaVu Sans']})

    matplotlib.rc('xtick.major', pad=12)
    matplotlib.rc('ytick.major', pad=12)
    matplotlib.rc('grid', linewidth=0.8)

    sns.set_style('white')


class LogNormalize(colors.Normalize):
    """Log normalizer for colormap.
    Without log normalize, colormap will be linear-scale.
    """
    def __init__(self, vmin=None, vmax=None, clip=None, log_alpha=None):
        self.log_alpha = log_alpha
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # shift value minimum from vmin to 0 & apply log.
        log_v = np.ma.log(value - self.vmin)
        # thresholding log_v to log_alpha
        log_v = np.ma.maximum(log_v, self.log_alpha)
        # normalize w/ log_alpha shifting.
        ret = 0.9 * (log_v - self.log_alpha) / (np.log(self.vmax - self.vmin) - self.log_alpha)
        return ret


def auto_logalpha(values, vmin=None, eps=1e-8):
    sv = values.reshape(-1).copy()
    sv.sort()
    topk_value = np.unique(sv)[3]
    if vmin is None:
        vmin = sv[0]
    log_alpha = np.log(topk_value - vmin + eps)
    return log_alpha


def draw_contour(grid, values, vmin=None, vmax=None, log_alpha=None, N=7, cmap='jet_r', alpha=0.55):
    """Draw contour
    Args:
        grid: grid[i, j] = (x, y)
        values: values[i, j] = value of (x, y); i.e., z.
        vmin: value min.
        vmax: value max. If given, upper-clip values by vmax.
        log_alpha: alpha (= e^log_alpha) is the tick range of first contour.
        N: # of ticks
    """
    cmap = plt.get_cmap(cmap)
    if vmax is None:
        clipped = values.copy()
        vmax = clipped.max()
    else:
        clipped = np.minimum(values, vmax)

    if vmin is None:
        vmin = clipped.min()

    if log_alpha is None:
        log_alpha = auto_logalpha(values, vmin)
        print("Auto log_alpha =", log_alpha)

    # log_gamma: log-scale contour tick size
    log_gamma = (np.log(vmax - vmin) - log_alpha) / N
    levels = vmin + np.exp(log_alpha + log_gamma * np.arange(N + 1))
    levels[0] = vmin
    levels[-1] = vmax
    #  levels = np.concatenate((levels, [1e10]))
    norm = LogNormalize(vmin - 1e-8, vmax + 1e-8, log_alpha=log_alpha)
    # plot contour
    #  contour = plt.contour(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
    #                        linewidths=2.5,
    #                        zorder=1,
    #                        levels=levels)
    # plot filled contour
    contourf = plt.contourf(grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, norm=norm,
                            levels=levels,
                            zorder=0,
                            alpha=alpha)
    if not SHAPE:
        colorbar = plt.colorbar(format='%.4f', ticks=levels)
    else:
        colorbar = None
    #  labels = list(colorbar.ax.get_yticklabels())
    #  labels[-1].set_text(r'$>\,$' + labels[-2].get_text())
    #  colorbar.ax.set_yticklabels(labels)
    return colorbar


def draw_losssurface_on_canvas(w_xy, grid, values, vmin, vmax, log_alpha, N, fontsize=14):
    """Draw SWA plane
    Should be prepare canvas before run this function, e.g. plt.figure().
    """
    #  cmap = cm.jet_r  # default
    cmap = cm.RdBu
    alpha = 0.8
    colorbar = draw_contour(
        grid,
        values,
        vmin=vmin,
        vmax=vmax,
        log_alpha=log_alpha,
        N=N,
        cmap=cmap,
        alpha=alpha
    )

    # plot three weight points
    #  if not SHAPE:
    S = 540
    plt.scatter(w_xy[0, 0], w_xy[0, 1], marker='^', c='k', s=S, zorder=2)
    plt.scatter(w_xy[1, 0], w_xy[1, 1], marker='^', c='k', s=S, zorder=2)
    plt.scatter(w_xy[2, 0], w_xy[2, 1], marker='^', c='k', s=S, zorder=2)
    center = w_xy.mean(0)
    plt.scatter(center[0], center[1], marker='P', c='w', edgecolors='k', s=S*1.6, zorder=2)
    #  plt.scatter(center[0], center[1], marker='P', c='mediumseagreen', s=S*1.6, zorder=2)

    plt.margins(0.0)
    if not SHAPE:
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        colorbar.ax.tick_params(labelsize=fontsize)

    plt.xticks([])
    plt.yticks([])


def extract_info(results, value_key):
    w_results = results[:3]
    w_xy = torch.stack([row['grid_xy'] for row in w_results]).numpy()

    results = results[3:]
    G = int(len(results) ** 0.5)
    assert G*G == len(results)

    grid_xy = np.zeros([G, G, 2])
    grid_value = np.zeros([G, G])
    for row in results:
        i, j = row['ij']
        grid_xy[i, j] = row['grid_xy'].tolist()
        grid_value[i, j] = row[value_key]

    return w_xy, grid_xy, grid_value


def draw_and_save_losssurface(results, value_key, outstem, vmin=None, vmax=None, log_alpha=None, N=7):
    w_xy, grid_xy, grid_value = extract_info(results, value_key)
    outpath = str(outstem) + f"_{value_key}_plane.pdf"

    #  fontsize = 14
    #  plt.figure(figsize=(12.4, 7))
    fontsize = 20
    if not SHAPE:
        plt.figure(figsize=(12.4, 9))
    else:
        plt.figure(figsize=(10.4, 9))

    draw_losssurface_on_canvas(w_xy, grid_xy, grid_value, vmin, vmax, log_alpha, N, fontsize)
    plt.savefig(outpath, format=Path(outpath).suffix[1:], bbox_inches='tight')
    plt.close()


def plot_from_path(path, log_alpha=-5.0):
    prefix = ''
    if DEBUG:
        prefix = 'debug_'
    path = Path(path)
    results = torch.load(path)
    outstem = path.parent / (prefix + path.stem)
    N = 7

    print(f"Generating loss and error planes for {path.name} ...")
    draw_and_save_losssurface(results, 'loss', outstem, log_alpha=log_alpha, N=N)


def extract_envinfo_from_path(path):
    chunks = path.stem.split("_")

    def find_chunk(chunks, pattern):
        for chunk in chunks:
            if re.match(pattern, chunk):
                return chunk

    TE = find_chunk(chunks, r"TE\d")
    env = find_chunk(chunks, r"env\d")
    inout = find_chunk(chunks, r"(in|out)")

    if inout is None or path.parent.stem == 'test':
        inout = 'test'

    return TE, env, inout


def integrate_results_from_paths(paths):
    """If multi-env results exists for single test env, integrate it here.
    """
    paths = list(paths)

    if "DomainNet" in str(paths[0]):
        n_envs = 6
    else:
        n_envs = 4

    out_dir = paths[0].parent
    path_dic = collections.defaultdict(list)
    for path in paths:
        te, env, inout = extract_envinfo_from_path(path)
        if te is None or env is None or inout is None:
            continue
        key = "_".join([te, inout])
        path_dic[key].append(path)

    for key, ipaths in path_dic.items():
        if len(ipaths) == 1:
            continue
        print(key + ":")
        agg_results = None
        for path in ipaths:
            print(f"\t{path}")
            results = torch.load(path)
            if not agg_results:
                agg_results = copy.deepcopy(results)
            else:
                for agg_row, row in zip(agg_results[3:], results[3:]):
                    agg_row['error'] += row['error']
                    agg_row['loss'] += row['loss']

        for agg_row in agg_results[3:]:
            agg_row['error'] /= len(ipaths)
            agg_row['loss'] /= len(ipaths)

        out_path = out_dir / (key + ".pth")

        if len(ipaths) != n_envs-1:
            print("Results are not enough -> remove")
            if out_path.exists():
                out_path.unlink()
        else:
            torch.save(agg_results, out_path)
            print(f"Aggregated result is saved into {out_path}")
        print("---")


def run(path, log_alpha=-5.0):
    path = Path(path)
    if path.is_dir():
        paths = list(path.glob("*.pth"))
        integrate_results_from_paths(paths)
        paths = list(path.glob("*.pth"))
    else:
        paths = [path]

    for p in paths:
        plot_from_path(p, log_alpha=log_alpha)


if __name__ == "__main__":
    import fire
    setup_matplotlib()
    fire.Fire(run)
