# SWAD: Domain Generalization by Seeking Flat Minima (NeurIPS'21)

Official PyTorch implementation of [SWAD: Domain Generalization by Seeking Flat Minima](https://arxiv.org/abs/2102.08604).

Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, Sungrae Park.

<p align="center">
    <img src="./assets/method.png" width="90%" />
</p>

Note that this project is built upon [DomainBed@3fe9d7](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414).

<p align="center">
    <img src="./assets/fig1.png" width="90%" />
</p>


## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

### Environments

Environment details used for our study.

```
Python: 3.8.6
PyTorch: 1.7.0+cu92
Torchvision: 0.8.1+cu92
CUDA: 9.2
CUDNN: 7603
NumPy: 1.19.4
PIL: 8.0.1
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py exp_name --dataset PACS --data_dir /my/datasets/path
```

Experiment results are reported as a table. In the table, the row `SWAD` indicates out-of-domain accuracy from SWAD.
The row `SWAD (inD)` indicates in-domain validation accuracy.

Example results:
```
+------------+--------------+---------+---------+---------+---------+
| Selection  | art_painting | cartoon |  photo  |  sketch |   Avg.  |
+------------+--------------+---------+---------+---------+---------+
|   oracle   |   82.245%    | 85.661% | 97.530% | 83.461% | 87.224% |
|    iid     |   87.919%    | 78.891% | 96.482% | 78.435% | 85.432% |
|    last    |   82.306%    | 81.823% | 95.135% | 82.061% | 85.331% |
| last (inD) |   95.807%    | 95.291% | 96.306% | 95.477% | 95.720% |
| iid (inD)  |   97.275%    | 96.619% | 96.696% | 97.253% | 96.961% |
|    SWAD    |   89.750%    | 82.942% | 97.979% | 81.870% | 88.135% |
| SWAD (inD) |   97.713%    | 97.649% | 97.316% | 98.074% | 97.688% |
+------------+--------------+---------+---------+---------+---------+
```
In this example, the DG performance of SWAD for PACS dataset is 88.135%.

If you set `indomain_test` option to `True`, the validation set is splitted to validation and test sets,
and the `(inD)` keys become to indicate in-domain test accuracy.


### Reproduce the results of the paper

We provide the instructions to reproduce the main results of the paper, Table 1 and 2.
Note that the difference in a detailed environment or uncontrolled randomness may bring a little different result from the paper.

- PACS

```
python train_all.py PACS0 --dataset PACS --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py PACS1 --dataset PACS --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py PACS2 --dataset PACS --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- VLCS

```
python train_all.py VLCS0 --dataset VLCS --deterministic --trial_seed 0 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
python train_all.py VLCS1 --dataset VLCS --deterministic --trial_seed 1 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
python train_all.py VLCS2 --dataset VLCS --deterministic --trial_seed 2 --checkpoint_freq 50 --tolerance_ratio 0.2 --data_dir /my/datasets/path
```

- OfficeHome

```
python train_all.py OH0 --dataset OfficeHome --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py OH1 --dataset OfficeHome --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py OH2 --dataset OfficeHome --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- TerraIncognita

```
python train_all.py TR0 --dataset TerraIncognita --deterministic --trial_seed 0 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR1 --dataset TerraIncognita --deterministic --trial_seed 1 --checkpoint_freq 100 --data_dir /my/datasets/path
python train_all.py TR2 --dataset TerraIncognita --deterministic --trial_seed 2 --checkpoint_freq 100 --data_dir /my/datasets/path
```

- DomainNet

```
python train_all.py DN0 --dataset DomainNet --deterministic --trial_seed 0 --checkpoint_freq 500 --data_dir /my/datasets/path
python train_all.py DN1 --dataset DomainNet --deterministic --trial_seed 1 --checkpoint_freq 500 --data_dir /my/datasets/path
python train_all.py DN2 --dataset DomainNet --deterministic --trial_seed 2 --checkpoint_freq 500 --data_dir /my/datasets/path
```


## Main Results

<p align="center">
    <img src="./assets/fig2.png" width="80%" />
</p>


## Citation

The paper will be published at NeurIPS 2021.

```
@inproceedings{cha2021swad,
  title={SWAD: Domain Generalization by Seeking Flat Minima},
  author={Cha, Junbum and Chun, Sanghyuk and Lee, Kyungjae and Cho, Han-Cheol and Park, Seunghyun and Lee, Yunsung and Park, Sungrae},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```

## License

This source code is released under the MIT license, included [here](./LICENSE).
