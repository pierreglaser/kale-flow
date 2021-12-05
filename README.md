# Code to reproduce experiments of the NeurIPS 2021 Paper KALE Flow: A Relaxed KL Gradient Flow for Probabilities with Disjoint Support

ArXiv link: https://arxiv.org/pdf/2106.08929.pdf

## Installation

Using `conda`: In a `bash` shell, clone the repository, and once inside it,
execute the following bash commands

### GPU install

```bash
conda create -n kale-flow && conda activate kale-flow
# To run experiments - lookup your own machine specs for exact pytorch and
# cudatoolkit versions.
# mine is: conda install pytorch==1.8.0 cudatoolkit=10.2 -c pytorch
conda install pytorch==<pytorch-version> cudatoolkit=<cudatoolkit-version> -c pytorch
conda install numpy scipy pot conda pillow

# Vendored gradient flow library
pip install --no-deps ./kernel-wasserstein-flows

# Used to compute sinkhorn flows; your mileage may vary
pip install --no-deps geomloss pykeops

# To plot experiment resulst
conda install matplotlib ipympl pandas
```

### CPU install

```bash
conda create -n kale-flow && conda activate kale-flow

# To run experiments
conda install pytorch numpy scipy pot pillow

# Vendored gradient flow library
pip install --no-deps ./kernel-wasserstein-flows

# Used to compute sinkhorn flows; your mileage may vary
python -m pip install --no-deps geomloss pykeops

# To plot experiment resulst
conda install matplotlib ipympl pandas
```


## Cite The paper:

```bibtex
@inproceedings{glaser2021kale,
  author={Pierre Glaser, Michael Arbel, Arthur Gretton},
  title={{KALE Flow: A Relaxed KL Gradient Flow for Probabilities with Disjoint Support}},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021},
}
```
