# LMCM-VAE

LMCM-VAE is a method for identifying latent mediator structures from
observational data. It models a three-layer causal chain `X -> Z -> Y`
where `Z` is an unobserved latent mediator, and learns both the latent
variables and the causal structure with a structured variational
autoencoder. The implementation accompanies the paper *"Latent-Mediator
Causal Structure Learning under Linear-Gaussian Models: Identifiability and
Variational Inference"*.
## Project structure

```
lmcm-vae/
├── README.md
├── requirements.txt
├── lmcm/                 # Core implementation
│   ├── __init__.py
│   ├── data.py           # Synthetic LMCM data generator
│   ├── model.py          # Encoder, decoder, VAE, ELBO loss
│   ├── train.py          # Training loop and dz selection
│   └── metrics.py        # LO/LC/WI and precision/recall/F1 metrics
├── baselines/            # Comparison methods
│   ├── __init__.py       # Registry and run_baseline()
│   ├── bpc.py
│   ├── fofc.py
│   ├── gin.py
│   ├── nocadilac.py
│   └── compare.py        # Baseline comparison driver
├── experiments/
│   ├── __init__.py
│   ├── synthetic.py      # benchmark / ablation / noise / quick
│   ├── reged.py          # semi-synthetic REGED (Section 6.2)
│   └── sachs.py          # Sachs case study (Section 6.6)
├── data/
│   ├── download_sachs.py # Fetches the Sachs dataset
│   ├── sachs/            # Sachs protein-signalling data
│   └── reged/            # REGED ground-truth DAG + observations
└── results/              # Raw and summarized outputs of the reported runs
```

## Installation

```bash
pip install -r requirements.txt
```

The GIN baseline requires the `causal-learn` package:

```bash
pip install causal-learn
```

## Minimal usage

```python
from lmcm.data import generate_lmcm_data
from lmcm.train import train_lmcm
from lmcm.metrics import compute_metrics

# Generate synthetic LMCM data.
X, Y, Z_true, gt = generate_lmcm_data(
    n_samples=2000, dx=10, dy=20, dz=5, seed=42
)

# Train the model.
model, history = train_lmcm(
    X, Y, dz=5,
    hidden_dims=[128, 64],
    B_param='single_parent',
    epochs=800,
    lr=1e-3,
    lambda_A=0.001,
    lambda_B=0.0,
    device='cpu',
    verbose=True,
)

# Recover the learned parameters and evaluate.
A_pred = model.get_A()
B_pred = model.get_B()

metrics = compute_metrics(A_pred, B_pred, gt, threshold=0.2)
print(f"F1 Score: {metrics['f1']:.4f}")
```

## Running the experiments

All synthetic experiments live in `experiments/synthetic.py` and are driven
by subcommands. Run them from the repository root.

```bash
# Full benchmark over dz and n (20 random seeds per configuration).
python -m experiments.synthetic benchmark --dz 5 10 15 --n 500 1000 2000 5000 --runs 20

# Ablation study over model variants.
python -m experiments.synthetic ablation --runs 20

# Noise robustness sweep.
python -m experiments.synthetic noise --runs 20
```

Baseline comparisons (`BPC`, `FOFC`, `GIN`, `NOCADILAC`) are run with:

```bash
python -m baselines.compare --dz 5 10 15 --n 500 1000 2000 5000 --methods BPC FOFC GIN NOCADILAC --runs 20
```

The semi-synthetic REGED experiment (Section 6.2) and the Sachs case study
(Section 6.6) are run with:

```bash
# REGED: hide the target variable T (2 parents / 13 children / 6 spouses)
# and recover it as a latent mediator, over independent restarts.
python -m experiments.reged --reps 20 --epochs 200

# Sachs: hide the node Erk and recover its 5 confounded edges.
python -m experiments.sachs --reps 5 --epochs 200
```

Both scripts write their summaries to `results/` .

## Citation

```bibtex
@article{lmcmvae2026,
  title   = {Latent-Mediator Causal Structure Learning under Linear-Gaussian Models: Identifiability and Variational Inference},
  author  = {Sun, Leigang and Ye, Lei and Zeng, Shuo and Yu, Li and Xiang, Xiaojia},
  journal = {arXiv preprint (under review)},
  year    = {2026},
}
```
