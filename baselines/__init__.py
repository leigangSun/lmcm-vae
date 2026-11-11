"""Baseline causal-discovery methods for comparison with LMCM-VAE.

Each baseline exposes a run function with signature run(X, Y, **kwargs)
and returns a dict with keys A_pred, B_pred, dz_pred and cluster_labels.
"""

from baselines.bpc import run_bpc
from baselines.fofc import run_fofc
from baselines.gin import run_gin
from baselines.nocadilac import run_nocadilac

BASELINES = {
    'BPC': run_bpc,
    'FOFC': run_fofc,
    'GIN': run_gin,
    'NOCADILAC': run_nocadilac,
}

REFERENCES = {
    'BPC': 'Silva, R. (2006). Learning the structure of linear latent variable models. JMLR.',
    'FOFC': 'Kummerfeld, E., & Ramsey, J. (2016). Causal clustering for 1-factor measurement models. KDD.',
    'GIN': 'Xie, F., et al. (2024). Generalized independent noise condition for estimating causal structure with latent variables. JMLR.',
    'NOCADILAC': 'Kaltenpoth, D., & Vreeken, J. (2023). Nonlinear causal discovery with latent confounders. ICML.',
}


def run_baseline(name, X, Y, **kwargs):
    """Run the named baseline on (X, Y) and return its result dict."""
    if name not in BASELINES:
        raise ValueError(
            f"Unknown baseline: {name}. Available: {list(BASELINES)}")
    return BASELINES[name](X, Y, **kwargs)
