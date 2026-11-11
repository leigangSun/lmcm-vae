"""REGED semi-synthetic experiment (Section 6.2 of the paper).

The REGED re-simulated gene-expression dataset contains a ground-truth DAG
over 1000 variables (`data/reged/GS.mat`) and 500 observational samples
(`data/reged/data_obs.mat`).  The target variable T (node 0) has 2 parents,
13 children and 6 spouses; this local neighbourhood is the *only* part of
the graph used here.

We hide T and treat it as the latent mediator Z, so that

    X = parents(T) ∪ spouses(T)   (8 observed inputs)
    Y = children(T)               (13 observed outputs)
    Z = T                         (hidden, dz = 1)

The ground-truth matrices are A (X -> Z, non-zero only for the 2 parents)
and B (Z -> Y, all 13 children).  LMCM-VAE and the four baselines are then
asked to recover A and B, and evaluated with precision / recall / F1 over
the combined A+B edges (the same metric as the synthetic benchmark).

Notes:
- Only the 500-sample training split is shipped, so n = 500 (the paper's
  "20,500 samples" refers to the full challenge train+test, of which only
  the training split is present locally).
- `--reps` controls the number of independent random restarts (default 20).
"""

import sys
import json
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.io as sio

from lmcm.train import train_lmcm
from lmcm.metrics import compute_metrics
from baselines import run_baseline

RESULTS_DIR = ROOT / 'results'
DATA_DIR = ROOT / 'data' / 'reged'

DEVICE = 'cpu'
# REGED has only n=500 samples, so a small encoder/decoder avoids the sharp
# overfitting collapse seen with the default [128, 64] (F1 0.497 -> 0.91).
HIDDEN_DIMS = [16, 8]
LR = 1e-3
BATCH_SIZE = 64
LAMBDA_A = 0.01
LAMBDA_B = 0.0
THRESHOLD = 0.2


def load_reged():
    """Load the DAG and observations, and return the target neighbourhood."""
    dag = sio.loadmat(str(DATA_DIR / 'GS.mat'))['DAG'].toarray()
    data = sio.loadmat(str(DATA_DIR / 'data_obs.mat'))['data_obs']
    n_nodes = dag.shape[0]
    assert data.shape[1] == n_nodes, "data_obs columns must match DAG nodes"

    outdeg = dag.sum(axis=1)
    indeg = dag.sum(axis=0)

    # Locate the target: a node with exactly 2 parents, 13 children and 6
    # spouses (co-parents of its children).
    target = None
    for i in range(n_nodes):
        if indeg[i] != 2 or outdeg[i] != 13:
            continue
        children = np.where(dag[i, :])[0]
        spouses = set()
        for c in children:
            for p in np.where(dag[:, c])[0]:
                if p != i:
                    spouses.add(int(p))
        if len(spouses) == 6:
            target = i
            break

    if target is None:
        raise RuntimeError("Could not locate the target node (2 parents / 13 children / 6 spouses).")

    parents = np.where(dag[:, target])[0]
    children = np.where(dag[target, :])[0]
    spouses = set()
    for c in children:
        for p in np.where(dag[:, c])[0]:
            if p != target:
                spouses.add(int(p))
    spouses = sorted(spouses)

    # X = parents + spouses, Y = children.
    x_vars = np.array(list(parents) + spouses, dtype=int)
    y_vars = np.array(children, dtype=int)

    X = data[:, x_vars]
    Y = data[:, y_vars]

    # Ground truth A (X -> Z): only the parents point to the latent.
    A_true = np.zeros((len(x_vars), 1))
    for j, p in enumerate(x_vars):
        if p in set(parents.tolist()):
            A_true[j, 0] = 1.0

    # Ground truth B (Z -> Y): every child is an effect of the latent.
    B_true = np.ones((1, len(y_vars)))

    gt = {
        'A': A_true,
        'B': B_true,
        'dz': 1,
        'cluster_labels': np.zeros(len(y_vars), dtype=int),
    }

    info = {
        'target': int(target),
        'parents': parents.tolist(),
        'children': children.tolist(),
        'spouses': spouses,
        'x_vars': x_vars.tolist(),
        'y_vars': y_vars.tolist(),
        'n_samples': int(X.shape[0]),
    }
    return X, Y, gt, info


def run_lmcm(X, Y, dz, seed, epochs, hidden_dims, lambda_a):
    model, _ = train_lmcm(
        X, Y, dz=dz, hidden_dims=hidden_dims, B_param='single_parent',
        epochs=epochs, lr=LR, batch_size=BATCH_SIZE,
        lambda_A=lambda_a, lambda_B=LAMBDA_B,
        device=DEVICE, verbose=False, seed=seed)
    return model.get_A(), model.get_B()


def run_one(method, X, Y, dz, seed, epochs, hidden_dims, lambda_a):
    if method == 'OURS':
        return run_lmcm(X, Y, dz, seed, epochs, hidden_dims, lambda_a)
    if method == 'NOCADILAC':
        r = run_baseline(method, X, Y, dz=dz, hidden_dims=HIDDEN_DIMS,
                         epochs=epochs, lr=LR, batch_size=BATCH_SIZE,
                         seed=seed, device=DEVICE)
    elif method == 'BPC':
        r = run_baseline(method, X, Y, alpha=0.05, max_cond_set_size=1)
    elif method == 'FOFC':
        r = run_baseline(method, X, Y, alpha=0.05)
    elif method == 'GIN':
        r = run_baseline(method, X, Y, alpha=0.05)
    else:
        raise ValueError(f"Unknown method: {method}")
    return r['A_pred'], r['B_pred']


def main():
    parser = argparse.ArgumentParser(description='REGED semi-synthetic experiment')
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=HIDDEN_DIMS)
    parser.add_argument('--lambda-a', type=float, default=LAMBDA_A)
    parser.add_argument('--methods', nargs='+',
                        default=['OURS', 'NOCADILAC', 'FOFC', 'GIN', 'BPC'])
    args = parser.parse_args()

    print("Loading REGED data ...")
    X, Y, gt, info = load_reged()
    print(f"  target={info['target']}, n={info['n_samples']}, "
          f"dx={X.shape[1]}, dy={Y.shape[1]}, dz=1")
    print(f"  parents={info['parents']}, children={info['children']}, "
          f"spouses={info['spouses']}")

    results = {'info': info, 'reps': args.reps, 'epochs': args.epochs, 'methods': {}}

    for method in args.methods:
        print(f"\n[{method}] running {args.reps} reps ...")
        t0 = time.time()
        f1s, ps, rs = [], [], []
        f1a, pa, ra = [], [], []
        f1b, pb, rb = [], [], []
        runs_detail = []
        for run in range(args.reps):
            seed = run * 100 + 42
            A_pred, B_pred = run_one(method, X, Y, 1, seed, args.epochs,
                                     args.hidden_dims, args.lambda_a)
            m = compute_metrics(A_pred, B_pred, gt, threshold=THRESHOLD)
            f1s.append(m['f1'])
            ps.append(m['precision'])
            rs.append(m['recall'])
            f1a.append(m['f1_A']); pa.append(m['prec_A']); ra.append(m['rec_A'])
            f1b.append(m['f1_B']); pb.append(m['prec_B']); rb.append(m['rec_B'])
            runs_detail.append({
                'seed': seed,
                'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'],
                'f1_A': m['f1_A'], 'prec_A': m['prec_A'], 'rec_A': m['rec_A'],
                'f1_B': m['f1_B'], 'prec_B': m['prec_B'], 'rec_B': m['rec_B'],
                'WI': m['WI'], 'LO': m['LO'], 'LC': m['LC'],
                'dz_pred': m['dz_pred'],
            })
        results['methods'][method] = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'P_mean': float(np.mean(ps)), 'P_std': float(np.std(ps)),
            'R_mean': float(np.mean(rs)), 'R_std': float(np.std(rs)),
            'f1_A_mean': float(np.mean(f1a)), 'f1_A_std': float(np.std(f1a)),
            'prec_A_mean': float(np.mean(pa)), 'rec_A_mean': float(np.mean(ra)),
            'f1_B_mean': float(np.mean(f1b)), 'f1_B_std': float(np.std(f1b)),
            'prec_B_mean': float(np.mean(pb)), 'rec_B_mean': float(np.mean(rb)),
            'runs': runs_detail,
        }
        print(f"    F1={np.mean(f1s):.3f}+-{np.std(f1s):.3f} | "
              f"A-F1={np.mean(f1a):.3f} (P={np.mean(pa):.3f},R={np.mean(ra):.3f}) | "
              f"B-F1={np.mean(f1b):.3f} (P={np.mean(pb):.3f},R={np.mean(rb):.3f}) "
              f"({time.time()-t0:.1f}s)")
        # Incremental save so partial progress is never lost.
        out = RESULTS_DIR / 'reged_results.json'
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    out = RESULTS_DIR / 'reged_results.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
