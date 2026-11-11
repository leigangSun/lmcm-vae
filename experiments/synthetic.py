"""Reproduce the synthetic experiments from the LMCM-VAE paper.

This single script replaces the scattered run_*_experiments scripts with
four subcommands:

    python -m experiments.synthetic benchmark [--dz 5 10 15] [--n 500 1000 2000 5000] [--runs 20]
    python -m experiments.synthetic ablation  [--runs 20]
    python -m experiments.synthetic noise     [--runs 20]
    python -m experiments.synthetic quick     [--epochs 200] [--runs 1]

Results are written to results/ as JSON and saved after every configuration
so a long run is not lost if interrupted; re-running the same command skips
configurations that are already present.

Run as a plain script with `python experiments/synthetic.py ...` from the
repo root; the repo root is added to sys.path automatically.
"""
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lmcm.data import generate_lmcm_data
from lmcm.train import train_lmcm
from lmcm.metrics import compute_metrics

RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = 'cpu'
DX = 10
HIDDEN_DIMS = [128, 64]
LR = 1e-3
BATCH_SIZE = 64
LAMBDA_A = 0.001
LAMBDA_B = 0.0
THRESHOLD = 0.2


def _save(name, data):
    with open(RESULTS_DIR / name, 'w') as f:
        json.dump(data, f, indent=2)


def _load(name, default):
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return default


def _run_one(dz, dx, dy, n, epochs, B_param, lambda_A, lambda_B, seed, noise_scale=1.0):
    X, Y, Z_true, gt = generate_lmcm_data(
        n_samples=n, dx=dx, dy=dy, dz=dz, seed=seed, noise_scale=noise_scale)
    model, _ = train_lmcm(
        X, Y, dz=dz, hidden_dims=HIDDEN_DIMS, B_param=B_param,
        epochs=epochs, lr=LR, batch_size=BATCH_SIZE,
        lambda_A=lambda_A, lambda_B=lambda_B,
        device=DEVICE, verbose=False, seed=seed)
    A_pred, B_pred = model.get_A(), model.get_B()
    return compute_metrics(A_pred, B_pred, gt, threshold=THRESHOLD)


def cmd_benchmark(args):
    """Vary dz and n with the full single-parent model (epochs=300)."""
    results = _load('benchmark_results.json', {})

    print("Benchmark experiment")
    print("=" * 70)
    for dz in args.dz:
        dy = dz * 5
        dz_s, n_key = str(dz), None
        results.setdefault(dz_s, {})
        for n in args.n:
            n_s = str(n)
            if n_s in results[dz_s]:
                print(f"  [SKIP] dz={dz}, n={n} (already done)")
                continue

            print(f"  dz={dz}, n={n}, dy={dy} ...")
            t0 = time.time()
            f1s, ps, rs, los, lcs, wis = [], [], [], [], [], []
            for run in range(args.runs):
                seed = run * 100 + 42
                m = _run_one(dz, DX, dy, n, 300, 'single_parent',
                             LAMBDA_A, LAMBDA_B, seed)
                f1s.append(m['f1'])
                ps.append(m['precision'])
                rs.append(m['recall'])
                los.append(m['LO'])
                lcs.append(m['LC'])
                wis.append(m['WI'])

            summary = {
                'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
                'P_mean': float(np.mean(ps)), 'P_std': float(np.std(ps)),
                'R_mean': float(np.mean(rs)), 'R_std': float(np.std(rs)),
                'LO_mean': float(np.mean(los)), 'LC_mean': float(np.mean(lcs)),
                'WI_mean': float(np.mean(wis)),
            }
            results[dz_s][n_s] = summary
            _save('benchmark_results.json', results)
            elapsed = time.time() - t0
            print(f"    F1={summary['f1_mean']:.3f}+-{summary['f1_std']:.3f}, "
                  f"P={summary['P_mean']:.3f}, R={summary['R_mean']:.3f} ({elapsed:.1f}s)")

    _save('benchmark_results.json', results)
    print("\nSaved to results/benchmark_results.json")


def cmd_ablation(args):
    """Compare model variants (epochs=400)."""
    dz, dx, dy, n = 5, 10, 25, 2000
    variants = [
        {'name': 'Full model (single_parent + sparse A)',
         'B_param': 'single_parent', 'lambda_A': 0.001, 'lambda_B': 0.0},
        {'name': 'Unconstrained B + L1 sparse',
         'B_param': 'unconstrained', 'lambda_A': 0.005, 'lambda_B': 0.005},
        {'name': 'No sparse regularization on A',
         'B_param': 'single_parent', 'lambda_A': 0.0, 'lambda_B': 0.0},
        {'name': 'No structural constraint (unconstrained + no sparse)',
         'B_param': 'unconstrained', 'lambda_A': 0.0, 'lambda_B': 0.0},
    ]

    results = _load('ablation_results.json', {})

    print("Ablation experiment")
    print("=" * 70)
    for variant in variants:
        name = variant['name']
        if name in results:
            print(f"  [SKIP] {name} (already done)")
            continue

        print(f"  {name} ...")
        t0 = time.time()
        f1s, wis = [], []
        for run in range(args.runs):
            seed = run * 100 + 42
            m = _run_one(dz, dx, dy, n, 400, variant['B_param'],
                         variant['lambda_A'], variant['lambda_B'], seed)
            f1s.append(m['f1'])
            wis.append(m['WI'])

        summary = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'WI_mean': float(np.mean(wis)), 'WI_std': float(np.std(wis)),
        }
        results[name] = summary
        _save('ablation_results.json', results)
        elapsed = time.time() - t0
        print(f"    F1={summary['f1_mean']:.4f}+-{summary['f1_std']:.4f}, "
              f"WI={summary['WI_mean']:.4f} ({elapsed:.1f}s)")

    _save('ablation_results.json', results)
    print("\nSaved to results/ablation_results.json")


def cmd_noise(args):
    """Vary the observation noise level (epochs=400)."""
    dz, dx, dy, n = 5, 10, 25, 2000
    noise_levels = [0.3, 0.5, 1.0, 1.5, 2.0]

    results = _load('noise_robustness_results.json', {})

    print("Noise robustness experiment")
    print("=" * 70)
    for noise in noise_levels:
        ns = str(noise)
        if ns in results:
            print(f"  [SKIP] noise={noise} (already done)")
            continue

        print(f"  noise_scale={noise} ...")
        t0 = time.time()
        f1s = []
        for run in range(args.runs):
            seed = run * 100 + 42
            m = _run_one(dz, dx, dy, n, 400, 'single_parent',
                         LAMBDA_A, LAMBDA_B, seed, noise_scale=noise)
            f1s.append(m['f1'])

        summary = {'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s))}
        results[ns] = summary
        _save('noise_robustness_results.json', results)
        elapsed = time.time() - t0
        print(f"    F1={summary['f1_mean']:.4f}+-{summary['f1_std']:.4f} ({elapsed:.1f}s)")

    _save('noise_robustness_results.json', results)
    print("\nSaved to results/noise_robustness_results.json")


def cmd_quick(args):
    """Single fast sanity run that trains one model and writes a JSON."""
    dz, dx, dy, n = args.dz, args.dx, args.dy, args.n
    seed = 0

    print(f"Quick sanity run: dz={dz}, dx={dx}, dy={dy}, n={n}, epochs={args.epochs}")
    t0 = time.time()
    m = _run_one(dz, dx, dy, n, args.epochs, 'single_parent',
                 LAMBDA_A, LAMBDA_B, seed)
    elapsed = time.time() - t0

    result = {k: float(v) for k, v in m.items()}
    result['config'] = {
        'dz': dz, 'dx': dx, 'dy': dy, 'n': n,
        'epochs': args.epochs, 'hidden_dims': HIDDEN_DIMS,
        'B_param': 'single_parent', 'lambda_A': LAMBDA_A,
        'lambda_B': LAMBDA_B, 'seed': seed, 'elapsed': round(elapsed, 1),
    }
    _save('synthetic_quick_results.json', result)

    print(f"  F1={m['f1']:.4f}, WI={m['WI']:.4f}, "
          f"LO={m['LO']:.4f}, LC={m['LC']:.4f} ({elapsed:.1f}s)")
    print("Saved to results/synthetic_quick_results.json")


def build_parser():
    parser = argparse.ArgumentParser(description='LMCM-VAE synthetic experiments')
    sub = parser.add_subparsers(dest='command', required=True)

    p_bench = sub.add_parser('benchmark', help='sweep over dz and n')
    p_bench.add_argument('--dz', type=int, nargs='+', default=[5, 10, 15])
    p_bench.add_argument('--n', type=int, nargs='+', default=[500, 1000, 2000, 5000])
    p_bench.add_argument('--runs', type=int, default=20)
    p_bench.set_defaults(func=cmd_benchmark)

    p_abl = sub.add_parser('ablation', help='ablation over model variants')
    p_abl.add_argument('--runs', type=int, default=20)
    p_abl.set_defaults(func=cmd_ablation)

    p_noise = sub.add_parser('noise', help='noise robustness sweep')
    p_noise.add_argument('--runs', type=int, default=20)
    p_noise.set_defaults(func=cmd_noise)

    p_quick = sub.add_parser('quick', help='single fast sanity run')
    p_quick.add_argument('--dz', type=int, default=3)
    p_quick.add_argument('--dx', type=int, default=10)
    p_quick.add_argument('--dy', type=int, default=15)
    p_quick.add_argument('--n', type=int, default=1000)
    p_quick.add_argument('--epochs', type=int, default=200)
    p_quick.add_argument('--runs', type=int, default=1)
    p_quick.set_defaults(func=cmd_quick)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
