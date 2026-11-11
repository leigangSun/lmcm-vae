"""Compare LMCM-VAE against the baseline methods on synthetic LMCM data.

Runs each baseline over a grid of (dz, n) settings with multiple random
seeds, evaluates with the same metrics as the main experiments, and writes
a summary plus a detailed dump and a progress file to results/. Each run is
guarded by a timeout so slow constraint-based baselines cannot stall the
loop; results are saved incrementally so an interrupted run can be resumed.
"""
import sys
import os
import json
import time
import threading
from pathlib import Path

# Allow running both as `python -m baselines.compare` and
# `python baselines/compare.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lmcm.data import generate_lmcm_data
from lmcm.metrics import compute_metrics
from baselines import run_baseline

RESULTS_DIR = ROOT / 'results'
PROGRESS_FILE = RESULTS_DIR / 'baseline_progress.json'
RESULTS_FILE = RESULTS_DIR / 'baseline_comparison_results.json'
DETAILED_FILE = RESULTS_DIR / 'baseline_comparison_detailed.json'

RESULTS_DIR.mkdir(exist_ok=True)

DX = 10
DEVICE = 'cpu'

# Per-method timeout (seconds). Constraint-based methods can be very slow
# on large dy, so they get a tighter budget.
DEFAULT_TIMEOUT = 300
SLOW_TIMEOUT = 60


class RunTimeoutError(Exception):
    pass


def run_with_timeout(func, args=(), kwargs=None, timeout=DEFAULT_TIMEOUT):
    """Run func in a thread and raise RunTimeoutError if it overruns."""
    if kwargs is None:
        kwargs = {}
    result = {'value': None, 'error': None}

    def target():
        try:
            result['value'] = func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - propagate to caller
            result['error'] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise RunTimeoutError(f"timeout ({timeout}s)")
    if result['error'] is not None:
        raise result['error']
    return result['value']


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {'completed': [], 'results': {}, 'detailed': {}}


def save_progress(progress):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress, f, indent=2)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress['results'], f, indent=2)
    with open(DETAILED_FILE, 'w', encoding='utf-8') as f:
        json.dump(progress['detailed'], f, indent=2)


def method_kwargs(method, dz, seed):
    if method == 'NOCADILAC':
        return dict(dz=dz, hidden_dims=[128, 64], epochs=300,
                    lr=1e-3, batch_size=64, seed=seed, device=DEVICE)
    if method == 'BPC':
        return dict(alpha=0.05, max_cond_set_size=1)
    if method in ('FOFC', 'GIN'):
        return dict(alpha=0.05)
    return {}


def method_timeout(method):
    if method in ('GIN', 'BPC'):
        return SLOW_TIMEOUT
    return DEFAULT_TIMEOUT


def run_single(method, X, Y, dz, seed):
    r = run_baseline(method, X, Y, **method_kwargs(method, dz, seed))
    return r['A_pred'], r['B_pred']


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Baseline comparison on LMCM data')
    parser.add_argument('--dz', type=int, nargs='+', default=[5, 10, 15])
    parser.add_argument('--n', type=int, nargs='+', default=[500, 1000, 2000, 5000])
    parser.add_argument('--methods', nargs='+',
                        default=['NOCADILAC', 'FOFC', 'GIN', 'BPC'])
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {DEVICE}")
    print(f"Results directory: {RESULTS_DIR}")

    progress = load_progress()
    completed = set(progress.get('completed', []))
    results = progress.get('results', {})
    detailed = progress.get('detailed', {})

    print(f"\nAlready completed: {len(completed)} configurations")

    total = len(args.dz) * len(args.n) * len(args.methods)
    print(f"Total configurations: {total}, remaining: {total - len(completed)}")
    print("=" * 70)

    for dz in args.dz:
        dy = dz * 5
        for n in args.n:
            for method in args.methods:
                config_key = f"{method}_dz{dz}_n{n}"
                if config_key in completed:
                    print(f"  [SKIP] {config_key} (already done)")
                    continue

                print(f"\n  [{config_key}] dz={dz}, n={n}, dy={dy} ...")
                t0 = time.time()

                f1s, ps, rs, los, lcs, wis = [], [], [], [], [], []
                runs_detail = []
                n_success = 0
                error_msg = None

                for run in range(args.runs):
                    seed = run * 100 + 42
                    try:
                        X, Y, Z_true, gt = generate_lmcm_data(
                            n_samples=n, dx=DX, dy=dy, dz=dz, seed=seed)
                        A_pred, B_pred = run_with_timeout(
                            run_single, args=(method, X, Y, dz, seed),
                            timeout=method_timeout(method))
                        m = compute_metrics(A_pred, B_pred, gt, threshold=0.2)
                        f1s.append(m['f1'])
                        ps.append(m['precision'])
                        rs.append(m['recall'])
                        los.append(m['LO'])
                        lcs.append(m['LC'])
                        wis.append(m['WI'])
                        runs_detail.append({
                            'seed': seed,
                            'metrics': {k: float(v) for k, v in m.items()},
                        })
                        n_success += 1
                    except RunTimeoutError as e:
                        err = f"run{run}(seed={seed}): TIMEOUT: {e}"
                        runs_detail.append({'seed': seed, 'error': err})
                        error_msg = error_msg or err
                        print(f"    [run {run} timeout] {err}")
                    except Exception as e:  # noqa: BLE001
                        err = f"run{run}(seed={seed}): {type(e).__name__}: {str(e)[:200]}"
                        runs_detail.append({'seed': seed, 'error': err})
                        error_msg = error_msg or err
                        print(f"    [run {run} failed] {err}")

                elapsed = time.time() - t0

                if n_success > 0:
                    summary = {
                        'f1_mean': float(np.mean(f1s)),
                        'f1_std': float(np.std(f1s)),
                        'P_mean': float(np.mean(ps)),
                        'P_std': float(np.std(ps)),
                        'R_mean': float(np.mean(rs)),
                        'R_std': float(np.std(rs)),
                        'LO_mean': float(np.mean(los)),
                        'LC_mean': float(np.mean(lcs)),
                        'WI_mean': float(np.mean(wis)),
                        'n_success': n_success,
                        'n_runs': args.runs,
                        'elapsed': round(elapsed, 1),
                    }
                else:
                    summary = {
                        'f1_mean': 0.0, 'f1_std': 0.0,
                        'P_mean': 0.0, 'P_std': 0.0,
                        'R_mean': 0.0, 'R_std': 0.0,
                        'LO_mean': 1.0, 'LC_mean': 1.0, 'WI_mean': 1.0,
                        'n_success': 0,
                        'n_runs': args.runs,
                        'elapsed': round(elapsed, 1),
                        'error': error_msg or 'all runs failed',
                    }

                results.setdefault(str(dz), {}).setdefault(str(n), {})[method] = summary
                detailed.setdefault(str(dz), {}).setdefault(str(n), {})[method] = {
                    'summary': summary,
                    'runs': runs_detail,
                    'config': {
                        'dz': dz, 'dx': DX, 'dy': dy, 'n': n,
                        'method': method, 'n_runs': args.runs,
                    },
                }

                completed.add(config_key)
                progress['completed'] = sorted(list(completed))
                progress['results'] = results
                progress['detailed'] = detailed
                progress['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                save_progress(progress)

                if n_success > 0:
                    print(f"    F1={summary['f1_mean']:.3f}+-{summary['f1_std']:.3f}, "
                          f"P={summary['P_mean']:.3f}, R={summary['R_mean']:.3f}, "
                          f"LO={summary['LO_mean']:.3f}, LC={summary['LC_mean']:.3f}, "
                          f"WI={summary['WI_mean']:.3f} "
                          f"({n_success}/{args.runs} ok, {elapsed:.1f}s)")
                else:
                    print(f"    all runs failed: {error_msg}")

    print("\n" + "=" * 70)
    print("Baseline comparison finished.")
    print("=" * 70)
    print(f"\nResults written to:\n  {RESULTS_FILE}\n  {DETAILED_FILE}\n  {PROGRESS_FILE}")


if __name__ == '__main__':
    main()
