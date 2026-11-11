"""FOFC (Find One Factor Clusters) baseline.

Reference: Kummerfeld, E., & Ramsey, J. (2016). Causal clustering for
1-factor measurement models. KDD.

Clusters observed variables into single-factor measurement models using
tetrad constraints. Like BPC it returns an empty A matrix.
"""

import numpy as np
from itertools import combinations
from scipy import stats


def compute_tetrad(data, i, j, k, l):
    """Tetrad tau_{ij,kl} = corr(i,k)*corr(j,l) - corr(i,l)*corr(j,k)."""
    corr_matrix = np.corrcoef(data.T)
    return corr_matrix[i, k] * corr_matrix[j, l] - corr_matrix[i, l] * corr_matrix[j, k]


def tetrad_test_fofc(data, i, j, k, l, alpha=0.05):
    """Return (holds, p-value) for the FOFC tetrad test."""
    n = data.shape[0]
    tetrad_val = compute_tetrad(data, i, j, k, l)
    corr_matrix = np.corrcoef(data.T)

    variances = np.array([np.var(data[:, v]) for v in range(data.shape[1])])

    se_sq = (variances[i] * variances[k] * variances[j] * variances[l]
             + corr_matrix[i, j] ** 2 * variances[k] * variances[l]
             + corr_matrix[k, l] ** 2 * variances[i] * variances[j]) / (n * n)
    se = np.sqrt(max(se_sq, 1e-15))

    z_stat = tetrad_val / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return p_value >= alpha, p_value


def find_one_factor_clusters(data, alpha=0.05):
    """Greedily build one-factor clusters from tetrad compatibility."""
    n_samples, n_vars = data.shape

    clusters = []
    assigned = set()

    for seed_var in range(n_vars):
        if seed_var in assigned:
            continue

        cluster = [seed_var]
        candidates = [v for v in range(n_vars) if v not in assigned and v != seed_var]

        for candidate in candidates:
            is_compatible = True

            for cluster_var in cluster:
                for test_var in range(n_vars):
                    if test_var in assigned or test_var in cluster or test_var == candidate:
                        continue

                    holds, _ = tetrad_test_fofc(data, cluster_var, candidate, seed_var, test_var, alpha)
                    if not holds:
                        is_compatible = False
                        break

                if not is_compatible:
                    break

            if is_compatible:
                cluster.append(candidate)

        for v in cluster:
            assigned.add(v)
        clusters.append(cluster)

    return clusters


def fofc_algorithm(data, alpha=0.05):
    """Build the B matrix from the FOFC clusters."""
    n_samples, n_vars = data.shape

    clusters = find_one_factor_clusters(data, alpha)

    dz_pred = len(clusters)

    B_pred = np.zeros((dz_pred, n_vars))
    cluster_labels = np.zeros(n_vars, dtype=int)

    for z_idx, cluster in enumerate(clusters):
        for var_idx in cluster:
            B_pred[z_idx, var_idx] = 1.0
            cluster_labels[var_idx] = z_idx

    return {
        'B': B_pred,
        'cluster_labels': cluster_labels,
        'dz': dz_pred,
        'clusters': clusters,
    }


def run_fofc(X, Y, alpha=0.05, **kwargs):
    """Run FOFC on Y and return a result dict compatible with compute_metrics."""
    result = fofc_algorithm(Y, alpha=alpha)

    dz_pred = result['dz']
    dx = X.shape[1] if X is not None else 0

    A_pred = np.zeros((dx, dz_pred))
    B_pred = result['B']

    return {
        'A_pred': A_pred,
        'B_pred': B_pred,
        'dz_pred': dz_pred,
        'cluster_labels': result['cluster_labels'],
        'method': 'FOFC',
    }
