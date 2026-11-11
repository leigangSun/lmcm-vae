"""BPC (Bayesian PC) baseline.

Reference: Silva, R. (2006). Learning the structure of linear latent
variable models. JMLR.

A constraint-based method that learns latent structure from conditional
independence and tetrad constraints. The returned A matrix is empty
because BPC operates on the observed variables only.
"""

import numpy as np
from itertools import combinations
from scipy import stats


def partial_correlation_test(data, x, y, z, alpha=0.05):
    """Return the p-value of the partial correlation between x and y given z."""
    n = data.shape[0]
    p = len(z)

    if p == 0:
        corr = np.corrcoef(data[:, x], data[:, y])[0, 1]
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr ** 2 + 1e-10)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        idx = [x, y] + list(z)
        sub_data = data[:, idx]
        corr_matrix = np.corrcoef(sub_data.T)

        try:
            inv_corr = np.linalg.inv(corr_matrix)
            partial_corr = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1])
            df = n - p - 2
            t_stat = partial_corr * np.sqrt(df) / np.sqrt(1 - partial_corr ** 2 + 1e-10)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        except np.linalg.LinAlgError:
            p_value = 1.0

    return p_value


def tetrad_test(data, i, j, k, l, alpha=0.05):
    """Test whether the tetrad tau_{ij,kl} is indistinguishable from zero."""
    corr_matrix = np.corrcoef(data.T)
    n = data.shape[0]

    tetrad = (corr_matrix[i, k] * corr_matrix[j, l]
              - corr_matrix[i, l] * corr_matrix[j, k])

    se = np.sqrt(
        (1 - corr_matrix[i, k] ** 2) * (1 - corr_matrix[j, l] ** 2) / n
        + (1 - corr_matrix[i, l] ** 2) * (1 - corr_matrix[j, k] ** 2) / n
        + 2 * (corr_matrix[i, j] * corr_matrix[k, l]
               + corr_matrix[i, l] * corr_matrix[j, k] * corr_matrix[i, k] * corr_matrix[j, l]) / n
    )

    z_stat = tetrad / (se + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return p_value >= alpha


def bpc_algorithm(data, alpha=0.05, max_cond_set_size=3):
    """BPC main routine: returns clusters and the implied B matrix."""
    n_samples, n_vars = data.shape
    var_set = list(range(n_vars))

    adj_matrix = np.ones((n_vars, n_vars), dtype=bool)
    np.fill_diagonal(adj_matrix, False)
    sep_sets = {}

    for cond_size in range(max_cond_set_size + 1):
        edges_to_remove = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if not adj_matrix[i, j]:
                    continue

                neighbors_i = [x for x in range(n_vars) if adj_matrix[i, x] and x != j]
                if len(neighbors_i) >= cond_size:
                    for z_set in combinations(neighbors_i, cond_size):
                        p_val = partial_correlation_test(data, i, j, list(z_set), alpha)
                        if p_val >= alpha:
                            edges_to_remove.append((i, j, list(z_set)))
                            break

        for i, j, z in edges_to_remove:
            adj_matrix[i, j] = False
            adj_matrix[j, i] = False
            sep_sets[(i, j)] = z
            sep_sets[(j, i)] = z

    correlation_matrix = np.corrcoef(data.T)

    clusters = []
    assigned = set()

    for i in range(n_vars):
        if i in assigned:
            continue
        cluster = [i]
        for j in range(i + 1, n_vars):
            if j in assigned:
                continue

            is_same_cluster = True
            for k in cluster:
                for l in range(n_vars):
                    if l in assigned or l == j or l in cluster:
                        continue
                    if not tetrad_test(data, k, j, i, l, alpha):
                        is_same_cluster = False
                        break
                if not is_same_cluster:
                    break

            if is_same_cluster:
                cluster.append(j)

        for v in cluster:
            assigned.add(v)
        clusters.append(cluster)

    dz_pred = len(clusters)
    dy = n_vars

    B_pred = np.zeros((dz_pred, dy))
    cluster_labels = np.zeros(dy, dtype=int)

    for z_idx, cluster in enumerate(clusters):
        for var_idx in cluster:
            B_pred[z_idx, var_idx] = 1.0
            cluster_labels[var_idx] = z_idx

    A_pred = np.zeros((0, dz_pred))

    return {
        'A': A_pred,
        'B': B_pred,
        'cluster_labels': cluster_labels,
        'dz': dz_pred,
        'clusters': clusters,
    }


def run_bpc(X, Y, alpha=0.05, **kwargs):
    """Run BPC on Y and return a result dict compatible with compute_metrics."""
    result = bpc_algorithm(Y, alpha=alpha)

    dz_pred = result['dz']
    dx = X.shape[1] if X is not None else 0

    A_pred = np.zeros((dx, dz_pred))
    B_pred = result['B']

    return {
        'A_pred': A_pred,
        'B_pred': B_pred,
        'dz_pred': dz_pred,
        'cluster_labels': result['cluster_labels'],
        'method': 'BPC',
    }
