"""GIN (Generalized Independent Noise) baseline.

Reference: Xie, F., et al. (2024). Generalized independent noise condition
for estimating causal structure with latent variables. JMLR.

This wrapper calls the official GIN implementation shipped with the
`causal-learn` library (`causallearn.search.HiddenCausal.GIN.GIN.GIN_MI`).
The recovered latent clusters (groups of observed children sharing a latent
parent) are converted into the (dz x dy) matrix B.  As in the paper, GIN
cannot recover the X -> Z edges, so the returned A matrix is empty.

A fast correlation-based clustering is kept only as an *explicit opt-in*
(`use_fast=True`) for exploratory use on very large problems; it is NOT used
by default and it is not the method reported in the paper.
"""

import numpy as np


def _clusters_to_result(clusters, dx, dy, method='GIN'):
    """Convert a list of clusters (lists of Y indices) into B_pred."""
    if not clusters:
        clusters = [list(range(dy))]
    dz_pred = len(clusters)

    B_pred = np.zeros((dz_pred, dy))
    cluster_labels = np.zeros(dy, dtype=int)

    for i, cluster in enumerate(clusters):
        for var_idx in cluster:
            if 0 <= var_idx < dy:
                B_pred[i, var_idx] = 1.0
                cluster_labels[var_idx] = i

    # Assign any variable not covered by the clusters to cluster 0.
    assigned = set(v for c in clusters for v in c)
    for v in range(dy):
        if v not in assigned:
            B_pred[0, v] = 1.0
            cluster_labels[v] = 0

    A_pred = np.zeros((dx, dz_pred))

    return {
        'A_pred': A_pred,
        'B_pred': B_pred,
        'dz_pred': dz_pred,
        'cluster_labels': cluster_labels,
        'method': method,
        'fast_mode': False,
    }


def _fast_gin_cluster(data, alpha=0.05):
    """Correlation-pattern clustering.  Kept only as an explicit fallback."""
    n_vars = data.shape[1]
    corr = np.corrcoef(data.T)
    similarity = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            corr_i = corr[i, :].copy()
            corr_j = corr[j, :].copy()
            corr_i[i] = 0
            corr_i[j] = 0
            corr_j[i] = 0
            corr_j[j] = 0
            norm_i = np.linalg.norm(corr_i)
            norm_j = np.linalg.norm(corr_j)
            if norm_i > 1e-10 and norm_j > 1e-10:
                similarity[i, j] = np.dot(corr_i, corr_j) / (norm_i * norm_j)
            similarity[j, i] = similarity[i, j]

    threshold_sim = 0.3
    clusters = []
    assigned = set()
    for i in range(n_vars):
        if i in assigned:
            continue
        cluster = [i]
        for j in range(i + 1, n_vars):
            if j in assigned:
                continue
            if similarity[i, j] > threshold_sim and abs(corr[i, j]) > alpha:
                cluster.append(j)
        for v in cluster:
            assigned.add(v)
        clusters.append(cluster)
    return clusters


def run_gin(X, Y, alpha=0.05, use_fast=False, **kwargs):
    """Run GIN via causal-learn and return A_pred / B_pred.

    Parameters
    ----------
    X : ndarray (n, dx) -- ignored (GIN cannot recover X -> Z edges).
    Y : ndarray (n, dy) -- the observed variables whose latent parents are
        to be clustered.
    alpha : float -- significance level (kept for interface compatibility).
    use_fast : bool -- if True, use the fast correlation approximation
        instead of causal-learn.  Default False (use the real GIN).
    """
    data = np.asarray(Y, dtype=float)
    dx = X.shape[1] if X is not None else 0
    dy = data.shape[1]

    if not use_fast:
        try:
            from causallearn.search.HiddenCausal.GIN.GIN import GIN_MI
        except ImportError as exc:
            print(f"causal-learn not available ({exc}); falling back to fast mode.")
            clusters = _fast_gin_cluster(data, alpha=alpha)
            res = _clusters_to_result(clusters, dx, dy)
            res['fast_mode'] = True
            res['warning'] = 'causal-learn unavailable; used fast approximation'
            return res

        G, causal_order = GIN_MI(data)
        # causal_order is a list of clusters (lists of Y indices).
        clusters = [list(c) for c in causal_order]
        res = _clusters_to_result(clusters, dx, dy)
        res['causal_order'] = [list(c) for c in causal_order]
        return res

    clusters = _fast_gin_cluster(data, alpha=alpha)
    res = _clusters_to_result(clusters, dx, dy)
    res['fast_mode'] = True
    return res
