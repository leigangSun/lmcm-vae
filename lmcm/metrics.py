"""Evaluation metrics for recovered latent mediator structures.

We report latent-structure errors (LO, LC, WI) and edge-recovery scores
(precision / recall / F1) for the two causal matrices A (X -> Z) and
B (Z -> Y). The predicted latent clusters are matched to the ground-truth
clusters with the Hungarian algorithm.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_metrics(A_pred, B_pred, gt, threshold=0.3):
    A_true = gt['A']
    B_true = gt['B']
    dz_true = gt['dz']
    cluster_labels_true = gt['cluster_labels']

    dz_pred = B_pred.shape[0]

    # Recover predicted cluster labels from the B matrix and match them
    # to the ground-truth clusters with the Hungarian algorithm.
    B_abs = np.abs(B_pred.T)
    pred_cluster_labels = np.argmax(B_abs, axis=1)

    confusion = np.zeros((dz_true, dz_pred))
    for i in range(dz_true):
        for j in range(dz_pred):
            mask_true = cluster_labels_true == i
            mask_pred = pred_cluster_labels == j
            confusion[i, j] = np.sum(mask_true & mask_pred)

    row_ind, col_ind = linear_sum_assignment(-confusion)

    # LO: fraction of true latents that were not recovered.
    matched_true = set(row_ind)
    lo = (dz_true - len(matched_true)) / dz_true

    # LC: fraction of predicted latents that are spurious / split.
    matched_pred = set(col_ind)
    lc = max(0, dz_pred - len(matched_pred)) / dz_true

    # WI: fraction of Y variables assigned to the wrong cluster.
    pred_to_true = {c: r for r, c in zip(row_ind, col_ind)}
    mapped_pred_labels = np.array([pred_to_true.get(p, -1) for p in pred_cluster_labels])
    wi = np.sum(mapped_pred_labels != cluster_labels_true) / len(cluster_labels_true)

    # Edge recovery for A (X -> Z).
    A_true_binary = (np.abs(A_true) > threshold).astype(int)

    A_pred_matched = np.zeros_like(A_true)
    for r, c in zip(row_ind, col_ind):
        if r < A_true.shape[1] and c < A_pred.shape[1]:
            A_pred_matched[:, r] = A_pred[:, c]

    A_pred_binary = (np.abs(A_pred_matched) > threshold).astype(int)

    tp_a = np.sum((A_pred_binary == 1) & (A_true_binary == 1))
    fp_a = np.sum((A_pred_binary == 1) & (A_true_binary == 0))
    fn_a = np.sum((A_pred_binary == 0) & (A_true_binary == 1))

    prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0
    rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0
    f1_a = 2 * prec_a * rec_a / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0

    # Edge recovery for B (Z -> Y).
    B_true_binary = (np.abs(B_true) > threshold).astype(int)

    B_pred_matched = np.zeros_like(B_true)
    for r, c in zip(row_ind, col_ind):
        if r < B_true.shape[0] and c < B_pred.shape[0]:
            B_pred_matched[r, :] = B_pred[c, :]

    B_pred_binary = (np.abs(B_pred_matched) > threshold).astype(int)

    tp_b = np.sum((B_pred_binary == 1) & (B_true_binary == 1))
    fp_b = np.sum((B_pred_binary == 1) & (B_true_binary == 0))
    fn_b = np.sum((B_pred_binary == 0) & (B_true_binary == 1))

    prec_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0
    rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0
    f1_b = 2 * prec_b * rec_b / (prec_b + rec_b) if (prec_b + rec_b) > 0 else 0

    all_tp = tp_a + tp_b
    all_fp = fp_a + fp_b
    all_fn = fn_a + fn_b

    prec_all = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    rec_all = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1_all = 2 * prec_all * rec_all / (prec_all + rec_all) if (prec_all + rec_all) > 0 else 0

    return {
        'LO': lo,
        'LC': lc,
        'WI': wi,
        'prec_A': prec_a,
        'rec_A': rec_a,
        'f1_A': f1_a,
        'prec_B': prec_b,
        'rec_B': rec_b,
        'f1_B': f1_b,
        'precision': prec_all,
        'recall': rec_all,
        'f1': f1_all,
        'dz_pred': dz_pred,
        'dz_true': dz_true,
    }


def threshold_matrix(M, threshold):
    """Zero out entries of M whose absolute value is below threshold."""
    return np.where(np.abs(M) > threshold, M, 0)
