"""OOD detection metrics.  Convention: label 1 = OOD (positive), score higher = more OOD."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def fpr_at_95_tpr(y_ood: np.ndarray, score: np.ndarray) -> float:
    """Pick the threshold that keeps 95% of ID queries (ID = retained), report the
    fraction of OOD queries that are wrongly accepted at that threshold."""
    id_scores = score[y_ood == 0]
    ood_scores = score[y_ood == 1]
    thr = np.percentile(id_scores, 95)            # 95% of ID have score <= thr -> accepted
    return float(np.mean(ood_scores <= thr))


def detection_metrics(y_ood: np.ndarray, score: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(y_ood, score)),
        "AUPR_OOD": float(average_precision_score(y_ood, score)),
        "AUPR_ID": float(average_precision_score(1 - y_ood, -score)),
        "FPR@95TPR": fpr_at_95_tpr(y_ood, score),
    }


def split_metrics(labels: np.ndarray, score: np.ndarray) -> dict:
    """Metrics for ID vs all-OOD, ID vs Near-OOD, ID vs Far-OOD."""
    out = {}
    is_id = labels == "IN_DISTRIBUTION"
    for name, mask in [("All", np.ones_like(is_id)), ("Near", is_id | (labels == "NEAR_OOD")),
                       ("Far", is_id | (labels == "FAR_OOD"))]:
        y = (~is_id[mask]).astype(int)
        for k, v in detection_metrics(y, score[mask]).items():
            out[f"{name} {k}"] = v
    return out


def roc_points(y_ood: np.ndarray, score: np.ndarray):
    fpr, tpr, _ = roc_curve(y_ood, score)
    return fpr, tpr
