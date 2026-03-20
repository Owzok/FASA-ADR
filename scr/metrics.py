import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, precision_score, 
    recall_score, matthews_corrcoef
)

# BETA 1.0 = F1-score
# ---------------------------------------------------------------------
# Threshold selection (Fβ)
# ---------------------------------------------------------------------
def find_best_threshold(y_true, y_pred, beta=1.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

    precision = precision[:-1]
    recall = recall[:-1]

    b2 = beta**2

    with np.errstate(divide='ignore', invalid='ignore'):
        fbeta = (1 + b2) * (precision * recall) / (b2 * precision + recall)
        fbeta = np.nan_to_num(fbeta)

    best_idx = int(np.argmax(fbeta))
    best_thr = thresholds[best_idx]

    return {
        "threshold": float(best_thr),
        f"fbeta": float(fbeta[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx])
    }

# ---------------------------------------------------------------------
# Mean Rank (MR)
# ---------------------------------------------------------------------
def mean_rank(scores, truths):
    n_drug = scores.shape[0]
    ranks = []

    for i in range(n_drug):
        relevant = np.where(truths[i] == 1)[0]
        if relevant.size == 0:
            continue

        order = np.argsort(-scores[i])
        pos = np.where(np.isin(order, relevant))[0]
        ranks.append(float(pos[0] + 1))

    return float(np.mean(ranks)) if ranks else np.nan

# ---------------------------------------------------------------------
# Top-K Macro Precision and Recall
# (ranking-based, not threshold-based)
# ---------------------------------------------------------------------
def precision_recall_at_k(scores, truths, k=15):
    n_drug = scores.shape[0]
    precisions = []
    recalls = []

    for i in range(n_drug):
        row_scores = scores[i]
        topk = np.argsort(-row_scores)[:k]

        hits = truths[i, topk].sum()
        precisions.append(hits / float(k))

        relevant = truths[i].sum()
        if relevant > 0:
            recalls.append(hits / float(relevant))

    precision_macro = float(np.mean(precisions))
    recall_macro = float(np.mean(recalls)) if recalls else 0.0

    return precision_macro, recall_macro

# ---------------------------------------------------------------------
# Full evaluation logic called by your main script
# ---------------------------------------------------------------------
def compute_all_metrics(truth_matrix, score_matrix, k=15, beta=2.0):
    """
    truth_matrix: (n_drug, n_adr) binary matrix
    score_matrix: (n_drug, n_adr) probabilities
    """

    # Flatten for threshold selection
    y_true = truth_matrix.flatten()
    y_pred = score_matrix.flatten()

    # ---- threshold finding ----
    thr_data = find_best_threshold(y_true, y_pred, beta=beta)
    thr = thr_data["threshold"]

    # ---- classification metrics at threshold ----
    binary_pred = (score_matrix > thr).astype(int)

    tp = int(((binary_pred == 1) & (truth_matrix == 1)).sum())
    fp = int(((binary_pred == 1) & (truth_matrix == 0)).sum())
    fn = int(((binary_pred == 0) & (truth_matrix == 1)).sum())
    tn = int(((binary_pred == 0) & (truth_matrix == 0)).sum())

    # F1, MCC etc.
    f1 = float(f1_score(y_true, binary_pred.flatten()))
    mcc = float(matthews_corrcoef(y_true, binary_pred.flatten()))

    # AUPR
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred)
    aupr = float(auc(rec_curve, prec_curve))

    # ---- Top-K metrics ----
    prec_k, rec_k = precision_recall_at_k(score_matrix, truth_matrix, k=k)

    # ---- Mean Rank ----
    mr = mean_rank(score_matrix, truth_matrix)

    # ---- return everything ----
    return {
        # threshold optimization
        "threshold": thr,
        "f_beta": thr_data["fbeta"],
        "precision_at_thr": thr_data["precision"],
        "recall_at_thr": thr_data["recall"],
        "f1_at_thr": f1,

        # confusion matrix
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,

        # main metrics
        "aupr": aupr,
        "mcc": mcc,
        "mr": mr,

        # top-k
        f"precision_at_{k}": prec_k,
        f"recall_at_{k}": rec_k,
    }, thr