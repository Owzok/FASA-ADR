from Framework import hybrid_stacking
from metrics import compute_all_metrics
from sklearn.metrics import precision_recall_curve
import pandas as pd
import numpy as np
import json
import os

path = "../../dataset/final_dataset1/"
#print("Running OURS W/o Dropout")

drug_se_mat = np.loadtxt(path+"drug_se_mat.txt")

n_drugs, n_adrs = drug_se_mat.shape
drug2id = {f"Drug_{i}": i for i in range(n_drugs)}
adr2id = {f"ADR_{j}": j for j in range(n_adrs)}

#with open(path+"drug2id.json", "r") as f:
#    drug2id = json.load(f)

#with open(path+"adr2id.json", "r") as f:
#    adr2id = json.load(f)
#
id2drug = {v: k for k, v in drug2id.items()}
id2adr = {v: k for k, v in adr2id.items()}

rows, cols = np.nonzero(drug_se_mat)
data = []

for r, c in zip(rows, cols):
    data.append({
        "drug_id": id2drug[r],
        "adr_id": id2adr[c],
        "association": 1
    })

df = pd.DataFrame(data)
df.to_csv("drug_adr_associations.csv", index=False)
print("✅ Generado: drug_adr_associations.csv")

drug_adr_matrix = df.pivot_table(
    index='drug_id',
    columns='adr_id',
    values='association',
    fill_value=0
)

n_drugs = drug_adr_matrix.shape[0]
n_adrs = drug_adr_matrix.shape[1]

total_entries = drug_adr_matrix.size
positive_entries = (drug_adr_matrix == 1).sum().sum()
zero_entries = (drug_adr_matrix == 0).sum().sum()
sparsity = (zero_entries / total_entries)

bias_factor = np.log((1-sparsity)/sparsity)
print("Optimal Bias Factor:", bias_factor)

clean_matrix = drug_adr_matrix.to_numpy().astype(np.float32)

hybrid_meta, hybrid_preds = hybrid_stacking(
    X=clean_matrix,
    n_folds=5,
    n_runs=1,
    random_state=42,
    bias_factor=bias_factor
)

final_prob_matrix = hybrid_meta['final_matrix']

print(f"\n{'='*70}")
print("CONVERTING TO BINARY PREDICTIONS")
print(f"{'='*70}")

results, thresh = compute_all_metrics(clean_matrix, final_prob_matrix, k=15, beta=1)

print("\n=== METRICS (Best Run) ===")
for k, v in results.items():
    print(f"{k}: {v}")

# Extract aggregated metrics across all runs from best_run
if 'metrics_summary' in hybrid_meta:
    print(f"\n{'='*70}")
    print("AGGREGATED METRICS ACROSS RUNS (Mean ± Std)")
    print(f"{'='*70}")
    for k, v in sorted(hybrid_meta['metrics_summary'].items()):
        mean_v = v['mean']
        std_v = v['std']
        print(f"{k}: {mean_v:.4f} ± {std_v:.4f}")

binary_matrix = (final_prob_matrix > thresh).astype(int)

print(f"\n{'='*70}")
print("BINARY MATRIX STATISTICS")
print(f"{'='*70}")
print(f"Shape: {binary_matrix.shape}")
print(f"Total predictions: {binary_matrix.size:,}")
print(f"Positive predictions: {binary_matrix.sum():,} ({binary_matrix.sum()/binary_matrix.size*100:.2f}%)")
print(f"Original positives: {clean_matrix.sum():,.0f} ({clean_matrix.sum()/clean_matrix.size*100:.2f}%)")

# Count true positives, false positives, false negatives
tp = ((binary_matrix == 1) & (clean_matrix == 1)).sum()
fp = ((binary_matrix == 1) & (clean_matrix == 0)).sum()
fn = ((binary_matrix == 0) & (clean_matrix == 1)).sum()
tn = ((binary_matrix == 0) & (clean_matrix == 0)).sum()

print(f"\nConfusion Matrix:")
print(f"  True Positives:  {tp:,}")
print(f"  False Positives: {fp:,}")
print(f"  False Negatives: {fn:,}")
print(f"  True Negatives:  {tn:,}")

# =============================================
# Export Binary Matrix (Multiple Formats)
# =============================================

output_dir = "final_results"
os.makedirs(output_dir, exist_ok=True)

np.savetxt(os.path.join(output_dir, 'x_final_probs.txt'),
           final_prob_matrix,
           fmt="%.6f")

np.savetxt(os.path.join(output_dir, 'x_final_preds.txt'),
           binary_matrix.astype(int),
           fmt="%d")

with open(os.path.join(output_dir, 'x_metrics.json'), 'w') as mf:
    json.dump(results, mf, indent=2)

print(f"\nExported correctly both matrices.")
print(f"Saved metrics to: {os.path.join(output_dir, 'x_metrics.json')}")
