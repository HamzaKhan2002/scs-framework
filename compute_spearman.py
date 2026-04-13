"""Compute Spearman rank correlation between SCS-A and OOS Sharpe for all 12 groups."""
import json
from scipy import stats

# Load the experiment_1 JSON which has both phase_a and phase_c for all 12 groups
with open("results/experiments/experiment_1_expanded.json") as f:
    data = json.load(f)

phase_a = data["phase_a"]
phase_c = data["phase_c"]

groups = sorted(phase_a.keys())

scs_a_vals = []
oos_sharpe_vals = []

print(f"{'Group':<30} {'SCS-A':>8} {'OOS Sharpe':>12}")
print("-" * 55)

for g in groups:
    scs_a = phase_a[g]["SCS_A"]
    sharpe = phase_c[g]["metrics"]["sharpe_ratio"]
    scs_a_vals.append(scs_a)
    oos_sharpe_vals.append(sharpe)
    print(f"{g:<30} {scs_a:>8.4f} {sharpe:>12.4f}")

rho, p = stats.spearmanr(scs_a_vals, oos_sharpe_vals)
print(f"\nSpearman rho = {rho:.4f}")
print(f"p-value      = {p:.4f}")
print(f"N            = {len(groups)}")

# Also compute excluding rejected groups (SCS-A = 0)
non_zero = [(a, s) for a, s in zip(scs_a_vals, oos_sharpe_vals) if a > 0]
if len(non_zero) >= 3:
    a_nz, s_nz = zip(*non_zero)
    rho_nz, p_nz = stats.spearmanr(a_nz, s_nz)
    print(f"\n--- Excluding rejected (SCS-A=0) groups ---")
    print(f"Spearman rho = {rho_nz:.4f}")
    print(f"p-value      = {p_nz:.4f}")
    print(f"N            = {len(non_zero)}")
