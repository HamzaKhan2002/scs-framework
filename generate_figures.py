"""Generate 3 publication figures for Paper v2."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "experiments"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})


def fig1_power_curve():
    """Figure 1: Detection power vs threshold at different noise levels."""
    pa = json.load(open(RESULTS / "power_analysis_summary.json"))

    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    selected_k = ["k_0.1", "k_0.5", "k_1.0", "k_3.0", "k_8.0"]
    labels = ["k=0.1 (strong)", "k=0.5", "k=1.0", "k=3.0", "k=8.0 (weak)"]
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for ki, (k_key, label, marker, color) in enumerate(zip(selected_k, labels, markers, colors)):
        powers = [pa[k_key]["power"][f"tau_{t:.2f}"] * 100 for t in taus]
        ax.plot(taus, powers, marker=marker, color=color, label=label,
                linewidth=1.8, markersize=6, markeredgewidth=0.5, markeredgecolor='black')

    # Add operational threshold line
    ax.axvline(x=0.70, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.705, 50, r'$\tau=0.70$', color='red', fontsize=9, alpha=0.7)

    ax.set_xlabel(r'Approval Threshold $\tau$')
    ax.set_ylabel('Detection Power (%)')
    ax.set_title('Oracle Detection Power by Signal Strength')
    ax.set_ylim(-5, 105)
    ax.set_xlim(0.48, 0.82)
    ax.legend(title='Noise level', loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(taus)

    fig.savefig(FIG_DIR / "fig_power_curve.pdf")
    fig.savefig(FIG_DIR / "fig_power_curve.png")
    plt.close(fig)
    print("  Figure 1: Power curve saved")


def fig2_fpr_curve():
    """Figure 2: False positive rate vs threshold (binary vs ternary)."""
    fdr = json.load(open(RESULTS / "fdr_summary.json"))

    taus = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    fpr_binary = []
    fpr_ternary = []
    fpr_all = []

    for t in taus:
        key = f"tau_{t:.2f}"
        if key in fdr:
            fpr_binary.append(fdr[key]["mean_fpr_binary"] * 100)
            fpr_ternary.append(fdr[key]["mean_fpr_ternary"] * 100)
            fpr_all.append(fdr[key]["mean_fpr_all"] * 100)
        else:
            fpr_binary.append(0)
            fpr_ternary.append(0)
            fpr_all.append(0)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(taus, fpr_binary, 'o-', color='#e41a1c', label='Binary signals',
            linewidth=1.8, markersize=6)
    ax.plot(taus, fpr_ternary, 's-', color='#377eb8', label='Ternary signals',
            linewidth=1.8, markersize=6)
    ax.plot(taus, fpr_all, '^--', color='#4daf4a', label='Overall',
            linewidth=1.5, markersize=5, alpha=0.7)

    # Operational threshold
    ax.axvline(x=0.70, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(0.705, max(fpr_binary) * 0.5, r'$\tau=0.70$', color='red', fontsize=9, alpha=0.7)

    # 5% significance line
    ax.axhline(y=5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.49, 5.5, '5% significance', color='gray', fontsize=8, alpha=0.5)

    ax.set_xlabel(r'Approval Threshold $\tau$')
    ax.set_ylabel('False Positive Rate (%)')
    n_seeds = fdr.get("tau_0.50", {}).get("n_valid_seeds", "?")
    ax.set_title(f'Monte Carlo FDR Calibration (N={n_seeds:,} seeds, 100% label corruption)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(taus)
    ax.set_ylim(-2, max(fpr_binary) * 1.1 + 2)

    fig.savefig(FIG_DIR / "fig_fpr_curve.pdf")
    fig.savefig(FIG_DIR / "fig_fpr_curve.png")
    plt.close(fig)
    print("  Figure 2: FPR curve saved")


def fig3_component_heatmap():
    """Figure 3: SCS-A component heatmap for all 12 signal groups."""
    pf = json.load(open(ROOT / "results" / "pipeline_final.json"))

    groups = []
    components = ["S_time", "S_asset", "S_model", "S_seed", "S_dist"]
    data_matrix = []
    scs_a_scores = []

    for gk in sorted(pf["phase_a"]["group_results"].keys()):
        r = pf["phase_a"]["group_results"][gk]
        scs_a = r.get("SCS_A", 0)
        row = [r.get(c, 0) for c in components]
        groups.append(gk.replace("_directional_binary", " bin").replace("_multiclass_volatility", " tern"))
        data_matrix.append(row)
        scs_a_scores.append(scs_a)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([r'$S_{time}$', r'$S_{asset}$', r'$S_{model}$', r'$S_{seed}$', r'$S_{dist}$'],
                       fontsize=11)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9)

    # Add SCS-A scores on the right
    for i, score in enumerate(scs_a_scores):
        color = 'green' if score >= 0.70 else ('orange' if score >= 0.50 else 'red')
        ax.text(len(components) + 0.3, i, f'{score:.2f}', va='center', ha='left',
                fontsize=9, fontweight='bold', color=color)

    # Add text annotations in each cell
    for i in range(len(groups)):
        for j in range(len(components)):
            val = data_matrix[i, j]
            text_color = 'white' if val < 0.3 or val > 0.8 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=text_color)

    ax.set_title('SCS-A Component Scores by Signal Group')

    # Add SCS-A header
    ax.text(len(components) + 0.3, -0.8, 'SCS-A', ha='left', fontsize=10, fontweight='bold')

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.15)
    cbar.set_label('Component Score')

    # Draw approval threshold line
    # Highlight approved groups
    for i, score in enumerate(scs_a_scores):
        if score >= 0.70:
            ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(components), 1,
                                       fill=False, edgecolor='green', linewidth=2))

    fig.savefig(FIG_DIR / "fig_component_heatmap.pdf")
    fig.savefig(FIG_DIR / "fig_component_heatmap.png")
    plt.close(fig)
    print("  Figure 3: Component heatmap saved")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_power_curve()
    fig2_fpr_curve()
    fig3_component_heatmap()
    print("Done. Figures in:", FIG_DIR)
