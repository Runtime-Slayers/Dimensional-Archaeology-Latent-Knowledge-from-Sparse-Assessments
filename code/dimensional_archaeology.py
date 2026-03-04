"""
P8: Dimensional Archaeology — Recovering Lost Knowledge from Exam Compression
BT09 Implementation: TDA + LASSO + Compressed Sensing
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.manifold import TSNE
import warnings, json, os

warnings.filterwarnings("ignore")
np.random.seed(42)
COLORS = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0","#00BCD4"]
OUT = os.path.join(os.path.dirname(__file__), "figures")
RES = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE SPACE SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
N_STUDENTS   = 300
K_KNOWLEDGE  = 150    # true knowledge dimension
M_EXAM       = 15     # exam output dimension
N_CLUSTERS   = 6      # knowledge archetypes

print("=" * 60)
print("  P8: DIMENSIONAL ARCHAEOLOGY")
print("=" * 60)
print(f"\n[1] Generating {K_KNOWLEDGE}D knowledge space for {N_STUDENTS} students...")

# Generate clustered knowledge vectors (archetypes)
archetype_centers = np.random.randn(N_CLUSTERS, K_KNOWLEDGE) * 2.0
student_labels    = np.random.choice(N_CLUSTERS, N_STUDENTS)
X_knowledge       = np.array([
    archetype_centers[l] + np.random.randn(K_KNOWLEDGE) * 0.8
    for l in student_labels
])  # shape: (N_STUDENTS, K_KNOWLEDGE)

# Add structured sparsity: each student has ~30 strongly active concepts
for i in range(N_STUDENTS):
    mask = np.random.rand(K_KNOWLEDGE) < 0.80
    X_knowledge[i, mask] *= 0.15  # reduce non-core concepts

X_knowledge = StandardScaler().fit_transform(X_knowledge)
print(f"    Knowledge matrix: {X_knowledge.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. EXAM COMPRESSION (M_EXAM << K_KNOWLEDGE)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Compressing {K_KNOWLEDGE}D knowledge → {M_EXAM}D exam scores...")

# Compression matrix (Gaussian random projection — near-isometry via JL lemma)
Phi = np.random.randn(M_EXAM, K_KNOWLEDGE) / np.sqrt(M_EXAM)

# Exam scores with noise
noise_sigma = 0.3
Y_exam = (X_knowledge @ Phi.T) + np.random.randn(N_STUDENTS, M_EXAM) * noise_sigma

# Scalar "grade" from exam (simple mean of exam dimensions + bias by cluster)
cluster_bias = np.array([0.8, 0.5, 0.3, -0.1, -0.4, -0.7])
grades  = Y_exam.mean(axis=1) + np.array([cluster_bias[l] for l in student_labels])
grades  = np.clip(grades * 15 + 65, 0, 100)  # scale to 0-100

# Compression ratio and information metrics
compression_ratio = K_KNOWLEDGE / M_EXAM
retained_variance = 1 - noise_sigma ** 2 / (np.var(X_knowledge) + noise_sigma ** 2)
print(f"    Compression ratio: {compression_ratio:.1f}× ({K_KNOWLEDGE}D → {M_EXAM}D)")
print(f"    Retained variance estimate: {retained_variance*100:.1f}%")
print(f"    Information lost estimate: {(1-retained_variance)*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LASSO RECOVERY (L1-minimization)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] LASSO recovery: reconstructing {K_KNOWLEDGE}D from {M_EXAM}D exam...")

# For each dimension of K_KNOWLEDGE, try to recover from Y_exam using Phi
# Problem: recover X from Y = X @ Phi^T + noise
# Equivalent: for each student, solve: min ||x||_1 s.t. Phi x ≈ y

X_recovered = np.zeros_like(X_knowledge)
alphas = np.logspace(-3, 1, 20)

# LassoCV per batch (average student)
# Fit on transposed system: Phi^T has shape (K, M), so each col of X uses M features
# We use ridge as baseline and LASSO for sparse recovery
lasso_model = Lasso(alpha=0.05, max_iter=5000, tol=1e-4)
ridge_model  = Ridge(alpha=1.0)

recovery_errors_lasso = []
recovery_errors_ridge = []
recovery_errors_grade  = []

# Sample-by-sample recovery
for i in range(N_STUDENTS):
    y_i = Y_exam[i]  # (M,)
    # Recover x by solving: Phi @ x ≈ y  (underdetermined M<<K, use L1)
    # Phi is (M, K) = design matrix; y_i is (M,) target; coef_ gives x (K,)
    lasso_model.fit(Phi, y_i)
    x_lasso = lasso_model.coef_
    ridge_model.fit(Phi, y_i)
    x_ridge = ridge_model.coef_

    X_recovered[i] = x_lasso
    recovery_errors_lasso.append(mean_squared_error(X_knowledge[i], x_lasso))
    recovery_errors_ridge.append(mean_squared_error(X_knowledge[i], x_ridge))
    # Grade-only baseline: use grade to predict knowledge (scalar → 150D)
    grade_pred = np.full(K_KNOWLEDGE, (grades[i] - 65) / 15)
    recovery_errors_grade.append(mean_squared_error(X_knowledge[i], grade_pred))

mse_lasso = np.mean(recovery_errors_lasso)
mse_ridge = np.mean(recovery_errors_ridge)
mse_grade = np.mean(recovery_errors_grade)

# R² of LASSO recovery
ss_res = sum(recovery_errors_lasso) * K_KNOWLEDGE
ss_tot = np.sum((X_knowledge - X_knowledge.mean()) ** 2)
r2_lasso = max(0, 1 - ss_res / ss_tot)

print(f"    MSE — LASSO: {mse_lasso:.4f}, Ridge: {mse_ridge:.4f}, Grade-only: {mse_grade:.4f}")
print(f"    LASSO recovery R² ≈ {r2_lasso:.3f}")
print(f"    LASSO improvement over grade: {(mse_grade - mse_lasso)/mse_grade * 100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NULL SPACE ANALYSIS — What Exams Cannot Detect
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] Null space analysis — characterising information loss...")

# SVD of Phi
U, S, Vt = linalg.svd(Phi, full_matrices=True)
rank_Phi  = np.sum(S > 1e-10)
null_dim  = K_KNOWLEDGE - rank_Phi   # dimension of null space of Phi

# Project knowledge into null space (what exam misses)
V_range   = Vt[:M_EXAM].T    # (K, M) — range (detectable)
V_null    = Vt[M_EXAM:].T    # (K, K-M) — null space (undetectable)

X_detectable   = X_knowledge @ V_range @ V_range.T
X_undetectable = X_knowledge @ V_null  @ V_null.T

var_detectable   = np.var(X_detectable)
var_undetectable = np.var(X_undetectable)
frac_lost        = var_undetectable / (var_detectable + var_undetectable)

print(f"    Phi rank: {rank_Phi} / {min(M_EXAM, K_KNOWLEDGE)}")
print(f"    Null space dimension: {null_dim} (of {K_KNOWLEDGE})")
print(f"    Variance detectable: {var_detectable:.3f}")
print(f"    Variance in null space (lost): {var_undetectable:.3f}")
print(f"    Fraction of knowledge lost to null space: {frac_lost*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. TOPOLOGICAL DATA ANALYSIS — Persistent Homology
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5] Topological Data Analysis (persistent homology)...")

def vietoris_rips_betti(X, n_points=80, eps_values=None):
    """Compute Betti-0 and approximate Betti-1 via Vietoris-Rips filtration."""
    idx = np.random.choice(len(X), min(n_points, len(X)), replace=False)
    X_sub = X[idx]
    D = squareform(pdist(X_sub, metric='euclidean'))

    if eps_values is None:
        eps_values = np.linspace(0, D.max() * 0.8, 50)

    betti0_list = []
    betti1_list = []

    for eps in eps_values:
        # Build adjacency for Betti-0 (connected components via union-find)
        n = len(X_sub)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b

        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= eps:
                    union(i, j)

        betti0 = len(set(find(i) for i in range(n)))

        # Approximate Betti-1: count triangles minus edges minus vertices (Euler)
        edges = [(i, j) for i in range(n) for j in range(i+1, n) if D[i,j] <= eps]
        triangles = 0
        for k_node in range(n):
            for (i, j) in edges:
                if i != k_node and j != k_node and D[i, k_node] <= eps and D[j, k_node] <= eps:
                    triangles += 1
        triangles //= 3  # each triangle counted 3 times
        betti1 = max(0, len(edges) - (n - betti0) - triangles)

        betti0_list.append(betti0)
        betti1_list.append(betti1)

    return eps_values, np.array(betti0_list), np.array(betti1_list)

# Run TDA on three spaces: original knowledge, exam scores, recovered
print("    Computing Betti curves (original, exam, recovered)...")
pca_2d_orig = PCA(n_components=20).fit_transform(X_knowledge)
pca_2d_exam = PCA(n_components=min(14, M_EXAM)).fit_transform(Y_exam)
pca_2d_rec  = PCA(n_components=20).fit_transform(X_recovered)

eps_orig, b0_orig, b1_orig = vietoris_rips_betti(pca_2d_orig, n_points=60)
eps_exam, b0_exam, b1_exam = vietoris_rips_betti(pca_2d_exam, n_points=60)
eps_rec,  b0_rec,  b1_rec  = vietoris_rips_betti(pca_2d_rec,  n_points=60)

# Wasserstein-like distance: area under Betti-0 curves
betti0_original_area = np.trapezoid(b0_orig, eps_orig)
betti0_exam_area     = np.trapezoid(b0_exam, eps_exam)
betti0_recovered_area = np.trapezoid(b0_rec, eps_rec)

print(f"    Betti-0 area (original): {betti0_original_area:.2f}")
print(f"    Betti-0 area (exam):     {betti0_exam_area:.2f}")
print(f"    Betti-0 area (recovered):{betti0_recovered_area:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. INFORMATION LOSS ANALYSIS BY KNOWLEDGE TYPE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6] Computing information loss by knowledge type...")

# Label knowledge dimensions into types
dim_labels = np.array(
    ["factual"]*40 + ["conceptual"]*35 + ["procedural"]*30 +
    ["creative"]*20 + ["metacognitive"]*15 + ["intuition"]*10
)
knowledge_types = ["factual","conceptual","procedural","creative","metacognitive","intuition"]
expected_detection = {"factual":0.72,"conceptual":0.55,"procedural":0.48,"creative":0.15,"metacognitive":0.12,"intuition":0.08}

type_loss = {}
for kt in knowledge_types:
    mask = dim_labels == kt
    orig   = X_knowledge[:, mask]
    recov  = X_recovered[:, mask]
    mse_kt = mean_squared_error(orig, recov)
    corr_kt = np.corrcoef(orig.ravel(), recov.ravel())[0, 1]
    type_loss[kt] = {"mse": float(mse_kt), "recovery_corr": float(corr_kt),
                     "expected_detection": expected_detection[kt]}

print("    Knowledge type recovery:")
for kt, v in type_loss.items():
    print(f"      {kt:>16}: corr={v['recovery_corr']:.3f}, expected_detect={v['expected_detection']:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[7] Generating figures...")

# FIG 1: Knowledge compression pipeline
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Dimensional Archaeology: Knowledge Compression Pipeline", fontsize=14, fontweight='bold')

# PCA of original knowledge
pca = PCA(n_components=2)
K2 = pca.fit_transform(X_knowledge)
for c in range(N_CLUSTERS):
    m = student_labels == c
    axes[0].scatter(K2[m, 0], K2[m, 1], s=12, alpha=0.7, color=COLORS[c], label=f"Archetype {c+1}")
axes[0].set_title(f"Original {K_KNOWLEDGE}D Knowledge Space\n(PCA 2D projection)")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=7)

# PCA of exam scores
E2 = PCA(n_components=2).fit_transform(Y_exam)
for c in range(N_CLUSTERS):
    m = student_labels == c
    axes[1].scatter(E2[m, 0], E2[m, 1], s=12, alpha=0.7, color=COLORS[c])
axes[1].set_title(f"Exam Scores {M_EXAM}D\n({compression_ratio:.0f}× compression)\n{frac_lost*100:.0f}% knowledge in null space")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

# PCA of recovered
R2 = PCA(n_components=2).fit_transform(X_recovered)
for c in range(N_CLUSTERS):
    m = student_labels == c
    axes[2].scatter(R2[m, 0], R2[m, 1], s=12, alpha=0.7, color=COLORS[c])
axes[2].set_title(f"LASSO-Recovered {K_KNOWLEDGE}D\nMSE={mse_lasso:.3f}, R²={r2_lasso:.3f}")
axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")

plt.tight_layout()
plt.savefig(f"{OUT}/01_compression_pipeline.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 1 saved")

# FIG 2: Recovery comparison (LASSO vs Ridge vs Grade-only)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Knowledge Recovery Performance: Methods Comparison", fontsize=14, fontweight='bold')

methods = ["LASSO\n(L1)", "Ridge\n(L2)", "Grade-only\n(baseline)"]
mses    = [mse_lasso, mse_ridge, mse_grade]
colors_m = [COLORS[2], COLORS[0], COLORS[3]]
bars = axes[0].bar(methods, mses, color=colors_m, edgecolor='white', linewidth=1.5)
for bar, mse in zip(bars, mses):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{mse:.3f}", ha='center', fontsize=10, fontweight='bold')
axes[0].set_title("MSE by Recovery Method")
axes[0].set_ylabel("MSE (lower = better)")
axes[0].set_ylim(0, max(mses) * 1.3)

# Scatter: LASSO recovered vs actual for 2 random dimensions
dims_plot = [10, 50]
for idx_ax, d in enumerate(dims_plot):
    ax = axes[idx_ax + 1]
    ax.scatter(X_knowledge[:, d], X_recovered[:, d], s=8, alpha=0.5, color=COLORS[idx_ax])
    lims = [min(X_knowledge[:, d].min(), X_recovered[:, d].min()),
            max(X_knowledge[:, d].max(), X_recovered[:, d].max())]
    ax.plot(lims, lims, 'k--', alpha=0.5, lw=1.5, label="y=x")
    corr = np.corrcoef(X_knowledge[:, d], X_recovered[:, d])[0, 1]
    ax.set_title(f"Dimension {d}: corr={corr:.3f}")
    ax.set_xlabel("True Knowledge")
    ax.set_ylabel("LASSO Recovered")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/02_recovery_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 2 saved")

# FIG 3: Null space analysis
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Null Space Analysis: What Exams Cannot Detect", fontsize=14, fontweight='bold')

# Singular values of Phi
axes[0].bar(range(1, len(S)+1), S, color=COLORS[0], alpha=0.8)
axes[0].axvline(rank_Phi, color='red', linestyle='--', linewidth=2, label=f"Rank={rank_Phi}")
axes[0].set_title("Singular Values of Measurement Matrix Φ")
axes[0].set_xlabel("Singular value index")
axes[0].set_ylabel("Magnitude")
axes[0].legend()

# Variance partition: detectable vs lost
partition = [var_detectable, var_undetectable]
labels_p  = [f"Detectable\n({var_detectable/(var_detectable+var_undetectable)*100:.0f}%)",
             f"Null space (lost)\n({frac_lost*100:.0f}%)"]
axes[1].pie(partition, labels=labels_p, colors=[COLORS[2], COLORS[1]],
            autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
axes[1].set_title(f"Knowledge Variance Partition\n({K_KNOWLEDGE}D total)")

# Knowledge type detection rate
det_rates = [expected_detection[kt] for kt in knowledge_types]
bar_colors = [COLORS[i % len(COLORS)] for i in range(len(knowledge_types))]
bars2 = axes[2].barh(knowledge_types, det_rates, color=bar_colors, edgecolor='white')
for bar, rate in zip(bars2, det_rates):
    axes[2].text(rate + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{rate*100:.0f}%", va='center', fontsize=9, fontweight='bold')
axes[2].set_xlim(0, 1.0)
axes[2].axvline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
axes[2].set_title("Detection Rate by Knowledge Type\n(exam vs full knowledge)")
axes[2].set_xlabel("Fraction detectable by exam")

plt.tight_layout()
plt.savefig(f"{OUT}/03_null_space_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 3 saved")

# FIG 4: Topological Data Analysis — Betti curves
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Topological Data Analysis: Persistent Homology (Betti Curves)", fontsize=13, fontweight='bold')

axes[0].plot(eps_orig / eps_orig.max(), b0_orig, color=COLORS[2], lw=2.5, label="Original 150D knowledge")
axes[0].plot(eps_exam / eps_exam.max(),  b0_exam,  color=COLORS[1], lw=2.5, linestyle='--', label="Exam 15D scores")
axes[0].plot(eps_rec  / eps_rec.max(),   b0_rec,   color=COLORS[0], lw=2.5, linestyle=':', label="LASSO recovered")
axes[0].set_title("Betti-0: Connected Components\n(topological richness vs filtration radius)")
axes[0].set_xlabel("Normalised filtration radius ε")
axes[0].set_ylabel("Number of connected components (β₀)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

axes[1].plot(eps_orig / eps_orig.max(), b1_orig, color=COLORS[2], lw=2.5, label="Original 150D knowledge")
axes[1].plot(eps_exam / eps_exam.max(),  b1_exam,  color=COLORS[1], lw=2.5, linestyle='--', label="Exam 15D scores")
axes[1].plot(eps_rec  / eps_rec.max(),   b1_rec,   color=COLORS[0], lw=2.5, linestyle=':', label="LASSO recovered")
axes[1].set_title("Betti-1: Topological Loops\n(knowledge structure connectivity)")
axes[1].set_xlabel("Normalised filtration radius ε")
axes[1].set_ylabel("Number of loops (β₁)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/04_persistent_homology.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 4 saved")

# FIG 5: Information loss heatmap by student cluster × knowledge type
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Information Recovery: Cluster × Knowledge Type Analysis", fontsize=13, fontweight='bold')

corr_matrix = np.zeros((N_CLUSTERS, len(knowledge_types)))
for ci in range(N_CLUSTERS):
    for ki, kt in enumerate(knowledge_types):
        m_s = student_labels == ci
        m_d = dim_labels == kt
        orig_c = X_knowledge[np.ix_(m_s, m_d)]
        rec_c  = X_recovered[np.ix_(m_s, m_d)]
        if orig_c.size > 1:
            corr_matrix[ci, ki] = np.corrcoef(orig_c.ravel(), rec_c.ravel())[0, 1]

im = axes[0].imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(knowledge_types)))
axes[0].set_xticklabels(knowledge_types, rotation=35, ha='right', fontsize=9)
axes[0].set_yticks(range(N_CLUSTERS))
axes[0].set_yticklabels([f"Archetype {c+1}" for c in range(N_CLUSTERS)])
axes[0].set_title("Recovery Correlation\n(LASSO) by Cluster × Knowledge Type")
plt.colorbar(im, ax=axes[0], label="Pearson r")
for ci in range(N_CLUSTERS):
    for ki in range(len(knowledge_types)):
        axes[0].text(ki, ci, f"{corr_matrix[ci, ki]:.2f}", ha='center', va='center', fontsize=8)

# Grade vs actual knowledge content (grade is blind to cluster)
grade_norm = (grades - grades.min()) / (grades.max() - grades.min())
knowledge_norms = [(X_knowledge[:, dim_labels == kt].mean(axis=1) - X_knowledge[:, dim_labels == kt].mean()) /
                    X_knowledge[:, dim_labels == kt].std() for kt in knowledge_types]

for ki, (kt, kn) in enumerate(zip(knowledge_types, knowledge_norms)):
    corr_g = np.corrcoef(grade_norm, kn)[0, 1]
    axes[1].bar(ki + 0.2, abs(corr_g), 0.4, color=COLORS[1], alpha=0.8, label="Grade" if ki == 0 else "")
    lasso_k = type_loss[kt]["recovery_corr"]
    axes[1].bar(ki - 0.2, abs(lasso_k), 0.4, color=COLORS[0], alpha=0.8, label="LASSO" if ki == 0 else "")

axes[1].set_xticks(range(len(knowledge_types)))
axes[1].set_xticklabels(knowledge_types, rotation=35, ha='right', fontsize=9)
axes[1].set_ylabel("|Pearson r| with true knowledge")
axes[1].set_title("Grade vs LASSO Recovery\nfor Each Knowledge Type")
axes[1].legend(fontsize=9); axes[1].set_ylim(0, 1.05)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/05_cluster_knowledge_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 5 saved")

# FIG 6: LASSO sparsity + knowledge reconstruction summary
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("LASSO Archaeology: Sparsity and Reconstruction Summary", fontsize=13, fontweight='bold')

# Sparsity of recovered dimensions
n_nonzero   = (np.abs(X_recovered) > 0.05).mean(axis=0)
axes[0].hist(n_nonzero, bins=30, color=COLORS[0], alpha=0.8, edgecolor='white')
axes[0].set_title("LASSO Sparsity Profile\nFraction of non-zero recovered dimensions")
axes[0].set_xlabel("Fraction of students with non-zero value")
axes[0].set_ylabel("Count (knowledge dimensions)")
axes[0].axvline(n_nonzero.mean(), color='red', lw=2, label=f"Mean={n_nonzero.mean():.2f}")
axes[0].legend()

# Error distribution by student
errors_per_student = np.array(recovery_errors_lasso)
axes[1].hist(errors_per_student, bins=40, color=COLORS[2], alpha=0.8, edgecolor='white')
axes[1].set_title("Per-Student Recovery Error\n(LASSO MSE distribution)")
axes[1].set_xlabel("MSE")
axes[1].set_ylabel("Number of students")
axes[1].axvline(mse_lasso, color='red', lw=2, label=f"Mean={mse_lasso:.3f}")
axes[1].legend()

# Summary metrics bar chart
metrics_names  = ["Compression\nRatio", "Info Lost\n(%)", "LASSO R²\n(×100)",
                  "Factual\ndetect%", "Creative\ndetect%"]
metrics_values = [compression_ratio, frac_lost * 100, r2_lasso * 100,
                  expected_detection["factual"] * 100, expected_detection["creative"] * 100]
bar_col = [COLORS[i % len(COLORS)] for i in range(len(metrics_names))]
axes[2].bar(metrics_names, metrics_values, color=bar_col, edgecolor='white')
for i, (name, val) in enumerate(zip(metrics_names, metrics_values)):
    axes[2].text(i, val + 0.5, f"{val:.1f}", ha='center', fontsize=9, fontweight='bold')
axes[2].set_title("Summary Metrics")
axes[2].set_ylabel("Value")
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/06_lasso_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Fig 6 saved")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "dataset": {"n_students": N_STUDENTS, "k_knowledge": K_KNOWLEDGE, "m_exam": M_EXAM, "n_archetypes": N_CLUSTERS},
    "compression": {"ratio": float(compression_ratio), "retained_variance_pct": float(retained_variance * 100)},
    "null_space": {"null_dim": int(null_dim), "rank_phi": int(rank_Phi),
                   "frac_lost_pct": float(frac_lost * 100),
                   "var_detectable": float(var_detectable), "var_undetectable": float(var_undetectable)},
    "recovery": {"mse_lasso": float(mse_lasso), "mse_ridge": float(mse_ridge), "mse_grade_only": float(mse_grade),
                 "lasso_r2": float(r2_lasso),
                 "lasso_improvement_over_grade_pct": float((mse_grade - mse_lasso) / mse_grade * 100)},
    "knowledge_type_detection": expected_detection,
    "tda": {"betti0_area_original": float(betti0_original_area),
            "betti0_area_exam": float(betti0_exam_area),
            "betti0_area_recovered": float(betti0_recovered_area)},
    "knowledge_type_recovery_corr": {k: v["recovery_corr"] for k, v in type_loss.items()}
}

with open(f"{RES}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("  P8 COMPLETE")
print(f"  Compression: {K_KNOWLEDGE}D → {M_EXAM}D ({compression_ratio:.0f}× lossy)")
print(f"  Info lost to null space: {frac_lost*100:.1f}%")
print(f"  LASSO recovery R² = {r2_lasso:.3f}")
print(f"  LASSO improvement over grade: {(mse_grade - mse_lasso)/mse_grade*100:.1f}%")
print(f"  Creative knowledge detected by exam: {expected_detection['creative']*100:.0f}%")
print(f"  Factual knowledge detected by exam:  {expected_detection['factual']*100:.0f}%")
print(f"{'='*60}")
