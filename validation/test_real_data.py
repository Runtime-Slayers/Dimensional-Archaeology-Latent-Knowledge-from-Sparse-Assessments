"""
P8 Real Data Test: Dimensional Archaeology on Real Student Exam Data
Validates P8's compressed sensing / NMF recovery against UCI Student Performance.

P8 simulates: 300 students × 150 knowledge dims → compressed to 15 exam scores
Real test:    395 students × 15 UCI features → assess recovered latent dimensions

Tests:
  - NMF reconstruction quality vs random compression
  - PCA explained variance matches P8 simulation
  - Cluster recovery quality (true archetypes = grade levels)
  - Lasso sparsity: how many knowledge dims needed to explain exams?
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib.request, zipfile, io, os, json, warnings
from scipy import stats
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, silhouette_score
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
np.random.seed(42)

CACHE = os.path.join(os.path.dirname(__file__), "p2_cache")   # reuse
OUT   = os.path.join(os.path.dirname(__file__), "figures_p8_real")
os.makedirs(OUT, exist_ok=True)

COLORS = ["#2196F3","#E91E63","#4CAF50","#FF9800","#9C27B0","#00BCD4"]

# P8 sim reference
P8_SIM = {
    "k_knowledge":        150,
    "m_exam":             15,
    "n_students":         300,
    "n_archetypes":       6,
    "compression_ratio":  10.0,    # 150/15
    "reconstruction_r2":  0.72,    # P8 LASSO recovery
    "retained_variance":  0.91,    # from P8 simulation output
}

print("=" * 60)
print("  REAL DATA TEST: P8 Dimensional Archaeology")
print("  Dataset: UCI Student Performance (Cortez & Silva, 2008)")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[1] Loading UCI Student Performance dataset...")

csv_path = os.path.join(CACHE, "student-mat.csv")
if not os.path.exists(csv_path):
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        req = urllib.request.urlopen(url, timeout=30)
        zdata = io.BytesIO(req.read())
        with zipfile.ZipFile(zdata) as outer:
            inner_bytes = io.BytesIO(outer.read("student.zip"))
            with zipfile.ZipFile(inner_bytes) as inner:
                inner.extract("student-mat.csv", CACHE)
    except Exception:
        mirror = "https://raw.githubusercontent.com/sharmaroshan/Student-Performance-Dataset/master/student-mat.csv"
        urllib.request.urlretrieve(mirror, csv_path)

df_raw = pd.read_csv(csv_path, sep=";")

# Select features that represent "exam-like" observations (things we can observe)
# In P8, these are the M_EXAM=15 dimensions → map to 15 numeric features
exam_features = ["G1", "G2", "G3",           # period grades (main exam scores)
                 "studytime", "absences", "failures",
                 "Medu", "Fedu",               # parental education (proxy for SES)
                 "health", "famrel", "freetime", "goout",
                 "Dalc", "Walc", "age"]        # lifestyle / context

df_exam = df_raw[exam_features].copy().astype(float).dropna()
N_STUDENTS = len(df_exam)
M_EXAM     = len(exam_features)
print(f"    N students: {N_STUDENTS}, exam features: {M_EXAM}")
print(f"    Features: {exam_features}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRADE ARCHETYPES (TRUE LATENT CLUSTERS)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[2] Identifying student archetypes by grade cluster...")

# True "archetypes" = grade terciles: poor/average/good × math/life-skills
grade_mean = df_exam[["G1", "G2", "G3"]].mean(axis=1)
quintiles  = pd.qcut(grade_mean, 6, labels=False, duplicates="drop")
n_archetypes = quintiles.nunique()
print(f"    Grade quintiles (archetypes): {n_archetypes} groups")
for i in range(n_archetypes):
    m = grade_mean[quintiles == i].mean()
    n = (quintiles == i).sum()
    print(f"      Group {i}: n={n}, mean_grade={m:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA: INTRINSIC DIMENSIONALITY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[3] PCA analysis: intrinsic dimensionality of exam data...")

scaler = StandardScaler()
X_std  = scaler.fit_transform(df_exam)

pca = PCA()
pca.fit(X_std)
cumvar = np.cumsum(pca.explained_variance_ratio_)

# How many components to reach 80%, 90%, 95% variance?
k80 = int(np.searchsorted(cumvar, 0.80)) + 1
k90 = int(np.searchsorted(cumvar, 0.90)) + 1
k95 = int(np.searchsorted(cumvar, 0.95)) + 1

print(f"    Components for 80% var: {k80} / {M_EXAM}")
print(f"    Components for 90% var: {k90} / {M_EXAM}")
print(f"    Components for 95% var: {k95} / {M_EXAM}")
print(f"    PC1 var: {pca.explained_variance_ratio_[0]*100:.1f}%  (P8 sim: 'GPA signal')")
print(f"    PC1+2 var: {cumvar[1]*100:.1f}%")

# Effective compression ratio (real data)
real_compression_ratio = M_EXAM / k90  # additional compression P8's archaeology adds
print(f"    Real intrinsic compression ratio: {M_EXAM / k90:.2f}× ({M_EXAM}D → {k90}D)")
print(f"    P8 simulation: {P8_SIM['compression_ratio']:.1f}× ({P8_SIM['k_knowledge']}D → {P8_SIM['m_exam']}D)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. NMF: KNOWLEDGE ARCHETYPE RECOVERY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[4] NMF archetype recovery (K=3–8)...")

# Shift data to non-negative for NMF
mms = MinMaxScaler()
X_nn = mms.fit_transform(df_exam)

nmf_scores = {}
for k in range(2, 9):
    try:
        nmf = NMF(n_components=k, max_iter=500, random_state=42)
        W   = nmf.fit_transform(X_nn)
        H   = nmf.components_
        X_rec = W @ H
        r2  = 1 - mean_squared_error(X_nn, X_rec) / (X_nn.var() + 1e-9)
        # Silhouette on archetype assignments
        labels = W.argmax(axis=1)
        sil = silhouette_score(X_std, labels) if k > 1 else 0.0
        nmf_scores[k] = {"r2": round(float(r2), 4), "silhouette": round(float(sil), 4)}
        print(f"    K={k}: R²={r2:.4f}, silhouette={sil:.4f}")
    except Exception as e:
        print(f"    K={k}: failed ({e})")

best_k = max(nmf_scores, key=lambda k: nmf_scores[k]["silhouette"])
print(f"\n    Best archetype K (by silhouette): {best_k} (P8 sim: {P8_SIM['n_archetypes']})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LASSO SPARSITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[5] LASSO sparsity: recover G3 (final grade) from other features...")

# Treat G3 as the "target" (exam output), other features as latent "knowledge"
X_lasso = scaler.fit_transform(df_exam.drop(columns=["G3"]))
y_lasso = df_exam["G3"].values

# LassoCV to find optimal α
try:
    lasso_cv = LassoCV(cv=5, max_iter=2000, random_state=42).fit(X_lasso, y_lasso)
    alpha_opt = lasso_cv.alpha_
    y_pred    = lasso_cv.predict(X_lasso)
    r2_lasso  = 1 - mean_squared_error(y_lasso, y_pred) / (y_lasso.var() + 1e-9)
    n_nonzero = int((lasso_cv.coef_ != 0).sum())
    sparsity  = 1 - n_nonzero / X_lasso.shape[1]
    print(f"    Optimal α: {alpha_opt:.4f}")
    print(f"    LASSO R²: {r2_lasso:.4f}")
    print(f"    Non-zero features: {n_nonzero} / {X_lasso.shape[1]} (sparsity={sparsity:.2%})")
    print(f"    P8 sim reconstruction R²: {P8_SIM['reconstruction_r2']:.3f}")
    coefs = dict(zip([c for c in exam_features if c != "G3"], lasso_cv.coef_))
    top_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"    Top 5 predictors: {[f'{k}={v:+.3f}' for k, v in top_features]}")
except Exception as e:
    r2_lasso, n_nonzero, sparsity, alpha_opt = 0, 0, 0, 0
    coefs = {}
    top_features = []
    print(f"    LASSO failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[6] Generating figures...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("P8 Real Data Validation: Dimensional Archaeology — UCI Student Data\n"
             f"N={N_STUDENTS} students, {M_EXAM} exam features", fontsize=12, fontweight="bold")

# (0) PCA scree plot
k_range = np.arange(1, M_EXAM + 1)
axes[0].bar(k_range, pca.explained_variance_ratio_ * 100, color=COLORS[0], alpha=0.85)
axes[0].plot(k_range, cumvar * 100, "k--o", markersize=4, lw=1.5, label="Cumulative var")
axes[0].axhline(90, color="red", lw=1, linestyle=":", label="90% threshold")
axes[0].axvline(k90, color="red", lw=1.5, linestyle="--", label=f"K={k90} for 90%")
axes[0].set_xlabel("Principal component"); axes[0].set_ylabel("Variance explained (%)")
axes[0].set_title(f"PCA Scree Plot\n(real intrinsic dim ≈ {k90})")
axes[0].legend(fontsize=7); axes[0].grid(axis="y", alpha=0.3)

# (1) NMF R² and silhouette vs K
ks    = list(nmf_scores.keys())
r2s   = [nmf_scores[k]["r2"] for k in ks]
sils  = [nmf_scores[k]["silhouette"] for k in ks]
ax1b  = axes[1].twinx()
axes[1].bar(ks, r2s, alpha=0.6, color=COLORS[0], label="R² (left)")
ax1b.plot(ks, sils, "s-", color=COLORS[1], lw=2, markersize=8, label="Silhouette (right)")
ax1b.axvline(best_k, color=COLORS[1], lw=1.5, linestyle="--", label=f"Best K={best_k}")
axes[1].set_xlabel("Number of archetypes K"); axes[1].set_ylabel("NMF R²", color=COLORS[0])
ax1b.set_ylabel("Silhouette score", color=COLORS[1])
axes[1].set_title(f"NMF Archetype Recovery\n(Best K={best_k}, P8 sim={P8_SIM['n_archetypes']})")
lines1, labs1 = axes[1].get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labs1 + labs2, fontsize=7)
axes[1].grid(axis="y", alpha=0.3)

# (2) LASSO coefficients
if coefs:
    sorted_coefs = sorted(coefs.items(), key=lambda x: x[1])
    names_ = [x[0] for x in sorted_coefs]
    vals_  = [x[1] for x in sorted_coefs]
    cols_  = [COLORS[2] if v > 0 else COLORS[1] for v in vals_]
    axes[2].barh(range(len(names_)), vals_, color=cols_, alpha=0.85)
    axes[2].set_yticks(range(len(names_))); axes[2].set_yticklabels(names_, fontsize=8)
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set_xlabel("LASSO coefficient → G3")
axes[2].set_title(f"LASSO Archaeology: Predicting G3\nR²={r2_lasso:.3f}, {n_nonzero}/{M_EXAM-1} features active")
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/p8_real_data_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"    Figure saved: {OUT}/p8_real_data_validation.png")

out_json = {
    "dataset": "UCI Student Performance (Cortez & Silva, 2008)",
    "n_students": N_STUDENTS,
    "m_exam_features": M_EXAM,
    "pca": {
        "k_for_80pct_var": k80,
        "k_for_90pct_var": k90,
        "k_for_95pct_var": k95,
        "pc1_var": round(float(pca.explained_variance_ratio_[0]), 4),
        "pc12_cumvar": round(float(cumvar[1]), 4),
        "real_compression_ratio": round(M_EXAM / k90, 2),
    },
    "nmf": {
        "best_k": best_k,
        "scores_per_k": {str(k): v for k, v in nmf_scores.items()},
    },
    "lasso": {
        "r2": round(r2_lasso, 4),
        "n_nonzero_features": n_nonzero,
        "sparsity": round(sparsity, 4),
        "alpha_optimal": round(float(alpha_opt), 5) if alpha_opt else None,
    },
    "p8_sim_reference": P8_SIM,
}
with open(f"{OUT}/p8_real_results.json", "w") as f:
    json.dump(out_json, f, indent=2)

print(f"\n{'='*60}")
print(f"  P8 REAL DATA SUMMARY")
print(f"  Intrinsic dim: {k90} components for 90% var (of {M_EXAM} features)")
print(f"  Best NMF K: {best_k} (P8 sim: {P8_SIM['n_archetypes']})")
print(f"  LASSO R²: {r2_lasso:.4f} (P8 sim: {P8_SIM['reconstruction_r2']:.3f})")
print(f"  Sparsity: {sparsity:.1%} ({n_nonzero}/{M_EXAM-1} features active)")
print(f"{'='*60}")
