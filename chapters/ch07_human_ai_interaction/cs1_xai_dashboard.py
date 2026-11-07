"""Chapter 7 - Case Study 1: Explainable AI Dashboards for Transportation Safety Decision Support.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/07a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch07` or `python scripts/run_all.py --chapter 7`;
outputs are written to outputs/ch07/.
"""
import numpy as np
import pandas as pd

# ---------- Synthetic rural network with a known crash-generating process ----------
rng = np.random.default_rng(7)
N = 1200
design = rng.normal(0, 1, N)                        # latent design standard (not observed)
seg = pd.DataFrame({
    "aadt": np.round(np.exp(rng.normal(np.log(3500), 0.6, N)), -1),          # veh/day
    "curve": np.round(rng.gamma(1.5, 2.0, N), 1),                            # degrees
    "access": rng.poisson(8, N),                                             # per mile
    "shoulder": np.clip(np.round(3 + 1.5 * design + rng.normal(0, 1, N)), 0, 8) + 0.0})
seg["speed"] = np.where(design + rng.normal(0, 0.8, N) > 0.3, 55, 45)        # mph limit
seg["lit"] = (rng.random(N) < 1 / (1 + np.exp(-(seg.access - 12) / 3))).astype(int)
seg["length"] = np.round(rng.uniform(0.3, 2.0, N), 2)                        # miles
FEAT = ["aadt", "curve", "access", "shoulder", "speed", "lit"]


def true_log_rate(d):
    # log crashes per mile-year: the truth that the analyst never sees
    return (-1.2 + 0.8 * np.log(d.aadt / 1000) + 0.09 * d.curve + 0.025 * d.access
            - 0.07 * d.shoulder + 0.01 * (d.speed - 45) - 0.25 * d.lit
            + 0.04 * d.curve * (1 - d.lit))


expo = 5 * seg.length                                                        # mile-years
seg["crashes"] = rng.poisson(expo * np.exp(true_log_rate(seg)) * rng.gamma(4, 0.25, N))
print(seg[FEAT + ["crashes"]].describe().T[["mean", "std", "min", "max"]].round(2)
      .to_string())

# ---------- Crash-rate model: Poisson gradient boosting with exposure weights ----------
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

tr, te = train_test_split(np.arange(N), test_size=0.25, random_state=0)


def fit(idx, seed=0):
    m = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                      max_leaf_nodes=15, min_samples_leaf=20,
                                      random_state=seed)
    return m.fit(seg[FEAT].iloc[idx], (seg.crashes / expo).iloc[idx],
                 sample_weight=expo.iloc[idx])


gbm = fit(tr)
pred = gbm.predict(seg[FEAT])                       # predicted crashes per mile-year
truth = np.exp(true_log_rate(seg)).to_numpy()
observed = (seg.crashes / expo).to_numpy()


def top_overlap(score, ref, idx, frac=0.10):
    k = int(len(idx) * frac)
    a, b = np.argsort(-score[idx])[:k], np.argsort(-ref[idx])[:k]
    return len(set(a) & set(b)) / k


print(f"\nHeld-out segments: {len(te)}")
for name, s in [("Observed crash rate", observed), ("Model prediction", pred)]:
    rs = spearmanr(s[te], truth[te]).statistic
    print(f"  {name:<20} Spearman with true rate {rs:.3f};"
          f"  top-decile overlap {top_overlap(s, truth, te):.0%}")

# ---------- Exact interventional Shapley values on the log scale ----------
from math import factorial

P = len(FEAT)
MASKS = (np.arange(2 ** P)[:, None] >> np.arange(P)) & 1 == 1     # all 64 coalitions
W = np.array([factorial(s) * factorial(P - s - 1) / factorial(P) for s in range(P)])


def shapley(log_f, X, bg):
    # phi[i, j]: Shapley value of feature j for row i, against background rows bg
    hyb = np.where(MASKS[None, :, None, :], X[:, None, None, :], bg[None, None, :, :])
    v = log_f(hyb.reshape(-1, P)).reshape(len(X), 2 ** P, -1).mean(axis=2)
    phi = np.zeros((len(X), P))
    for j in range(P):
        S = np.flatnonzero(~MASKS[:, j])                            # coalitions without j
        phi[:, j] = ((v[:, S | (1 << j)] - v[:, S]) * W[MASKS[S].sum(1)]).sum(axis=1)
    return phi, v[:, 0]


as_df = lambda Z: pd.DataFrame(Z, columns=FEAT)
log_model = lambda m: (lambda Z: np.log(m.predict(as_df(Z))))
log_truth = lambda Z: true_log_rate(as_df(Z)).to_numpy()
bg = seg[FEAT].iloc[tr].sample(100, random_state=1).to_numpy(float)
ex = seg[FEAT].iloc[te].sample(120, random_state=2).to_numpy(float)
phi_m, base_m = shapley(log_model(gbm), ex, bg)
phi_t, base_t = shapley(log_truth, ex, bg)
gap = np.abs(base_m + phi_m.sum(1) - log_model(gbm)(ex)).max()
print(f"Efficiency check: max |base + sum(phi) - log f(x)| = {gap:.1e}")

# ---------- Stability: refit on bootstrap resamples and re-explain ----------
B, k_top = 10, 20
boot_imp, boot_pred, jacc = [], [], []
top0 = set(np.argsort(-pred)[:k_top])
for b in range(B):
    m_b = fit(rng.choice(tr, len(tr), replace=True), seed=b + 1)
    boot_imp.append(np.abs(shapley(log_model(m_b), ex, bg)[0]).mean(0))
    boot_pred.append(m_b.predict(seg[FEAT]))
    jacc.append(len(top0 & set(np.argsort(-boot_pred[-1])[:k_top])) / len(
        top0 | set(np.argsort(-boot_pred[-1])[:k_top])))
boot_imp, boot_pred = np.array(boot_imp), np.array(boot_pred)
corr = [np.corrcoef(phi_m[:, j], phi_t[:, j])[0, 1] for j in range(P)]
imp = pd.DataFrame({"model": np.abs(phi_m).mean(0), "boot_sd": boot_imp.std(0),
                    "truth": np.abs(phi_t).mean(0),
                    "corr": corr},
                   index=FEAT).sort_values("model", ascending=False)
print("\nGlobal importance, mean |phi| on the log scale (120 held-out segments):")
print(imp.round(3).to_string())
print(f"\nTop-{k_top} list stability over {B} refits: mean Jaccard {np.mean(jacc):.2f} "
      f"(min {np.min(jacc):.2f})")

# ---------- Local explanation for the segment ranked first ----------
k = int(np.argmax(pred))
xk = seg[FEAT].iloc[[k]].to_numpy(float)
pk_m, bk_m = shapley(log_model(gbm), xk, bg)
pk_t, bk_t = shapley(log_truth, xk, bg)
print(f"\nSegment {k}: predicted {pred[k]:.2f} vs true {truth[k]:.2f} "
      "crashes per mile-year")
print(f"{'feature':<10}{'value':>9}{'phi model':>11}{'phi truth':>11}")
for j in np.argsort(-np.abs(pk_m[0])):
    print(f"{FEAT[j]:<10}{xk[0, j]:>9.1f}{pk_m[0, j]:>11.3f}{pk_t[0, j]:>11.3f}")
print(f"{'base':<10}{'':>9}{bk_m[0]:>11.3f}{bk_t[0]:>11.3f}")

# ---------- What-if counterfactuals for the segment ranked first ----------
from itertools import combinations

thr = np.quantile(pred, 0.90)                       # review threshold: top decile
OPTIONS = {"Shoulder +2 ft": ("shoulder", +2, 0.18),
           "Shoulder +4 ft": ("shoulder", +4, 0.35),
           "Add lighting": ("lit", None, 0.12), "Access -30%": ("access", 0.7, 0.09),
           "Limit 55 to 45": ("speed", 45, 0.005)}   # (feature, change, $M per mile)


def apply(x, names):
    x = x.copy()
    for n in names:
        f, ch, _ = OPTIONS[n]
        x[f] = {"shoulder": lambda v: min(v + ch, 8), "lit": lambda v: 1,
                "access": lambda v: round(v * ch), "speed": lambda v: ch}[f](x[f].iloc[0])
    return x


x0 = seg[FEAT].iloc[[k]]
valid = [n for n in OPTIONS if not (apply(x0, [n]).to_numpy() == x0.to_numpy()).all()]
rows = []
for r in (1, 2):
    for names in combinations(valid, r):              # skip changes that do nothing here
        if len({OPTIONS[n][0] for n in names}) < r:
            continue                                  # one change per feature
        x1 = apply(x0, names)
        m1, t1 = gbm.predict(x1)[0], np.exp(true_log_rate(x1)).iloc[0]
        rows.append((" + ".join(names), sum(OPTIONS[n][2] for n in names),
                     100 * (1 - m1 / pred[k]), 100 * (1 - t1 / truth[k]), m1 < thr))
cf = pd.DataFrame(rows, columns=["change", "cost", "model_cut", "true_cut", "clears"])
cf["model_per_M"] = cf.model_cut / cf.cost
cf = cf.sort_values("model_per_M", ascending=False)
print(f"Review threshold (90th percentile of predictions): {thr:.2f} per mile-year")
print(f"Segment {k}: model {pred[k]:.2f}, truth {truth[k]:.2f}; "
      f"options that clear the threshold: {int(cf.clears.sum())} of {len(cf)}")
print(cf.drop(columns="clears").to_string(index=False, float_format=lambda v: f"{v:.1f}"))
best_m = cf.iloc[0]
best_t = cf.loc[(cf.true_cut / cf.cost).idxmax()]
print(f"\nBest reduction per $M by the model: {best_m.change}; "
      f"by the truth: {best_t.change}")

# ---------- Figure: the dashboard ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
GOLD, SAGE, SKY = "#DAB400", "#78978D", "#6ADBF1"  # Moonrise Kingdom palette
RUST, OLIVE = "#CE6B08", "#A59D30"
INK, PAPER, GRID = "#3F4A45", "#FBF8F4", "#D5D5D3" # text, background, grid
plt.rcParams.update({
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SAGE,
    "xtick.color": INK, "ytick.color": INK,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(11, 7.6))
order = np.argsort(-pred)[:15]                                           # (a) ranking
lo, hi = np.percentile(boot_pred[:, order], [5, 95], axis=0)
yy = np.arange(15)[::-1]
ax[0, 0].hlines(yy, lo, hi, color=SAGE, lw=2.2)
ax[0, 0].plot(pred[order], yy, "o", color=GOLD, mec=INK, ms=6,
              label="model (90% refit band)")
ax[0, 0].plot(truth[order], yy, "x", color=RUST, ms=6, label="true rate")
ax[0, 0].axvline(thr, color=OLIVE, ls="--", lw=1.2)
ax[0, 0].set_yticks(yy, [f"seg {i}" for i in order], fontsize=7)
ax[0, 0].set(xlabel="Crashes per mile-year", title="(a) Top 15 segments, with uncertainty")
ax[0, 0].legend(fontsize=7, frameon=False, loc="center right")
xs = np.arange(P)                                                        # (b) global
ax[0, 1].bar(xs - 0.2, imp.model, 0.38, yerr=imp.boot_sd, color=GOLD, ecolor=INK,
             label="model (sd over refits)")
ax[0, 1].bar(xs + 0.2, imp.truth, 0.38, color=SAGE, label="true process")
ax[0, 1].set_xticks(xs, imp.index, fontsize=8)
ax[0, 1].set(ylabel="Mean |phi| (log scale)",
             title="(b) Global importance: model vs. truth")
ax[0, 1].legend(fontsize=7, frameon=False)
oj = np.argsort(np.abs(pk_m[0]))                                         # (c) local
ax[1, 0].barh(np.arange(P), pk_m[0, oj], color=np.where(pk_m[0, oj] > 0, RUST, SAGE))
ax[1, 0].plot(pk_t[0, oj], np.arange(P), "D", color=INK, ms=5, label="truth")
ax[1, 0].set_yticks(np.arange(P), [f"{FEAT[j]} = {xk[0, j]:g}" for j in oj], fontsize=8)
ax[1, 0].axvline(0, color=INK, lw=0.8)
ax[1, 0].set(xlabel="phi (log crashes per mile-year)",
             title=f"(c) Why segment {k} ranks first")
ax[1, 0].legend(fontsize=7, frameon=False, loc="lower right")
cfp = cf.sort_values("cost")                                             # (d) what-if
yy = np.arange(len(cfp))
ax[1, 1].barh(yy + 0.2, cfp.model_cut, 0.38, color=GOLD, label="model")
ax[1, 1].barh(yy - 0.2, cfp.true_cut, 0.38, color=SAGE, label="true process")
ax[1, 1].set_yticks(yy, [f"{c} (${v:.2f}M)" for c, v in zip(cfp.change, cfp.cost)],
                    fontsize=7)
ax[1, 1].set(xlabel="Predicted reduction in crash rate (%)",
             title=f"(d) What-if for segment {k}")
ax[1, 1].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("chap07_fig01_xai_dashboard.png", dpi=200)
