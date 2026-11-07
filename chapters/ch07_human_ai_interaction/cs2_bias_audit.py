"""Chapter 7 - Case Study 2: Bias, Accountability, and Transparency in AI-Enabled Road Safety Systems.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/07a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch07` or `python scripts/run_all.py --chapter 7`;
outputs are written to outputs/ch07/.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import spearmanr

# ---------- Synthetic city: the same road physics, a different crash record ----------
rng = np.random.default_rng(31)
N, T = 800, 5                                            # intersections, years of data
grp = np.where(rng.random(N) < 0.5, "A", "B")            # two communities
city = pd.DataFrame({"group": grp})
city["arterial"] = (rng.random(N) < np.where(grp == "B", 0.55, 0.35)).astype(int)
city["veh"] = np.round(np.exp(rng.normal(np.log(14000), 0.45, N)
                              + 0.35 * city.arterial), -2)
city["ped"] = np.round(np.exp(rng.normal(np.log(np.where(grp == "B", 900, 500)),
                                         0.6, N)), -1)
city["signal"] = (rng.random(N) < 0.25 + 0.45 * city.arterial).astype(int)
city["speed"] = np.where(city.arterial == 1, 40, 30) + 5 * rng.integers(0, 2, N)
FEAT = ["arterial", "veh", "ped", "signal", "speed"]
city["true_rate"] = np.exp(-1.9 + 0.55 * np.log(city.veh / 1000)       # injury crashes/yr
                           + 0.35 * np.log(city.ped / 100) + 0.30 * city.arterial
                           - 0.20 * city.signal + 0.03 * (city.speed - 30))
REPORT = {"A": 0.90, "B": 0.60}                          # share of injury crashes reported
city["reported"] = rng.poisson(city.group.map(REPORT) * city.true_rate * T)
print(city.groupby("group").agg(sites=("veh", "size"), arterial=("arterial", "mean"),
      ped_median=("ped", "median"), true_rate=("true_rate", "mean"),
      reported_per_yr=("reported", lambda s: s.mean() / T)).round(2).to_string())


def oof_predict(y_rate, seed=0, folds=5):
    # out-of-fold predictions, so that no site is scored by a model trained on itself
    fold = np.random.default_rng(seed).integers(0, folds, N)
    out = np.zeros(N)
    for f in range(folds):
        m = HistGradientBoostingRegressor(loss="poisson", max_iter=250, learning_rate=0.05,
                                          max_leaf_nodes=15, min_samples_leaf=20,
                                          random_state=seed)
        keep = fold != f
        m.fit(city[FEAT][keep], y_rate[keep], sample_weight=np.full(keep.sum(), T))
        out[fold == f] = m.predict(city[FEAT][fold == f])
    return out


# ---------- Three ways to choose 160 sites (20%) for engineering review ----------
K = int(0.2 * N)
high = (city.true_rate >= np.sort(city.true_rate)[-K]).to_numpy()   # truly riskiest 20%
isB = (city.group == "B").to_numpy()
pred_base = oof_predict((city.reported / T).to_numpy())
# linked hospital records per group
n_link = 200
rho_hat = {g: rng.binomial(n_link, REPORT[g]) / n_link for g in "AB"}
pred_corr = oof_predict((city.reported / (T * city.group.map(rho_hat))).to_numpy())


def top_k(score):
    sel = np.zeros(N, bool); sel[np.argsort(-score)[:K]] = True
    return sel


def group_thresholds(score):
    # choose k_A + k_B = K so that true-positive rates are as equal as possible
    best = None
    for kA in range(K + 1):
        sel = np.zeros(N, bool)
        sel[np.flatnonzero(~isB)[np.argsort(-score[~isB])[:kA]]] = True
        sel[np.flatnonzero(isB)[np.argsort(-score[isB])[:K - kA]]] = True
        gap = abs((sel & high & ~isB).sum() / (high & ~isB).sum()
                  - (sel & high & isB).sum() / (high & isB).sum())
        if best is None or gap < best[0]:
            best = (gap, sel)
    return best[1]


policies = {"Baseline": (top_k(pred_base), pred_base),
            "Corrected labels": (top_k(pred_corr), pred_corr),
            "Group thresholds": (group_thresholds(pred_base), pred_base)}
rows = []
for name, (sel, score) in policies.items():
    for g, m in [("A", ~isB), ("B", isB)]:
        tp, fp = (sel & high & m).sum(), (sel & ~high & m).sum()
        rows.append((name, g, high[m].mean(), sel[m].mean(), tp / (high & m).sum(),
                     fp / (~high & m).sum(), tp / max(sel[m].sum(), 1),
                     score[m].sum() / city.true_rate[m].sum()))
audit = pd.DataFrame(rows, columns=["policy", "group", "base", "selected", "TPR", "FPR",
                                    "PPV", "pred/true"])
print(f"\nLinkage audit: estimated reporting rate A {rho_hat['A']:.3f}, "
      f"B {rho_hat['B']:.3f}")
print(audit.round(3).to_string(index=False))
print("\nTrue injury crashes per year at the 160 selected sites:")
for name, (sel, score) in policies.items():
    print(f"  {name:<18}{city.true_rate[sel].sum():7.1f}   "
          f"(Spearman with true risk {spearmanr(score, city.true_rate).statistic:.3f})")
print(f"  {'Oracle':<18}{city.true_rate[high].sum():7.1f}")

# ---------- Feedback loop: detection units allocated by what they detect ----------
U, MONTHS, RUNS = 40, 24, 500
TRUE = np.array([10.0, 10.0])           # near-misses per unit-month in A and B: equal


def simulate(policy, seed):
    r = np.random.default_rng(seed)
    units = np.array([28, 12])          # historical allocation favors A
    seen, exposure, share = np.zeros(2), np.zeros(2), []
    for t in range(MONTHS):
        seen += r.poisson(units * TRUE)  # near-misses are seen only where units are
        exposure += units
        w = seen if policy == "naive" else seen / exposure   # counts vs. counts per unit
        units = np.round(U * w / w.sum()).astype(int)
        units = np.clip(units, 1, U - 1); units[0] = U - units[1]
        share.append(units[1] / U)
    return np.array(share), seen[1] / seen.sum()


loop = {}
for policy in ["naive", "per-unit"]:
    sims = [simulate(policy, s) for s in range(RUNS)]
    loop[policy] = np.array([s[0] for s in sims])
    data_share = np.mean([s[1] for s in sims])
    q = np.percentile(loop[policy][:, -1], [5, 95])
    print(f"{policy:<9} share of units in B: month 1 {loop[policy][:, 0].mean():.2f}, "
          f"month 24 {loop[policy][:, -1].mean():.2f} [{q[0]:.2f}, {q[1]:.2f}];  "
          f"B's share of recorded near-misses {data_share:.2f}")
print("True share of near-miss risk in B: 0.50")

# ---------- Accountability artifact: a model card written from the audit ----------
a = audit.set_index(["policy", "group"])
sp = {n: spearmanr(s, city.true_rate).statistic for n, (_, s) in policies.items()}
risk = {n: city.true_rate[sel].sum() for n, (sel, _) in policies.items()}
card = [
    ("Model", "Intersection injury-risk screening; Poisson gradient boosting"),
    ("Intended use", "Rank intersections for engineering review; not for enforcement"),
    ("Training data", f"{T} years of police-reported injury crashes, {N} intersections"),
    ("Known data issue", f"Linkage audit reporting rates A {rho_hat['A']:.2f}, "
                         f"B {rho_hat['B']:.2f} ({n_link} records each)"),
    ("Mitigation", "Counts divided by each group's reporting rate before training"),
    ("Ranking quality", f"Spearman with true risk {sp['Baseline']:.2f} -> "
                        f"{sp['Corrected labels']:.2f}"),
    ("Equal opportunity", f"TPR A/B {a.loc[('Baseline', 'A'), 'TPR']:.2f}/"
                          f"{a.loc[('Baseline', 'B'), 'TPR']:.2f} -> "
                          f"{a.loc[('Corrected labels', 'A'), 'TPR']:.2f}/"
                          f"{a.loc[('Corrected labels', 'B'), 'TPR']:.2f}"),
    ("Calibration", f"predicted/true A/B {a.loc[('Baseline', 'A'), 'pred/true']:.2f}/"
                    f"{a.loc[('Baseline', 'B'), 'pred/true']:.2f} -> "
                    f"{a.loc[('Corrected labels', 'A'), 'pred/true']:.2f}/"
                    f"{a.loc[('Corrected labels', 'B'), 'pred/true']:.2f}"),
    ("Risk addressed", f"{risk['Baseline']:.0f} -> {risk['Corrected labels']:.0f} true "
                       f"injury crashes/yr at selected sites"),
    ("Human oversight", "Engineer reviews each flagged site; overrides logged"),
    ("Monitoring", "Linkage audit yearly; subgroup metrics each release; incident log"),
    ("Limitations", "One linkage audit; unmeasured site factors; one city only")]
width = max(len(k) for k, _ in card)
print("MODEL CARD (generated from the audit)")
for key, value in card:
    print(f"  {key:<{width}}  {value}")

# ---------- Figure: calibration, equal opportunity, and the feedback loop ----------
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
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "legend.frameon": False})
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.45))
edges = np.quantile(city.true_rate, np.linspace(0, 1, 9))              # (a) calibration
for g, m, c in [("A", ~isB, SAGE), ("B", isB, RUST)]:
    b = np.clip(np.digitize(city.true_rate[m], edges[1:-1]), 0, 7)
    t = pd.Series(city.true_rate[m].to_numpy()).groupby(b).mean()
    for score, mk, lab in [(pred_base, "o", "baseline"), (pred_corr, "s", "corrected")]:
        p = pd.Series(score[m]).groupby(b).mean()
        ax[0].plot(t, p, mk + ("-" if lab == "corrected" else "--"), color=c, ms=3,
                   mfc=c if lab == "corrected" else PAPER, label=f"{g}, {lab}")
lim = [0, city.true_rate.quantile(0.98)]
ax[0].plot(lim, lim, ":", color=INK, lw=0.6)
ax[0].set(xlabel="True injury crashes per year", ylabel="Predicted",
          title="(a) Calibration by community")
ax[0].legend(fontsize=6.5, loc="upper left")
# (b) equal opportunity
names = list(policies); x = np.arange(len(names))
for k, (g, c) in enumerate([("A", SAGE), ("B", RUST)]):
    tpr = [audit[(audit.policy == n) & (audit.group == g)].TPR.iloc[0] for n in names]
    ppv = [audit[(audit.policy == n) & (audit.group == g)].PPV.iloc[0] for n in names]
    ax[1].bar(x + (k - 0.5) * 0.36, tpr, 0.34, color=c, label=f"TPR, {g}")
    ax[1].plot(x + (k - 0.5) * 0.36, ppv, "D", color=INK, mfc=GOLD, ms=3.6,
               label="PPV" if k == 0 else None)
ax[1].set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=6.5)
ax[1].set(ylim=(0, 1.15), ylabel="Rate", title="(b) True-positive rate and PPV")
ax[1].legend(fontsize=6.5, loc="upper center", ncol=3)
months = np.arange(1, MONTHS + 1)                                      # (c) feedback loop
for policy, c in [("naive", RUST), ("per-unit", SAGE)]:
    lo, hi = np.percentile(loop[policy], [5, 95], axis=0)
    ax[2].fill_between(months, lo, hi, color=c, alpha=0.25)
    ax[2].plot(months, loop[policy].mean(0), color=c, lw=1.2,
               label="by detections" if policy == "naive"
               else "by detections per unit")
ax[2].axhline(0.5, color=INK, ls=":", lw=0.6)
ax[2].text(24.5, 0.52, "true share of risk", fontsize=6.5, ha="right")
ax[2].set(xlabel="Month", ylabel="Share of units in B", ylim=(0, 0.7),
          title="(c) Feedback locks in history")
ax[2].legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig("chap07_fig02_bias_audit.png", dpi=300)
