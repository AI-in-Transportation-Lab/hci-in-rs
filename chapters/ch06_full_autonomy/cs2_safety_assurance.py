"""Chapter 6 - Case Study 2: Safety Assurance and Standardization for SAE Level 4-5 Deployment.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/06a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch06` or `python scripts/run_all.py --chapter 6`;
outputs are written to outputs/ch06/.
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Synthetic record: rider-only fleet vs. ODD-matched benchmark ----------
rng = np.random.default_rng(23)
od = pd.DataFrame({
    "stratum":   ["Arterial, day", "Arterial, night", "Local, day", "Local, night"],
    "av_mi":     [7.0, 2.0, 7.5, 1.5],        # fleet rider-only miles (millions), 18 months
    "hum_mi":    [2000, 1100, 1500, 900],     # human VMT in the same zone (millions)
    "hum_rate":  [2.6, 3.9, 1.5, 2.9],        # benchmark injury crashes per million miles
    # fleet / human rate ratio (unknown to reviewer)
    "true_rr":   [0.40, 0.50, 0.55, 0.90],
    "expand":    [0.25, 0.15, 0.35, 0.25]})   # mileage mix of the proposed expansion area
od["av_crash"] = rng.poisson(od.av_mi * od.hum_rate * od.true_rr)

def exact_rate_ci(k, m, conf=0.95):               # Garwood interval for a Poisson rate
    a = (1 - conf) / 2
    lo = stats.chi2.ppf(a, 2 * k) / 2 / m if k > 0 else 0.0
    return lo, stats.chi2.ppf(1 - a, 2 * k + 2) / 2 / m

od["av_rate"] = od.av_crash / od.av_mi
od[["lo", "hi"]] = [exact_rate_ci(k, m) for k, m in zip(od.av_crash, od.av_mi)]
od["rr"], od["rr_lo"], od["rr_hi"] = (od.av_rate / od.hum_rate, od.lo / od.hum_rate,
                                      od.hi / od.hum_rate)
print(od[["stratum", "av_mi", "av_crash", "av_rate", "hum_rate", "rr", "rr_lo",
          "rr_hi"]].round(2).to_string(index=False))

# ---------- Three ways to summarize the same record ----------
K, M = od.av_crash.sum(), od.av_mi.sum()
hum_all = (od.hum_mi * od.hum_rate).sum() / od.hum_mi.sum()
crude = (K / M) / hum_all                                 # ignores the fleet's ODD mix
# expected crashes, humans on fleet mix
E = (od.av_mi * od.hum_rate).sum()
sir, (s_lo, s_hi) = K / E, exact_rate_ci(K, E)
w_city = od.hum_mi / od.hum_mi.sum()
dsr = (w_city * od.av_rate).sum() / (w_city * od.hum_rate).sum()
exp_rr = (od.expand * od.av_rate).sum() / (od.expand * od.hum_rate).sum()
print(f"\nFleet: {K} injury crashes in {M:.1f} million miles ({K / M:.2f} per million)")
print(f"Crude rate ratio (fleet vs. all human driving)        {crude:.2f}")
print(f"Standardized ratio, fleet mix (observed/expected)     {sir:.2f}"
      f"  [{s_lo:.2f}, {s_hi:.2f}]")
print(f"Directly standardized ratio, city-wide mix            {dsr:.2f}")
print(f"Projected ratio in the expansion area mix             {exp_rr:.2f}")
print(f"Share of fleet miles in Local, night                  {od.av_mi.iloc[3] / M:.1%}"
      f"   (expansion: {od.expand.iloc[3]:.0%})")

# ---------- Scenario coverage: interaction classes x ODD context ----------
# interaction class: (risk tier, tests per cell in Arterial-day, true failure prob.)
INTER = {
    "Pedestrian, crosswalk": ("high", 2400, 0.0008),
    "Pedestrian, midblock":  ("high", 1500, 0.0015),
    "Cyclist, bike lane":    ("med",  1800, 0.0010),
    "Emergency vehicle":     ("high",  700, 0.0020),
    "Police hand signals":   ("med",   260, 0.0040),
    "Work-zone flagger":     ("med",   340, 0.0030),
    "School bus, stop arm":  ("high",  420, 0.0020),
    "Remote-assist handoff": ("med",   380, 0.0020)}
CTX = dict(zip(od.stratum, [1.0, 0.6, 0.9, 0.45]))       # relative test effort by context
# tolerable failure prob. per encounter
P_MAX = {"high": 0.005, "med": 0.010}
CONF = 0.95

# success-run theorem, zero failures
def n_required(p_max, conf=CONF):
    return int(np.ceil(np.log(1 - conf) / np.log(1 - p_max)))

def upper_bound(f, n, conf=CONF):                        # one-sided Clopper-Pearson bound
    return stats.beta.ppf(conf, f + 1, n - f) if n > f else 1.0

cells = []
for name, (tier, base, p_true) in INTER.items():
    for ctx, effort in CTX.items():
        n = int(round(base * effort * np.exp(rng.normal(0, 0.15))))
        f = rng.binomial(n, p_true * (1.5 if "night" in ctx else 1.0))
        ub = upper_bound(f, n)
        extra = 0
        # further failure-free tests needed
        while upper_bound(f, n + extra) > P_MAX[tier]:
            extra += 10
        w = (2.0 if tier == "high" else 1.0) * od.set_index("stratum").expand[ctx]
        cells.append((name, ctx, tier, n, f, ub, ub <= P_MAX[tier], extra, w))
cv = pd.DataFrame(cells, columns=["inter", "ctx", "tier", "n", "fails", "ub", "pass",
                                  "extra", "weight"])
print("Tests required for zero failures:",
      {t: n_required(p) for t, p in P_MAX.items()})
print("\nCoverage by interaction class (4 ODD contexts each):")
print(cv.groupby("inter", sort=False).agg(tier=("tier", "first"), tests=("n", "sum"),
      failures=("fails", "sum"), cells_passing=("pass", "sum"),
      extra_tests=("extra", "sum")).to_string())
print(f"\nCells passing: {cv['pass'].sum()} of {len(cv)};  "
      f"risk-weighted coverage: {(cv.weight * cv['pass']).sum() / cv.weight.sum():.1%}")
gaps = cv[~cv["pass"]].sort_values("weight", ascending=False)
print("Largest risk-weighted gaps:")
for _, r in gaps.head(5).iterrows():
    print(f"  {r.inter:<23}{r.ctx:<17}n={r.n:<5d}fails={r.fails}  upper={r.ub:.4f}"
          f"  +{r.extra} tests")

# ---------- Post-release monitoring: Gamma-Poisson with a power prior ----------
# benchmark crashes per million miles, new area
r_exp = (od.expand * od.hum_rate).sum()
MILES_MO, MONTHS = 0.8, 18                       # fleet miles per month (millions), horizon
# human-equivalent expected crashes per month
e_mo = MILES_MO * r_exp
# declare if P(theta<1) >= 0.95 after month 6;
DEMO, HOLD, MIN_MO = 0.95, 0.50, 6
                                                 # hold at once if P(theta>1) >= 0.50
def p_below_one(a0, crashes):
    # power prior discounts pre-release evidence
    a = 0.5 + a0 * K + np.cumsum(crashes)
    b = a0 * E + e_mo * np.arange(1, len(crashes) + 1)
    return stats.gamma.cdf(1.0, a, scale=1 / b)

def first(mask):
    idx = np.flatnonzero(mask)
    return idx[0] + 1 if idx.size else np.nan

print(f"Benchmark in expansion mix: {r_exp:.3f} per million miles;"
      f" expected human-equivalent crashes per month: {e_mo:.2f}")
for a0 in [0.1, 1.0]:
    a, b = 0.5 + a0 * K, a0 * E
    print(f"  a0 = {a0:3.1f}: prior mean {a / b:.2f}, prior P(theta < 1) = "
          f"{stats.gamma.cdf(1, a, scale=1 / b):.3f}")

sim = np.random.default_rng(7)
print(f"\nSequential decisions over {MONTHS} months (4,000 simulated deployments each):")
print(f"  {'Scenario':<25}{'a0':>4}{'hold first':>12}{'declare first':>15}"
      f"{'undecided':>11}{'median month':>14}")
paths = {}
for label, theta in [("A: as projected (0.62)", 0.62), ("B: regression (1.25)", 1.25)]:
    draws = sim.poisson(theta * e_mo, size=(4000, MONTHS))
    for a0 in [0.1, 1.0]:
        P = np.array([p_below_one(a0, d) for d in draws])
        month = np.arange(1, MONTHS + 1)
        t_hold = np.array([first(p <= 1 - HOLD) for p in P])
        t_demo = np.array([first((p >= DEMO) & (month >= MIN_MO)) for p in P])
        hold = t_hold < np.nan_to_num(t_demo, nan=99)          # first boundary crossed wins
        demo = t_demo < np.nan_to_num(t_hold, nan=99)
        t_dec = np.where(hold, t_hold, np.where(demo, t_demo, np.nan))
        print(f"  {label:<25}{a0:>4.1f}{hold.mean():>12.3f}{demo.mean():>15.3f}"
              f"{1 - hold.mean() - demo.mean():>11.3f}{np.nanmedian(t_dec):>14.0f}")
        # median path
        paths[(label, a0)] = P[np.argsort(np.nan_to_num(t_dec, nan=99))[2000]]

# ---------- Figure: rate ratios, monitoring paths, scenario coverage ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import textwrap
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
fig = plt.figure(figsize=(6.2, 5.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

labels = list(od.stratum) + ["Crude (all human driving)", "Standardized, fleet mix",
                             "Standardized, city mix", "Projected, expansion mix"]
vals = list(od.rr) + [crude, sir, dsr, exp_rr]
y = np.arange(len(labels))[::-1]
for i, (v, yy) in enumerate(zip(vals, y)):                  # (a) forest plot
    if i < 4:
        ax1.plot([od.rr_lo[i], od.rr_hi[i]], [yy, yy], color=SAGE, lw=1.4)
        ax1.plot(v, yy, "o", color=GOLD, mec=INK, ms=4.9)
    else:
        if i == 5:
            ax1.plot([s_lo, s_hi], [yy, yy], color=RUST, lw=1.4)
        ax1.plot(v, yy, "D", color=RUST, mec=INK, ms=4.9)
ax1.axvline(1, color=OLIVE, ls="--", lw=0.8)
ax1.set_yticks(y, labels, fontsize=6.5)
ax1.set(xlabel="Fleet / human injury-crash rate ratio (95% CI)", xlim=(0, 2.5),
        title="(a) Same record, four comparisons")

for (label, a0), P in paths.items():                        # (b) monitoring paths
    ax2.plot(np.arange(1, MONTHS + 1), P, "-" if a0 < 1 else "--", lw=1.3,
             color=SAGE if label.startswith("A") else RUST,
             label=f"{label.split(':')[0]}, a0 = {a0:.1f}")
ax2.axhline(DEMO, color=GOLD, lw=0.8); ax2.axhline(1 - HOLD, color=INK, lw=0.7, ls=":")
ax2.text(18, DEMO - 0.05, "declare (after month 6)", fontsize=6.5, color=INK, ha="right")
ax2.text(4.2, 1 - HOLD + 0.02, "hold", fontsize=6.5, color=INK)
ax2.text(9.5, 1.04, "a0 = 1.0: A and B look alike", fontsize=6.5, color=RUST, ha="center")
ax2.set(xlabel="Months after release", ylabel="P(fleet rate < benchmark)", ylim=(0, 1.12),
        xticks=[1, 3, 6, 9, 12, 15, 18],
        title="(b) Posterior monitoring in the expansion area")
ax2.legend(fontsize=6.5, loc="upper right", bbox_to_anchor=(1, 0.77), ncol=2)

inter, ctx = list(INTER), list(CTX)                         # (c) coverage heatmap
ratio = np.array([[cv[(cv.inter == i) & (cv.ctx == c)].ub.iloc[0] /
                   P_MAX[INTER[i][0]] for i in inter] for c in ctx])
cmap = LinearSegmentedColormap.from_list("cov", [(0, SAGE), (0.333, "#DCE5E1"),
                                         (0.334, "#F6E7A6"), (0.667, GOLD), (1, RUST)])
im = ax3.imshow(np.clip(ratio, 0, 3), cmap=cmap, vmin=0, vmax=3, aspect="auto")
for r, c in enumerate(ctx):
    for k, i in enumerate(inter):
        s = cv[(cv.inter == i) & (cv.ctx == c)].iloc[0]
        ax3.text(k, r, f"n={s.n}\nf={s.fails}" + ("" if s["pass"] else "\nGAP"),
                 ha="center", va="center", fontsize=6, color=INK,
                 fontweight="normal" if s["pass"] else "bold")
ax3.set_xticks(range(len(inter)), [textwrap.fill(i, 11) for i in inter], fontsize=6)
ax3.set_yticks(range(len(ctx)), ctx, fontsize=6.5)
ax3.grid(False)
ax3.set_title("(c) Scenario coverage: upper failure bound / tolerable rate (GAP if > 1)")
cb = fig.colorbar(im, ax=ax3, fraction=0.025, pad=0.01)
cb.set_label("Bound / tolerable rate", fontsize=6.5)
fig.savefig("chap06_fig02_safety_assurance.png", dpi=300, bbox_inches="tight")
