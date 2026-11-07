"""Chapter 11 - Case Study 2: AR/VR Training Environments for Safety Evaluation.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/11a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch11` or `python scripts/run_all.py --chapter 11`;
outputs are written to outputs/ch11/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from scipy.optimize import brentq, curve_fit
import statsmodels.formula.api as smf

# ---------- A randomized trial of VR hazard-anticipation training ----------
rng = np.random.default_rng(111)
n = 120
vr = np.repeat([1, 0], 60)                     # 60 trained in VR, 60 controls
base = rng.normal(40, 10, n)                   # baseline: % of hazards anticipated
ssq = np.exp(rng.normal(3.0 - 0.02 * (base - 40), 0.6, n))    # simulator sickness score
p_drop = vr * expit(-3.2 + 0.06 * ssq)         # trainees may stop training when sick
drop = rng.random(n) < p_drop
EFFECT, PARTIAL, PRACTICE, TRANSFER = 15.0, 0.3, 3.0, 0.7
dose = vr * np.where(drop, PARTIAL, 1.0)       # share of the training each person received
post = base + PRACTICE + EFFECT * dose + rng.normal(0, 8, n)              # VR test
road = base + PRACTICE + TRANSFER * EFFECT * dose + rng.normal(0, 8, n)   # on-road test
df = pd.DataFrame({"vr": vr, "base": base, "ssq": ssq, "drop": drop, "post": post,
                   "road": road})
itt_true = EFFECT * (1 - (1 - PARTIAL) * p_drop[vr == 1].mean())
print(f"Trainees who stopped: {drop.sum()} of 60; mean sickness score "
      f"{ssq[drop].mean():.1f} (stopped) vs "
      f"{ssq[(vr == 1) & ~drop].mean():.1f} (completed)")
print(f"Mean baseline: {base[drop].mean():.1f} (stopped) vs "
      f"{base[(vr == 1) & ~drop].mean():.1f} (completed)")


def ancova(d, y):
    f = smf.ols(f"{y} ~ vr + base", d).fit()
    return f.params.vr, *f.conf_int().loc["vr"]


# Frisch-Waugh-Lovell: same coefficient by hand
def fwl(d, y):
    ry, rx = d[y] - np.polyval(np.polyfit(d.base, d[y], 1), d.base), \
        d.vr - np.polyval(np.polyfit(d.base, d.vr, 1), d.base)
    return (rx * ry).sum() / (rx ** 2).sum()


truth = {("intention to treat", "post"): itt_true, ("intention to treat", "road"):
         TRANSFER * itt_true, ("per protocol", "post"): EFFECT,
         ("per protocol", "road"): TRANSFER * EFFECT}
rows = []
for label, d in [("intention to treat", df), ("per protocol", df[~df["drop"]])]:
    for y in ["post", "road"]:
        est, lo, hi = ancova(d, y)
        rows.append((label, y, len(d), est, lo, hi, fwl(d, y), truth[(label, y)]))
res = pd.DataFrame(rows, columns=["analysis", "test", "n", "effect", "lo", "hi",
                                  "effect_by_hand", "true_value"])
print(res.round(3).to_string(index=False))

# ---------- From VR to the road: transfer effectiveness (Roscoe) and retention ----------
S_INF, TAU, CRIT, X, KAPPA = 85.0, 4.0, 70.0, 2.0, 1.5   # ceiling, h, criterion, VR h, h
COST = 0.25                                    # cost of a VR hour / cost of a road hour
s0 = base.mean() + PRACTICE                    # untrained drivers' starting road score
g_road = res[(res.analysis == "intention to treat") & (res.test == "road")].effect.iloc[0]


# s(h) = S_INF - (S_INF - start) exp(-h / TAU)
def hours_closed(start):
    return TAU * np.log((S_INF - start) / (S_INF - CRIT))


def hours_root(start):
    return brentq(lambda h: S_INF - (S_INF - start) * np.exp(-h / TAU) - CRIT, 0, 200)


h_c, h_x = hours_closed(s0), hours_closed(s0 + g_road)
print(f"Road hours to criterion: control {h_c:.4f} (root {hours_root(s0):.4f}); "
      f"trained {h_x:.4f} (root {hours_root(s0 + g_road):.4f})")
print(f"Transfer effectiveness ratio for {X:.0f} VR hours: TER = {(h_c - h_x) / X:.4f}")
G = g_road / (1 - np.exp(-X / KAPPA))          # gain ceiling consistent with the trial


def saved(x):                                  # road hours saved by x hours in VR
    return h_c - hours_closed(s0 + G * (1 - np.exp(-x / KAPPA)))


def iter_closed(x):                            # incremental TER, derivative of saved(x)
    g = G * (1 - np.exp(-x / KAPPA))
    return TAU * (G / KAPPA) * np.exp(-x / KAPPA) / (S_INF - s0 - g)


for x in [0.5, 1.0, 2.0, 4.0]:
    num = (saved(x + 1e-5) - saved(x - 1e-5)) / 2e-5
    print(f"  incremental TER at {x:.1f} h: {iter_closed(x):.4f} (numerical {num:.4f})")
x1 = brentq(lambda x: iter_closed(x) - COST, 0.0, 20)
print(f"An extra VR hour pays for itself while ITER > {COST}: up to {x1:.2f} VR hours")

# ---------- Retention: the trained advantage decays (true half-life 4 months) ----------
HALF, months = 4.0, np.array([0, 1, 3, 6])
keep = (vr == 1) & ~drop


def half_lives(t, a):                          # two estimators of the same half-life
    k1 = -np.polyfit(t, np.log(a), 1)[0]                       # log-linear least squares
    (g, k2), _ = curve_fit(lambda x, g, k: g * np.exp(-k * x), t, a, p0=(15, 0.1))
    return np.log(2) / k1, np.log(2) / k2, g, k2


exact = EFFECT * np.exp(-np.log(2) / HALF * months)
print("Noise-free check, half-life (log-linear, nonlinear): %.4f, %.4f"
      % half_lives(months, exact)[:2])
adv = []
for t in months:                               # gain over each driver's own baseline
    tr = PRACTICE + EFFECT * np.exp(-np.log(2) / HALF * t) + rng.normal(0, 6, keep.sum())
    co = PRACTICE + rng.normal(0, 6, (vr == 0).sum())
    adv.append(tr.mean() - co.mean())
adv = np.array(adv)
h_log, h_nls, g0, k_nls = half_lives(months, adv)
k_log = np.log(2) / h_log
by_month = ", ".join(f"{m}: {a:.2f}" for m, a in zip(months, adv))
print("Trained advantage by month:", by_month)
print(f"Half-life: {h_log:.2f} months (log-linear), {h_nls:.2f} (nonlinear), "
      f"true {HALF:.1f}; "
      f"\nadvantage left at 12 months {g0 * np.exp(-12 * k_nls):.2f} "
      f"(true {EFFECT * 2 ** (-12 / HALF):.2f})")

# ---------- Figure ----------
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
fig, ax = plt.subplots(2, 2, figsize=(6.2, 4.8))
a = ax[0, 0]                                                   # (a) two estimands
for k, (lab, col) in enumerate([("intention to treat", SAGE), ("per protocol", GOLD)]):
    r = res[res.analysis == lab]
    x = np.arange(2) + (k - 0.5) * 0.3
    a.errorbar(x, r.effect, yerr=[r.effect - r.lo, r.hi - r.effect], fmt="o", color=col,
               mec=INK, ms=4.5, capsize=1.9, ecolor=INK, label=lab)
    a.plot(x, r.true_value, "_", color=RUST, ms=14.2, mew=1.6)
a.plot([], [], "_", color=RUST, ms=9, mew=1.6, label="true value")
a.set_xticks([0, 1], ["VR hazard test", "on-road test"])
a.set(ylabel="Training effect (percentage points)", ylim=(0, 20),
      title="(a) Effect of offering training vs. of completing it")
a.legend(fontsize=6.5, loc="upper right")
a = ax[0, 1]                                                   # (b) sickness and dropout
ss = np.linspace(0, 120, 300)
a.plot(ss, expit(-3.2 + 0.06 * ss), color=RUST, lw=1.4, label="probability of stopping")
t = vr == 1
a.plot(ssq[t & ~drop], np.full((t & ~drop).sum(), -0.04), "|", color=SAGE, ms=7.8,
       label="trainees who completed")
a.plot(ssq[t & drop], np.full((t & drop).sum(), -0.09), "|", color=RUST, ms=7.8,
       label="trainees who stopped")
a.set(xlim=(0, 120), ylim=(-0.13, 1), xlabel="Simulator sickness score (SSQ total)",
      ylabel="Probability of stopping training", title="(b) Sickness drives dropout")
a.legend(fontsize=6.5, loc="upper left")
a = ax[1, 0]                                                   # (c) road learning curves
hh = np.linspace(0, 8, 300)
for start, col, lab, h in [(s0, RUST, "no VR training", h_c),
                           (s0 + g_road, SAGE, f"after {X:.0f} h of VR", h_x)]:
    a.plot(hh, S_INF - (S_INF - start) * np.exp(-hh / TAU), color=col, lw=1.4, label=lab)
    a.plot(h, CRIT, "o", color=col, mec=INK, ms=4.5)
a.axhline(CRIT, color=INK, ls="--", lw=0.6)
a.annotate("", (h_x, CRIT), (h_c, CRIT),
           arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
a.text(h_c + 0.25, CRIT - 3.5, f"{h_c - h_x:.2f} road hours saved", ha="left", va="top",
       fontsize=6.5)
a.set(xlabel="Supervised road practice (h)", ylabel="Hazards anticipated (%)",
      title=f"(c) Transfer: TER = {(h_c - h_x) / X:.2f} for {X:.0f} VR hours")
a.legend(fontsize=6.5, loc="lower right")
a = ax[1, 1]                                                   # (d) retention
tt = np.linspace(0, 12, 200)
a.plot(months, adv, "o", color=INK, ms=4.5, label="trained minus control, by month")
a.plot(tt, np.exp(np.polyfit(months, np.log(adv), 1)[1]) * np.exp(-k_log * tt), color=GOLD,
       lw=1.3, label=f"log-linear fit: half-life {np.log(2) / k_log:.1f} months")
a.plot(tt, g0 * np.exp(-k_nls * tt), color=SAGE, lw=1.3, ls="--",
       label=f"nonlinear fit: half-life {np.log(2) / k_nls:.1f} months")
a.plot(tt, EFFECT * np.exp(-np.log(2) / HALF * tt), color=RUST, lw=0.8, ls=":",
       label=f"true decay: half-life {HALF:.0f} months")
a.set(xlabel="Months after training", ylabel="Advantage of trained drivers (points)",
      ylim=(0, 20), title="(d) Skills decay without practice")
a.legend(fontsize=6.5, loc="upper right")
fig.tight_layout(); fig.savefig("chap11_fig02_vr_training.png", dpi=300)
