"""Chapter 4 - Case Study 1: Glance Management and Usability Testing for In-Vehicle HMIs.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/04a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch04` or `python scripts/run_all.py --chapter 4`;
outputs are written to outputs/ch04/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import betaln

# ---------- Three designs for entering a destination: their glances ----------
rng = np.random.default_rng(4)
# mean glances, median glance (s), log-SD
DESIGNS = {"full keyboard": (9.0, 1.30, 0.45),
           "predictive list": (5.0, 1.10, 0.40),
           "voice + confirm": (2.0, 0.85, 0.35)}
N, SD_DRIVER = 50_000, 0.15                          # drivers; between-driver log-SD


def drivers(design, n=N):
    lam, med, sd = DESIGNS[design]
    k = np.maximum(1, rng.poisson(lam, n))
    style = rng.normal(0, SD_DRIVER, n)
    g = [med * np.exp(style[i] + sd * rng.normal(size=k[i])) for i in range(n)]
    return pd.DataFrame({"glances": k, "teort": [x.sum() for x in g],
                         "longest": [x.max() for x in g],
                         "long": [(x > 2).sum() for x in g]})


pop = {d: drivers(d) for d in DESIGNS}
rows, comp = [], []
for d, t in pop.items():
    c1 = t.teort / t.glances <= 2                    # mean glance at most 2 s
    c2 = t.long / t.glances <= 0.15                  # at most 15% of glances over 2 s
    c3 = t.teort <= 12                               # total eyes-off-road at most 12 s
    rows.append((d, t.teort.mean(), (t.teort > 12).mean(), (t.longest > 2).mean()))
    comp.append((d, c1.mean(), c2.mean(), c3.mean(), (c1 & c2 & c3).mean()))
print(pd.DataFrame(rows, columns=["design", "mean_TEORT_s", "P_TEORT_over_12",
                                  "P_any_over_2s"]).round(4).to_string(index=False))
print("\nShare of drivers meeting each NHTSA criterion (85% required):")
print(pd.DataFrame(comp, columns=["design", "mean_glance", "long_share", "total_12s",
                                  "all_three"]).round(4).to_string(index=False))

# ---------- Cross-checks for a fixed number of glances and no driver effect ----------
chk = []
for d, (lam, med, sd) in DESIGNS.items():
    k = int(lam)
    g = med * np.exp(sd * rng.normal(size=(400_000, k)))
    p2 = 1 - stats.lognorm.cdf(2, sd, scale=med) ** k        # P(longest glance > 2 s)
    m1 = k * med * np.exp(sd ** 2 / 2)                       # Fenton-Wilkinson moments
    v1 = k * med ** 2 * np.exp(sd ** 2) * (np.exp(sd ** 2) - 1)
    s2 = np.log(1 + v1 / m1 ** 2)
    fw = stats.lognorm.sf(12, np.sqrt(s2), scale=m1 * np.exp(-s2 / 2))
    chk.append((d, k, p2, (g.max(axis=1) > 2).mean(), fw, (g.sum(axis=1) > 12).mean()))
print("\nFixed glance count: closed form vs simulation (400,000 tasks)")
chk_df = pd.DataFrame(chk, columns=["design", "k", "P_long_closed", "P_long_sim",
                                    "P_TEORT12_FW", "P_TEORT12_sim"])
print(chk_df.round(4).to_string(index=False))

# ---------- The occlusion test (ISO 16673): 1.5 s open, 1.5 s closed ----------
OPEN, RESUME = 1.5, 0.3                      # open interval (s); cost of re-orienting (s)


def tsot_closed(work):
    return OPEN * np.ceil(work / (OPEN - RESUME))


def tsot_stepwise(work):                     # second implementation: shutter by shutter
    done, opens = 0.0, 0
    while done < work - 1e-12:
        opens += 1
        done += OPEN - RESUME
    return OPEN * opens


occ = []
for d, t in pop.items():
    work = t.teort.to_numpy()[:2000]         # visual work for 2,000 drivers
    a = tsot_closed(work)
    b = np.array([tsot_stepwise(w) for w in work])
    occ.append((d, np.median(a), (a <= 12).mean(), np.abs(a - b).max()))
occ = pd.DataFrame(occ, columns=["design", "median_TSOT_s", "share_TSOT_le_12",
                                 "max_diff_methods"])
print(occ.round(4).to_string(index=False))

# ---------- How many participants does a usability test need? ----------
# problem detectability ~ beta(0.6, 1.4), mean 0.3
A, B, PROBLEMS = 0.6, 1.4, 30


def found_heterogeneous(n):                  # E[1 - (1 - p)^n] for p ~ beta(A, B)
    return 1 - np.exp(betaln(A, B + n) - betaln(A, B))


def found_homogeneous(n, p=A / (A + B)):
    return 1 - (1 - p) ** n


ns = [1, 3, 5, 8, 10, 15, 20, 30]
sim = []
for n in ns:
    reps = rng.beta(A, B, (20_000, PROBLEMS))   # a fresh product's problems each time
    seen = rng.random((20_000, PROBLEMS, n)) < reps[:, :, None]
    sim.append(seen.any(axis=2).mean())
disc = pd.DataFrame({"participants": ns,
                     "homogeneous_p": [found_homogeneous(n) for n in ns],
                     "beta_closed": [found_heterogeneous(n) for n in ns], "beta_sim": sim})
print(disc.round(4).to_string(index=False))
for target in [0.80, 0.90]:
    n_h = next(n for n in range(1, 500) if found_homogeneous(n) >= target)
    n_b = next(n for n in range(1, 500) if found_heterogeneous(n) >= target)
    print(f"Participants to find {target:.0%} of problems: {n_h} if all are equally easy "
          f"to find,\n  {n_b} if detectability varies")

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
col = {"full keyboard": RUST, "predictive list": GOLD, "voice + confirm": SAGE}
a = ax[0, 0]                                                   # (a) total eyes-off-road
bins = np.linspace(0, 30, 61)
for d, t in pop.items():
    a.hist(t.teort, bins, density=True, histtype="step", lw=1.3, color=col[d],
           label=f"{d}: {(t.teort > 12).mean():.1%} over 12 s")
a.axvline(12, ymax=0.62, color=INK, ls="--", lw=0.7)
a.set(xlabel="Total eyes-off-road time per task (s)", ylabel="Density",
      title="(a) Visual demand of three designs")
a.legend(fontsize=6.5, loc="upper right")
a = ax[0, 1]                                                   # (b) long-glance risk
kk = np.arange(1, 13)
for med, sd, c in [(1.30, 0.45, RUST), (1.10, 0.40, GOLD), (0.85, 0.35, SAGE)]:
    a.plot(kk, 1 - stats.lognorm.cdf(2, sd, scale=med) ** kk, color=c, lw=1.4,
           label=f"median glance {med:.2f} s")
for (d, k, p_c, p_s, _, _) in chk:
    a.plot(k, p_s, "o", color=INK, ms=3.9)
a.plot([], [], "o", color=INK, ms=3.9, label="simulated designs")
a.set(xlabel="Glances per task", ylabel="P(at least one glance over 2 s)", ylim=(0, 1),
      title="(b) Every extra glance is another chance")
a.legend(fontsize=6.5, loc="upper left")
a = ax[1, 0]                                                   # (c) occlusion test
data = [tsot_closed(pop[d].teort.to_numpy()[:2000]) for d in DESIGNS]
bp = a.boxplot(data, patch_artist=True, widths=0.5, showfliers=False,
               medianprops=dict(color=INK, lw=1))
for patch, d in zip(bp["boxes"], DESIGNS):
    patch.set(facecolor=col[d], alpha=0.6)
a.axhline(12, color=INK, ls="--", lw=0.7)
a.set_xticks([1, 2, 3], list(DESIGNS), fontsize=6.5)
a.set(ylabel="Total shutter-open time (s)",
      title="(c) Occlusion test, 1.5 s open / 1.5 s closed")
a = ax[1, 1]                                                   # (d) usability testing
nn = np.arange(1, 41)
a.plot(nn, [found_homogeneous(n) for n in nn], color=GOLD, lw=1.4,
       label="every problem equally easy to find (p = 0.3)")
a.plot(nn, [found_heterogeneous(n) for n in nn], color=SAGE, lw=1.4,
       label="detectability varies, beta(0.6, 1.4)")
a.plot(disc.participants, disc.beta_sim, "o", color=INK, ms=3.2, label="simulated tests")
a.axvline(5, color=RUST, ls=":", lw=0.9)
a.text(5.8, 0.45, "five participants", fontsize=6.5, color=RUST)
a.set(xlabel="Participants in the usability test", ylabel="Share of problems found",
      ylim=(0, 1.02), title="(d) Five participants find less than promised")
a.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig("chap04_fig01_glance_usability.png", dpi=300)
