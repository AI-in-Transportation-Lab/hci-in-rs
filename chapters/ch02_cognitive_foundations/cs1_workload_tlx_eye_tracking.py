"""Chapter 2 - Case Study 1: Measuring Driver Workload Using NASA-TLX and Eye Tracking.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: code/ch02_case1_workload.py (shown in chapters/02a.tex).
Run from the repository root with `make ch02` or `python scripts/run_all.py --chapter 2`;
outputs are written to outputs/ch02/.

Case Study 1 -- Measuring driver workload with NASA-TLX and eye tracking.

A within-subject simulator study of a freeway work-zone approach, run on
synthetic data generated from the assumptions stated in Table 2.5, so that
every number below can be checked against a known generating process.

The question is not "is the revised interface better?" -- the generator was
built so that it is -- but "which analysis would have detected it, and which
would have missed it?"
"""
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(20260212)

N_DRIVERS = 32
SEGMENTS = ["open freeway", "advance warning", "taper", "merge"]
CONDITIONS = ["baseline HMI", "revised HMI"]

# Segment-level temporal demand added to a driver's own baseline (TLX points).
SEG_DEMAND = {"open freeway": 0.0, "advance warning": 8.0,
              "taper": 31.0, "merge": 24.0}
# The revised interface removes demand only where it was designed to.
REVISED_RELIEF = {"open freeway": 0.0, "advance warning": 2.0,
                  "taper": 14.0, "merge": 9.0}
# Mean longest single off-road glance (s) under the baseline interface.
SEG_GLANCE = {"open freeway": 1.05, "advance warning": 1.35,
              "taper": 2.25, "merge": 1.80}
GLANCE_RELIEF = {"open freeway": 0.00, "advance warning": 0.10,
                 "taper": 0.62, "merge": 0.33}

SUBSCALES = ["mental", "physical", "temporal", "performance", "effort",
             "frustration"]


def simulate():
    """One row per driver x condition x segment."""
    driver_offset = RNG.normal(0, 7.0, N_DRIVERS)       # trait workload
    driver_glance = RNG.normal(0, 0.18, N_DRIVERS)      # trait glance length
    rows = []
    for d in range(N_DRIVERS):
        for cond in CONDITIONS:
            rev = (cond == "revised HMI")
            for seg in SEGMENTS:
                relief = REVISED_RELIEF[seg] if rev else 0.0
                base = 22 + driver_offset[d] + SEG_DEMAND[seg] - relief
                sub = {
                    "mental":      base + RNG.normal(0, 6),
                    "physical":    12 + RNG.normal(0, 5),
                    "temporal":    base + 6 + RNG.normal(0, 6),
                    "performance": 18 + 0.25 * SEG_DEMAND[seg] + RNG.normal(0, 7),
                    "effort":      base + 2 + RNG.normal(0, 6),
                    "frustration": 14 + 0.30 * SEG_DEMAND[seg]
                                   - 0.5 * relief + RNG.normal(0, 7),
                }
                sub = {k: float(np.clip(v, 0, 100)) for k, v in sub.items()}

                g_relief = GLANCE_RELIEF[seg] if rev else 0.0
                longest = max(0.20, RNG.gamma(
                    shape=9.0,
                    scale=(SEG_GLANCE[seg] - g_relief + driver_glance[d]) / 9.0))
                n_glances = RNG.poisson(1.1 + 0.9 * SEG_DEMAND[seg] / 31.0)

                rows.append(dict(driver=d, condition=cond, segment=seg,
                                 longest_glance=longest,
                                 eors=longest + 0.55 * n_glances,
                                 **sub))
    df = pd.DataFrame(rows)
    df["tlx_raw"] = df[SUBSCALES].mean(axis=1)
    return df


df = simulate()

# ---------------------------------------------------------------- 1. session
print("1. SESSION-LEVEL SUMMARY  (what a single post-drive rating would give)")
sess = df.groupby("condition")["tlx_raw"].mean()
for c in CONDITIONS:
    print(f"   {c:<13}  raw TLX = {sess[c]:5.1f}")
print(f"   difference   = {sess['baseline HMI'] - sess['revised HMI']:5.1f}"
      "  TLX points")

# ------------------------------------------------------------- 2. by segment
print("\n2. SEGMENT-LEVEL RAW TLX")
print(f"   {'segment':<17}{'baseline':>10}{'revised':>10}{'diff':>8}")
seg_tbl = df.pivot_table(index="segment", columns="condition",
                         values="tlx_raw", aggfunc="mean")
for seg in SEGMENTS:
    b, r = seg_tbl.loc[seg, "baseline HMI"], seg_tbl.loc[seg, "revised HMI"]
    print(f"   {seg:<17}{b:>10.1f}{r:>10.1f}{b - r:>8.1f}")

# -------------------------------------------------- 3. paired test at taper
print("\n3. PAIRED COMPARISON AT THE TAPER  (n = 32 drivers)")
tap = df[df.segment == "taper"].pivot_table(index="driver",
                                            columns="condition",
                                            values="tlx_raw")
d_i = tap["baseline HMI"] - tap["revised HMI"]
t, p = stats.ttest_rel(tap["baseline HMI"], tap["revised HMI"])
dz = d_i.mean() / d_i.std(ddof=1)
ci = stats.t.interval(0.95, len(d_i) - 1, loc=d_i.mean(),
                      scale=stats.sem(d_i))
print(f"   mean paired difference = {d_i.mean():5.1f} TLX points")
print(f"   95% CI                 = [{ci[0]:.1f}, {ci[1]:.1f}]")
print(f"   t({len(d_i) - 1}) = {t:.2f},  p = {p:.2e},  Cohen's dz = {dz:.2f}")

# ------------------------------------------------------------ 4. glance data
print("\n4. LONGEST SINGLE OFF-ROAD GLANCE (s), against the 2.0 s criterion")
print(f"   {'segment':<17}{'cond':<10}{'mean':>7}{'85th':>7}{'95th':>7}"
      f"{'% > 2.0 s':>11}")
for seg in SEGMENTS:
    for cond in CONDITIONS:
        g = df[(df.segment == seg) & (df.condition == cond)]["longest_glance"]
        print(f"   {seg:<17}{cond.split()[0]:<10}{g.mean():>7.2f}"
              f"{np.percentile(g, 85):>7.2f}{np.percentile(g, 95):>7.2f}"
              f"{100 * (g > 2.0).mean():>10.1f}%")

# ------------------------------------------------- 5. do the measures agree?
print("\n5. AGREEMENT BETWEEN THE SUBJECTIVE AND OBJECTIVE MEASURES")
for seg in SEGMENTS:
    s = df[df.segment == seg]
    r = np.corrcoef(s["temporal"], s["longest_glance"])[0, 1]
    print(f"   {seg:<17} r(temporal demand, longest glance) = {r:+.2f}")

# ------------------------------------------------------------- 6. the figure
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
col = {"baseline HMI": GOLD, "revised HMI": SAGE}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

x = np.arange(len(SEGMENTS))                                   # (a) workload by segment
for k, cond in enumerate(CONDITIONS):
    s = df[df.condition == cond].groupby("segment")["tlx_raw"]
    a1.bar(x + (k - 0.5) * 0.38, s.mean()[SEGMENTS], 0.36, yerr=s.sem()[SEGMENTS],
           capsize=2.3, color=col[cond], edgecolor=INK, lw=0.5, ecolor=INK, label=cond)
a1.set_xticks(x, [s.replace(" ", "\n") for s in SEGMENTS], fontsize=7)
a1.set(ylabel="Raw NASA-TLX", ylim=(0, 45),
       title="(a) Workload is a property of the segment")
a1.legend(fontsize=6.5, loc="upper left")

for cond in CONDITIONS:                                        # (b) longest glance at the taper
    g = np.sort(df[(df.segment == "taper") & (df.condition == cond)]["longest_glance"])
    a2.step(g, np.arange(1, len(g) + 1) / len(g), where="post", color=col[cond], lw=1.7,
            label=cond)
a2.axvline(2.0, color=RUST, ls="--", lw=0.9)
a2.text(2.06, 0.06, "2.0 s criterion", fontsize=6.5, color=RUST)
a2.set(xlabel="Longest single off-road glance at the taper (s)",
       ylabel="Cumulative share of drivers", ylim=(0, 1.04),
       title="(b) The criterion is met by the median, not the tail")
a2.legend(fontsize=6.5, loc="lower right")

fig.tight_layout()
fig.savefig("fig02_06_cs1_workload.png", dpi=300)
print("\n   figure written to fig02_06_cs1_workload.png")
