"""Chapter 10 - Case Study 1: Simulator and Field Evaluation of Distraction, Workload, and Trust.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/10a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch10` or `python scripts/run_all.py --chapter 10`;
outputs are written to outputs/ch10/.
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Glances away from the road while entering a destination ----------
rng = np.random.default_rng(10)
HMI = {"touchscreen": (6.5, 1.25, 0.45),      # mean glances, median glance (s), log-SD
       "voice": (3.0, 0.90, 0.40)}


def glance_test(n, hmi):
    lam, med, sd = HMI[hmi]
    style = rng.normal(0, 0.15, n)             # each driver's own glance style (log scale)
    rows = []
    for i in range(n):
        g = med * np.exp(style[i] + sd * rng.normal(size=max(1, rng.poisson(lam))))
        rows.append((g.mean(), (g > 2.0).sum(), np.ceil(0.15 * len(g)), g.sum()))
    m = pd.DataFrame(rows, columns=["mean_glance", "long", "allowed", "teort"])
    # no more than 15% of glances (rounded up) may exceed 2 s
    m["c1"], m["c2"] = m.mean_glance <= 2.0, m.long <= m.allowed
    m["c3"] = m.teort <= 12.0                  # NHTSA: each met by 21 of 24 drivers
    return m


print("One 24-driver simulator test (number of drivers meeting each criterion):")
tests = {}
for hmi in HMI:
    t = tests[hmi] = glance_test(24, hmi)
    met = t[["c1", "c2", "c3"]].sum()
    print(f"  {hmi:<12} mean glance {met.c1:>2}/24   long glances {met.c2:>2}/24   "
          f"total eyes-off-road {met.c3:>2}/24   "
          f"-> {'PASS' if met.min() >= 21 else 'FAIL'}")

# ---------- How reliable is the 21-of-24 rule? ----------
pop = {h: glance_test(100_000, h) for h in HMI}
# population share meeting the 12-s criterion
q = {h: pop[h].c3.mean() for h in HMI}
oc = []
for h in HMI:
    exact = stats.binom.sf(20, 24, q[h])       # P(at least 21 of 24)
    reps = rng.random((20_000, 24)) < q[h]
    sim = (reps.sum(axis=1) >= 21).mean()
    oc.append((h, q[h], exact, sim, (sim - exact) / np.sqrt(exact * (1 - exact) / 20_000)))
oc = pd.DataFrame(oc, columns=["interface", "share_meeting_12s", "P_pass_binomial",
                               "P_pass_simulated", "z"])
print("\nOperating characteristic of the 12-s criterion:")
print(oc.round(4).to_string(index=False))
for h in HMI:                                  # all three criteria together, by resampling
    idx = rng.integers(0, 100_000, (20_000, 24))
    ok = np.stack([pop[h][c].to_numpy()[idx].sum(axis=1) >= 21
                   for c in ["c1", "c2", "c3"]])
    print(f"P(pass all three criteria), {h}: {ok.all(axis=0).mean():.4f}")

# ---------- Workload, trust, usability: simulator (24 drivers), track (12) ----------
import statsmodels.formula.api as smf
EFFECT = {"simulator": 0.120, "test track": 0.095}       # DRT slowing with touchscreen (s)
# each driver's sensitivity, kept across settings
share = rng.normal(0, 0.4, 24)


def drt(setting, ids, k=30):
    rows = []
    for i in ids:
        base = 0.45 + rng.normal(0, 0.06) + (0.03 if setting == "test track" else 0.0)
        for hmi in ["voice", "touchscreen"]:
            eff = EFFECT[setting] * (1 + share[i]) if hmi == "touchscreen" else 0.0
            rows += [(setting, i, hmi, r) for r in base + eff + rng.normal(0, 0.12, k)]
    return pd.DataFrame(rows, columns=["setting", "pid", "hmi", "rt"])


trials = pd.concat([drt("simulator", range(24)), drt("test track", range(12))])
res = []
for setting, d in trials.groupby("setting", sort=False):
    m = d.groupby(["pid", "hmi"]).rt.mean().unstack()
    diff = m.touchscreen - m.voice
    t = stats.ttest_rel(m.touchscreen, m.voice)
    cells = m.stack().rename("rt").reset_index()      # two cell means per driver
    fit = smf.ols("rt ~ C(hmi, Treatment('voice')) + C(pid)", cells).fit()
    b = fit.params.iloc[1]
    res.append((setting, len(m), diff.mean(), diff.std(ddof=1) / np.sqrt(len(m)), t.pvalue,
                b, fit.bse.iloc[1]))
res = pd.DataFrame(res, columns=["setting", "n", "paired_diff_s", "paired_se", "p",
                                 "fixed_effects_s", "fixed_effects_se"])
print("Detection-response slowing with the touchscreen: paired t vs driver fixed effects")
res["p"] = res.p.map(lambda x: f"{x:.1e}")
print(res.round(4).to_string(index=False))
res["p"] = res.p.astype(float)
eff = {s: d.groupby(["pid", "hmi"]).rt.mean().unstack()
       for s, d in trials.groupby("setting")}
e_sim = (eff["simulator"].touchscreen - eff["simulator"].voice).iloc[:12]
e_trk = eff["test track"].touchscreen - eff["test track"].voice
r = np.corrcoef(e_sim, e_trk)[0, 1]
print(f"Track / simulator effect ratio {res.paired_diff_s[1] / res.paired_diff_s[0]:.2f}; "
      f"same-driver correlation r = {r:.2f} (12 drivers)")

# ---------- Subjective scales: NASA-TLX (raw), trust (Jian et al.), SUS ----------
def scales(hmi):
    load = 55 if hmi == "touchscreen" else 35
    tlx = np.clip(rng.normal(load, 12, (24, 6)), 0, 100).mean(axis=1)
    lean = rng.normal(0.6 if hmi == "voice" else 0.0, 0.8, 24)[:, None]
    trust = np.clip(np.round(4 + lean + rng.normal(0, 0.9, (24, 12))), 1, 7)
    trust[:, :5] = 8 - trust[:, :5]            # items 1-5 are worded as distrust
    sus = np.clip(np.round(3 + lean * 0.8 + rng.normal(0, 0.7, (24, 10))), 1, 5)
    sus[:, 1::2] = 6 - sus[:, 1::2]            # even items are worded negatively
    score = 2.5 * ((sus[:, 0::2] - 1).sum(axis=1) + (5 - sus[:, 1::2]).sum(axis=1))
    trust_mean = np.c_[8 - trust[:, :5], trust[:, 5:]].mean(axis=1)
    return tlx, trust_mean, score


subj = {h: scales(h) for h in ["voice", "touchscreen"]}
for k, name in enumerate(["NASA-TLX (raw, 0-100)", "Trust (1-7)", "SUS (0-100)"]):
    v, t_ = subj["voice"][k], subj["touchscreen"][k]
    tt = stats.ttest_rel(t_, v)
    print(f"{name:<22} voice {v.mean():5.1f}  touchscreen {t_.mean():5.1f}  "
          f"difference {np.mean(t_ - v):+6.1f}  p = {tt.pvalue:.4f}")

# ---------- How many drivers does the track study need? ----------
sd_d = e_trk.std(ddof=1)
delta = 0.030                                  # smallest slowing worth detecting (s)


def power_exact(n, d=delta, s=sd_d, alpha=0.05):
    df, nc = n - 1, d / s * np.sqrt(n)         # noncentral t for a paired test
    tc = stats.t.ppf(1 - alpha / 2, df)
    return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def power_sim(n, reps=20_000):
    x = rng.normal(delta, sd_d, (reps, n))
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / np.sqrt(n))
    return (np.abs(t) > stats.t.ppf(0.975, n - 1)).mean()


ns = [6, 8, 10, 12, 16, 20, 24, 30]
pw = pd.DataFrame([(n, power_exact(n), power_sim(n)) for n in ns],
                  columns=["drivers", "power_noncentral_t", "power_simulated"])
print(f"SD of per-driver effects on the track: {sd_d * 1000:.1f} ms; smallest effect "
      f"of interest {delta * 1000:.0f} ms")
print(pw.round(4).to_string(index=False))
n80 = next(n for n in range(3, 200) if power_exact(n) >= 0.8)
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
n_norm = np.ceil((z * sd_d / delta) ** 2)
print(f"Drivers for 80% power: {n80} (noncentral t); {n_norm:.0f} "
      f"(normal approximation)")

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
a = ax[0, 0]                                                   # (a) one 24-driver test
for k, (h, c) in enumerate([("voice", SAGE), ("touchscreen", RUST)]):
    t = tests[h].sort_values("teort").reset_index(drop=True)
    ok = t.c1 & t.c2 & t.c3
    a.scatter(np.arange(24), t.teort, s=15.8, color=np.where(ok, c, "white"), edgecolor=c,
              lw=1, label=f"{h}: {int(ok.sum())} of 24 meet all three", zorder=3)
a.axhline(12, color=INK, ls="--", lw=0.7)
a.text(0.3, 12.4, "12 s total eyes-off-road", fontsize=6.5)
a.set(xlabel="Driver (sorted)", ylabel="Total eyes-off-road time (s)",
      title="(a) One NHTSA-style test; open markers fail a criterion")
a.legend(fontsize=6.5, loc="upper left")
# (b) operating characteristic
a = ax[0, 1]
qq = np.linspace(0.6, 1.0, 200)
a.plot(qq, stats.binom.sf(20, 24, qq), color=GOLD, lw=1.4, label="binomial, 21+ of 24")
for qs in [0.7, 0.8, 0.85, 0.9, 0.95]:
    a.plot(qs, ((rng.random((20_000, 24)) < qs).sum(axis=1) >= 21).mean(), "o", color=INK,
           ms=3.2)
a.plot([], [], "o", color=INK, ms=3.2, label="20,000 simulations")
for h, c in [("touchscreen", RUST), ("voice", SAGE)]:
    a.axvline(q[h], color=c, ls=":", lw=1)
    a.text(q[h] - 0.005, 0.52, h, color=c, fontsize=6.5, rotation=90, ha="right")
a.set(xlabel="Share of all drivers who meet the 12-s criterion",
      ylabel="Probability the interface passes", title="(b) How often the test passes")
a.legend(fontsize=6, loc="upper left")
a = ax[1, 0]                                                   # (c) relative validity
a.scatter(e_sim * 1000, e_trk * 1000, s=19.2, color=SKY, edgecolor=INK, zorder=3)
lim = [0, 1000 * max(e_sim.max(), e_trk.max()) * 1.1]
a.plot(lim, lim, color=INK, lw=0.5, ls=":", label="equal effects")
ratio = res.paired_diff_s[1] / res.paired_diff_s[0]
a.plot(lim, np.array(lim) * ratio, color=RUST, lw=1,
       label=f"mean ratio {res.paired_diff_s[1] / res.paired_diff_s[0]:.2f}")
a.set(xlim=lim, ylim=lim, xlabel="Touchscreen slowing in the simulator (ms)",
      ylabel="Touchscreen slowing on the track (ms)",
      title=f"(c) Same drivers, two settings (r = {r:.2f})")
a.legend(fontsize=6.5, loc="upper left")
a = ax[1, 1]                                                   # (d) power
nn = np.arange(4, 41)
a.plot(nn, [power_exact(n) for n in nn], color=SAGE, lw=1.4, label="noncentral t")
a.plot(pw.drivers, pw.power_simulated, "o", color=INK, ms=3.2, label="simulation")
a.axhline(0.8, color=INK, ls="--", lw=0.6)
a.axvline(n80, color=RUST, ls=":", lw=1)
a.text(n80 + 0.5, 0.3, f"{n80} drivers", color=RUST, fontsize=6.5)
a.set(ylim=(0, 1.02), xlabel="Drivers in the track study",
      ylabel="Power to detect 30 ms", title="(d) Sample size for the smallest effect")
a.legend(fontsize=6.5, loc="lower right")
fig.tight_layout(); fig.savefig("chap10_fig01_sim_field.png", dpi=300)
