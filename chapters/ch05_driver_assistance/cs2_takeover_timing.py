"""Chapter 5 - Case Study 2: Takeover Request Timing and Transition Design for SAE Level 3 Systems.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/05a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch05` or `python scripts/run_all.py --chapter 5`;
outputs are written to outputs/ch05/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# ---------- Level 3: how long do drivers take to take over? ----------
rng = np.random.default_rng(55)
TASKS = {"no secondary task": (2.2, 0.40, 0.40),     # median (s), log-SD, share of drivers
         "phone": (3.0, 0.45, 0.35),
         "reading": (3.6, 0.50, 0.25)}
MANEUVER = 2.0                                       # seconds needed after taking over


def cdf_task(t, task):
    med, sd, _ = TASKS[task]
    return stats.lognorm.cdf(t, sd, scale=med)


def cdf_mix(t):
    return sum(w * cdf_task(t, k) for k, (_, _, w) in TASKS.items())


def sample(n):
    pick = rng.choice(len(TASKS), n, p=[v[2] for v in TASKS.values()])
    med = np.array([v[0] for v in TASKS.values()])[pick]
    sd = np.array([v[1] for v in TASKS.values()])[pick]
    return med * np.exp(sd * rng.normal(size=n)), pick


tot, pick = sample(1_000_000)
mix_mean = sum(w * med * np.exp(sd ** 2 / 2) for med, sd, w in TASKS.values())
print(f"Mean takeover time of all drivers: {mix_mean:.3f} s (closed form), "
      f"{tot.mean():.3f} s (simulated)\n")
rows = []
for b in [5.0, 7.0, 10.0]:
    rows.append((b, *[cdf_task(b - MANEUVER, k) for k in TASKS], cdf_mix(b - MANEUVER),
                 (tot <= b - MANEUVER).mean()))
cov = pd.DataFrame(rows, columns=["lead_time_s", "none", "phone", "reading", "all_closed",
                                  "all_sim"])
print("Share of drivers who take over in time:")
print(cov.round(4).to_string(index=False))
print("\nLead time needed for a given share of drivers:")
for share in [0.95, 0.99]:
    per = {k: stats.lognorm.ppf(share, v[1], scale=v[0]) + MANEUVER
           for k, v in TASKS.items()}
    mix = brentq(lambda t: cdf_mix(t) - share, 0.1, 60) + MANEUVER
    print(f"  {share:.0%}: " + ", ".join(f"{k} {t:.2f} s" for k, t in per.items()))
    sim_q = np.quantile(tot, share) + MANEUVER
    print(f"       all drivers {mix:.2f} s (simulated {sim_q:.2f} s)")

# ---------- A stopped vehicle ahead: when is a takeover too late? ----------
A_MAX = 7.0                                          # firm braking (m/s^2)


def p_crash(lead, kmh):
    v = kmh / 3.6
    return 1 - cdf_mix(max(lead - v / (2 * A_MAX), 1e-9))


rows2 = []
for kmh in [80, 100, 120, 130]:
    v = kmh / 3.6
    need = brentq(lambda t: cdf_mix(t) - 0.99, 0.1, 60) + v / (2 * A_MAX)
    rows2.append((kmh, v / (2 * A_MAX), p_crash(7.0, kmh),
                  (tot > 7.0 - v / (2 * A_MAX)).mean(), p_crash(10.0, kmh), need))
phys = pd.DataFrame(rows2, columns=["speed_kmh", "braking_s", "P_crash_7s",
                                    "P_crash_7s_sim", "P_crash_10s", "lead_for_1pct"])
print(phys.round(4).to_string(index=False))
by_task = {k: 1 - cdf_task(7.0 - (120 / 3.6) / (2 * A_MAX), k) for k in TASKS}
print("\nAt 120 km/h with 7 s of warning, P(crash) by task:")
print("  " + ", ".join(f"{k} {p:.4f}" for k, p in by_task.items()))

# ---------- Validating the lead time: tolerance tests and their power ----------
def wilks_n(cover=0.95, conf=0.95, c=0):
    return next(n for n in range(1, 5000) if stats.binom.cdf(c, n, 1 - cover) <= 1 - conf)


print(f"Drivers needed to show 95% coverage with 95% confidence: {wilks_n()} with no "
      f"failures,\n  {wilks_n(c=1)} with one; closed form for no failures "
      f"{np.log(0.05) / np.log(0.95):.2f}")
for n, c in [(59, 0), (93, 1)]:
    p97, p98 = stats.binom.cdf(c, n, 0.03), stats.binom.cdf(c, n, 0.02)
    print(f"  plan ({n}, {c}): P(pass) {p97:.4f} if true coverage is 97%, "
          f"{p98:.4f} if 98%")


def plan(power=0.80, good=0.98, bad=0.95, risk=0.05):
    for n in range(10, 3000):
        ok = [c for c in range(n) if stats.binom.cdf(c, n, 1 - bad) <= risk]
        c = max(ok, default=-1)
        if c >= 0 and stats.binom.cdf(c, n, 1 - good) >= power:
            return n, c


n_p, c_p = plan()
sim = (rng.binomial(n_p, [0.05, 0.02], (200_000, 2)) <= c_p).mean(axis=0)
print(f"Plan with 80% power at 98% coverage: {n_p} drivers, at most {c_p} late;")
print(f"  P(pass) {stats.binom.cdf(c_p, n_p, 0.05):.4f} at 95% and "
      f"{stats.binom.cdf(c_p, n_p, 0.02):.4f} at 98% "
      f"(simulated {sim[0]:.4f} and {sim[1]:.4f})")

# ---------- When nobody takes over: minimum risk maneuvers ----------
TOR_PER_H, BUDGET = 0.25, 10.0
p_late = 1 - cdf_mix(BUDGET)
mrm = 1000 * TOR_PER_H * p_late
counts = rng.poisson(TOR_PER_H * 1000, 20_000)
late = rng.binomial(counts, p_late)
print(f"P(no takeover within {BUDGET:.0f} s) {p_late:.4f}; "
      "minimum risk maneuvers per 1,000 h:")
print(f"  {mrm:.3f} (closed form), {late.mean():.3f} (simulated)")
share = {k: v[2] * (1 - cdf_task(BUDGET, k)) / p_late for k, v in TASKS.items()}
print("Share of those maneuvers by task:")
print("  " + ", ".join(f"{k} {s:.3f}" for k, s in share.items()))

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
col = {"no secondary task": SAGE, "phone": GOLD, "reading": RUST}
a = ax[0, 0]                                                   # (a) takeover times
tt = np.linspace(0.01, 12, 400)
for k, (med, sd, w) in TASKS.items():
    a.plot(tt, stats.lognorm.pdf(tt, sd, scale=med), color=col[k], lw=1.4,
           label=f"{k} (median {med:.1f} s)")
a.set(xlabel="Takeover time (s)", ylabel="Density",
      title="(a) The secondary task sets the takeover time")
a.legend(fontsize=6.5)
a = ax[0, 1]                                                   # (b) coverage
ll = np.linspace(2.5, 16, 300)
for k in TASKS:
    a.plot(ll, cdf_task(ll - MANEUVER, k), color=col[k], lw=1, ls="--")
a.plot(ll, cdf_mix(ll - MANEUVER), color=INK, lw=1.7, label="all drivers")
a.plot(cov.lead_time_s, cov.all_sim, "o", color=INK, ms=3.9, label="simulated")
for share, ls in [(0.95, ":"), (0.99, "-.")]:
    a.axhline(share, color=INK, ls=ls, lw=0.6)
a.set(xlabel="Lead time of the takeover request (s)", ylabel="Share taking over in time",
      title="(b) What a lead time buys (dashed: by task)", ylim=(0, 1.02))
a.legend(fontsize=6.5, loc="lower right")
a = ax[1, 0]                                                   # (c) crash risk
for kmh, c in [(80, SAGE), (100, GOLD), (120, RUST), (130, INK)]:
    a.semilogy(ll, [p_crash(x, kmh) for x in ll], color=c, lw=1.3, label=f"{kmh} km/h")
a.semilogy(phys.assign(l=7.0).l, phys.P_crash_7s_sim, "o", color=INK, ms=3.2)
a.axhline(0.01, color=INK, ls=":", lw=0.6)
a.axvline(7.0, color=INK, ls="--", lw=0.6)
a.set(xlabel="Lead time of the takeover request (s)",
      ylabel="P(collision with a stopped vehicle)",
      title="(c) Seven seconds is not enough at speed", ylim=(1e-3, 1))
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
a.legend(fontsize=6.5, loc="lower left", ncol=2)
a = ax[1, 1]                                                   # (d) validation tests
cc = np.linspace(0.90, 1.0, 300)
for (n, c, lab, colr) in [(59, 0, "59 drivers, none late", GOLD),
                          (93, 1, "93 drivers, at most 1 late", OLIVE),
                          (n_p, c_p, f"{n_p} drivers, at most {c_p} late", SAGE)]:
    a.plot(cc, stats.binom.cdf(c, n, 1 - cc), color=colr, lw=1.4, label=lab)
a.axvline(0.95, color=RUST, ls=":", lw=0.8)
a.axvline(0.98, color=SAGE, ls=":", lw=0.8)
a.set(xlabel="True share of drivers taking over in time",
      ylabel="Probability the test passes",
      title="(d) A 95/95 test needs power, not only confidence")
a.legend(fontsize=6, loc="center left")
fig.tight_layout(); fig.savefig("chap05_fig02_takeover.png", dpi=300)
