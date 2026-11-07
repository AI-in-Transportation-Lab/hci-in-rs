"""Chapter 12 - Case Study 1: Using HCI Evidence in Procurement for Safety-Critical Interfaces.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/12a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch12` or `python scripts/run_all.py --chapter 12`;
outputs are written to outputs/ch12/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# ---------- Three bids for a fleet driver-monitoring system ----------
rng = np.random.default_rng(12)
BIDS = {"A": dict(det=(184, 200), fa=(60, 150.0), sus=(76, 10, 20), price=900),
        "B": dict(det=(354, 400), fa=(90, 300.0), sus=(72, 12, 40), price=750),
        "C": dict(det=(34, 36), fa=(22, 40.0), sus=(81, 9, 10), price=600)}
REQ = {"det": 0.85, "fa": 0.60, "sus": 68.0}   # detection, false alarms per h, SUS


def det_lower(x, n, a=0.05):                     # one-sided Clopper-Pearson lower bound
    return stats.beta.ppf(a, x, n - x + 1)


def det_lower_root(x, n, a=0.05):                # same bound by exact binomial root
    return brentq(lambda p: stats.binom.sf(x - 1, n, p) - a, 1e-9, x / n)


def fa_upper(k, hours, a=0.05):                  # one-sided Poisson upper bound on rate
    return stats.chi2.ppf(1 - a, 2 * (k + 1)) / 2 / hours


def fa_upper_root(k, hours, a=0.05):
    return brentq(lambda mu: stats.poisson.cdf(k, mu) - a, k, 10 * k + 10) / hours


rows, sus_data = [], {}
for v, b in BIDS.items():
    (x, n), (k, h), (m, sd, ns) = b["det"], b["fa"], b["sus"]
    s = np.clip(rng.normal(m, sd, ns), 0, 100)   # each participant's SUS score
    sus_data[v] = s
    t_lo = s.mean() - stats.t.ppf(0.95, ns - 1) * s.std(ddof=1) / np.sqrt(ns)
    boot = np.percentile(rng.choice(s, (20_000, ns)).mean(axis=1), 5)
    rows.append((v, x / n, det_lower(x, n), det_lower_root(x, n), k / h, fa_upper(k, h),
                 fa_upper_root(k, h), s.mean(), t_lo, boot))
ev = pd.DataFrame(rows, columns=["bid", "det", "det_lo", "det_lo_root", "fa_h", "fa_hi",
                                 "fa_hi_root", "sus", "sus_lo_t", "sus_lo_boot"])
print(ev.round(4).to_string(index=False))
ev["passes"] = ((ev.det_lo >= REQ["det"]) & (ev.fa_hi <= REQ["fa"])
                & (ev.sus_lo_t >= REQ["sus"]))
ev["passes_on_points"] = ((ev.det >= REQ["det"]) & (ev.fa_h <= REQ["fa"])
                          & (ev.sus >= REQ["sus"]))
print("\nMeets every requirement on its 95% bound:", dict(zip(ev.bid, ev.passes)))
print("Meets every requirement on point estimates:",
      dict(zip(ev.bid, ev.passes_on_points)))
lo352 = det_lower(352, 400)
print(f"Sensitivity: with 352 of 400 detections, B's bound would be "
      f"{lo352:.4f}, short of {REQ['det']} by {REQ['det'] - lo352:.4f}")

# ---------- What each bid is worth per vehicle-year, with its uncertainty ----------
# $ per detected episode, per FA
EPISODES, BENEFIT, HOURS, FA_COST = 20, 150.0, 2000, 0.40
D = 200_000


def value(det, fa, price):
    return EPISODES * BENEFIT * det - HOURS * FA_COST * fa - price


draws, closed = {}, {}
for v, b in BIDS.items():
    (x, n), (k, h) = b["det"], b["fa"]
    det = rng.beta(x + 0.5, n - x + 0.5, D)      # Jeffreys posteriors
    fa = rng.gamma(k + 0.5, 1 / h, D)
    draws[v] = value(det, fa, b["price"])
    closed[v] = value((x + 0.5) / (n + 1), (k + 0.5) / h, b["price"])
M = np.column_stack([draws[v] for v in BIDS])
best = pd.Series(M.argmax(axis=1)).map(dict(enumerate(BIDS))).value_counts(normalize=True)
tab = pd.DataFrame({"E_value_closed": closed,
                    "E_value_sim": {v: draws[v].mean() for v in BIDS},
                    "P5": {v: np.percentile(draws[v], 5) for v in BIDS},
                    "P_best": {v: best.get(v, 0.0) for v in BIDS}})
print(tab.round(2).to_string())
evpi = M.max(axis=1).mean() - max(closed.values())
ok = list(ev.bid[ev.passes])
M2 = np.column_stack([draws[v] for v in ok])
print(f"\nExpected value of perfect information, all bids: ${evpi:.2f} per vehicle-year")
evpi_ok = M2.max(axis=1).mean() - max(closed[v] for v in ok)
print(f"Among bids that meet the requirements ({', '.join(ok)}): best on expectation "
      f"{max(ok, key=closed.get)}, EVPI ${evpi_ok:.2f}")

# ---------- Designing the acceptance test written into the contract ----------
P_BAD, P_CLAIM, ALPHA, POWER = 0.85, 0.92, 0.05, 0.90


def plan_exact():
    for n in range(10, 2000):                    # smallest n with an acceptance number
        c = int(stats.binom.isf(ALPHA, n, P_BAD)) + 1
        while stats.binom.sf(c - 1, n, P_BAD) > ALPHA:
            c += 1                               # accept only if at least c detections
        if stats.binom.sf(c - 1, n, P_CLAIM) >= POWER:
            return n, c


n_ex, c_ex = plan_exact()
za, zb = stats.norm.isf(ALPHA), stats.norm.isf(1 - POWER)
n_norm = ((za * np.sqrt(P_BAD * (1 - P_BAD)) + zb * np.sqrt(P_CLAIM * (1 - P_CLAIM)))
          / (P_CLAIM - P_BAD)) ** 2
print(f"Exact plan: test {n_ex} staged events, accept if at least {c_ex} are detected")
print(f"  P(accept | true 0.85) = {stats.binom.sf(c_ex - 1, n_ex, P_BAD):.4f}; "
      f"P(accept | true 0.92) = {stats.binom.sf(c_ex - 1, n_ex, P_CLAIM):.4f}")
sim = rng.binomial(n_ex, [P_BAD, P_CLAIM], (200_000, 2)) >= c_ex
print(f"  simulated: {sim[:, 0].mean():.4f} and {sim[:, 1].mean():.4f}")
print(f"Normal approximation: {n_norm:.1f} events")

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
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})
fig, ax = plt.subplots(2, 2, figsize=(12, 8.2))
col = {"A": SAGE, "B": GOLD, "C": RUST}
x = np.arange(3)
PANELS = [("det", "det_lo", REQ["det"], "Detection rate"),
          ("fa_h", "fa_hi", REQ["fa"], "False alarms per hour")]
for k, (metric, lo, req, lab) in enumerate(PANELS):
    # (a), (b) evidence vs. requirement
    a = ax[0, k]
    for i, r in ev.iterrows():
        a.plot([i, i], sorted([r[metric], r[lo]]), color=col[r.bid], lw=5, alpha=0.6,
               solid_capstyle="round")
        a.plot(i, r[metric], "o", color=col[r.bid], mec=INK, ms=8)
        a.plot(i, r[lo], "_", color=INK, ms=18, mew=2)
    a.axhline(req, color=INK, ls="--", lw=1.2)
    a.set_xticks(x, [f"bid {v}" for v in BIDS])
    a.set(ylabel=lab, title=f"({'ab'[k]}) {lab}: estimate (dot) and 95% bound (bar)")
    a.text(-0.35, req, "requirement", fontsize=7.8, va="bottom", ha="left")
    a.set_xlim(-0.4, 2.4)
ax[0, 0].set_ylim(0.8, 0.97)
ax[0, 1].set_ylim(0.2, 1.1)
# (c) value under uncertainty
a = ax[1, 0]
bins = np.linspace(1100, 2300, 120)
for v in BIDS:
    a.hist(draws[v], bins, density=True, histtype="step", lw=2, color=col[v],
           label=f"bid {v}: mean ${draws[v].mean():,.0f}, P(best) {best.get(v, 0):.2f}")
a.set(xlabel="Net value per vehicle-year ($)", ylabel="Posterior density",
      title="(c) Value of each bid, with its uncertainty")
a.legend(fontsize=8, frameon=False, loc="upper left")
a = ax[1, 1]                                                   # (d) acceptance plan
pp = np.linspace(0.80, 0.97, 300)
a.plot(pp, stats.binom.sf(c_ex - 1, n_ex, pp), color=SAGE, lw=2.4,
       label=f"accept if at least {c_ex} of {n_ex} detected")
a.axvline(P_BAD, color=RUST, ls=":", lw=1.4)
a.axvline(P_CLAIM, color=SAGE, ls=":", lw=1.4)
a.plot([P_BAD, P_CLAIM], [stats.binom.sf(c_ex - 1, n_ex, P_BAD),
                          stats.binom.sf(c_ex - 1, n_ex, P_CLAIM)], "o", color=INK, ms=7)
a.text(P_BAD - 0.003, 0.5, "requirement\n0.85", fontsize=7.8, color=RUST, ha="right")
a.text(P_CLAIM + 0.003, 0.5, "claim\n0.92", fontsize=7.8, color=SAGE)
a.set(xlabel="True detection rate of the delivered system",
      ylabel="Probability of acceptance", title="(d) The contract's acceptance test")
a.legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("chap12_fig01_procurement.png", dpi=200)
