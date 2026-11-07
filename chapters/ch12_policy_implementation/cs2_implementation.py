"""Chapter 12 - Case Study 2: Implementation Framework for Agency and Industry Deployment.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/12a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch12` or `python scripts/run_all.py --chapter 12`;
outputs are written to outputs/ch12/.
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Staged rollout with Bayesian gates on the crash rate ratio ----------
rng = np.random.default_rng(1212)
# prior on the rate ratio: gamma(shape 4, rate 4), mean 1
A0, B0 = 4.0, 4.0
PHASES = [("pilot", 40.0), ("limited", 160.0)]   # crashes expected without the program
# advance if P(RR < 1) >= 0.90; stop if P(RR > 1) >= 0.90
GO = 0.90
RR_TRUE = 0.85


def p_benefit(a, b):
    return stats.gamma.cdf(1.0, a, scale=1 / b)


a, b, log = A0, B0, []
for name, e in PHASES:
    y = rng.poisson(RR_TRUE * e)     # crashes observed while the program runs
    a, b = a + y, b + e
    mc = (rng.gamma(a, 1 / b, 400_000) < 1).mean()
    log.append((name, e, y, a / b, p_benefit(a, b), mc,
                stats.gamma.ppf(0.05, a, scale=1 / b),
                stats.gamma.ppf(0.95, a, scale=1 / b)))
log = pd.DataFrame(log, columns=["phase", "expected", "observed", "post_mean_RR",
                                 "P_benefit", "P_benefit_mc", "RR_5pct", "RR_95pct"])
print(log.round(4).to_string(index=False))


def rollout(rr, n=20_000):           # operating characteristics of the gates
    a, b = np.full(n, A0), np.full(n, B0)
    alive, first = np.ones(n, bool), None
    for _, e in PHASES:
        decided, advance = ~alive, np.zeros(n, bool)
        for attempt in range(2):     # one look, then at most one extension
            act = ~decided
            y = rng.poisson(rr * e, n)
            a, b = np.where(act, a + y, a), np.where(act, b + e, b)
            pb = p_benefit(a, b)
            go, stop = act & (pb >= GO), act & (1 - pb >= GO)
            if first is None:
                first = go.mean()    # share advancing at the very first look
            advance |= go
            decided |= go | stop
        alive &= advance             # still undecided after the extension: stop
    return alive.mean(), first


y_star = max(y for y in range(200) if p_benefit(A0 + y, B0 + PHASES[0][1]) >= GO)
rows = []
for rr in [0.75, 0.85, 0.95, 1.00, 1.10]:
    full, first = rollout(rr)
    rows.append((rr, stats.poisson.cdf(y_star, rr * PHASES[0][1]), first, full))
oc = pd.DataFrame(rows, columns=["true_RR", "P_go_at_pilot_closed", "P_go_at_pilot_sim",
                                 "P_full_deployment"])
print(f"\nThe pilot advances at its first look if it sees at most {y_star} crashes")
print("Operating characteristics of the gates (20,000 simulated rollouts each):")
print(oc.round(4).to_string(index=False))

# ---------- From efficacy to population impact (RE-AIM) ----------
EFF = 0.25                           # crash reduction for reached drivers at full fidelity
# adoption; beta reach, fidelity, upkeep
ADOPT, R_AB, F_AB, M_AB = 0.70, (6, 4), (8, 2), (3, 1)


def agencies(n):
    adopt = rng.random(n) < ADOPT
    return (adopt, rng.beta(*R_AB, n), rng.beta(*F_AB, n), rng.beta(*M_AB, n))


def impact(adopt, reach, fid, maint):
    return adopt * reach * EFF * fid * maint


means = [ADOPT, R_AB[0] / sum(R_AB), F_AB[0] / sum(F_AB), M_AB[0] / sum(M_AB)]
closed = EFF * np.prod(means)
big = impact(*agencies(2_000_000)).mean()
prog = agencies(40)                  # one program of 40 agencies
print(f"Population impact: closed form {closed:.4f}, two million simulated agencies "
      f"{big:.4f},\none 40-agency program {impact(*prog).mean():.4f}")
steps = [("efficacy among reached drivers", EFF), ("x adoption", ADOPT),
         ("x reach", means[1]), ("x fidelity", means[2]), ("x maintenance", means[3])]
level = 1.0
for name, f in steps:
    level *= f
    print(f"  {name:<32} {f:.3f}   cumulative {level:.4f}")

# ---------- Watching for a regression after deployment: a Poisson CUSUM ----------
LAM0, LAM1 = 4.0, 6.0                # incidents per month: in control, after a bad update
k_exact = (LAM1 - LAM0) / np.log(LAM1 / LAM0)
K = int(round(k_exact))              # integer reference value keeps the chart on integers


def arl_markov(lam, h):              # exact average run length by Markov chain
    P = np.zeros((h, h))
    for s in range(h):
        for x in range(0, h + K + 60):
            nxt = max(0, s + x - K)
            if nxt < h:
                P[s, nxt] += stats.poisson.pmf(x, lam)
    return np.linalg.solve(np.eye(h) - P, np.ones(h))[0]


def arl_sim(lam, h, reps=20_000, horizon=3000):
    x = rng.poisson(lam, (reps, horizon)) - K
    s, t = np.zeros(reps), np.full(reps, horizon)
    for i in range(horizon):
        s = np.maximum(0, s + x[:, i])
        hit = (s >= h) & (t == horizon)
        t[hit] = i + 1
    return t.mean()


H = next(h for h in range(2, 40) if arl_markov(LAM0, h) >= 120)
print(f"Reference value {k_exact:.4f}, rounded to {K}; decision interval H = {H}")
for lam, lab in [(LAM0, "in control"), (LAM1, "after regression")]:
    print(f"  ARL {lab:<17} Markov chain {arl_markov(lam, H):8.3f}   "
          f"simulation {arl_sim(lam, H):8.3f} months")
months = np.arange(1, 121)
counts = rng.poisson(np.where(months <= 60, LAM0, LAM1))
path, s = [], 0.0
for x_ in counts:
    s = max(0.0, s + x_ - K)
    path.append(s)
path = np.array(path)
signal = int(months[np.argmax(path >= H)]) if (path >= H).any() else None
print(f"Regression introduced after month 60; the chart signals in month {signal}")

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
a = ax[0, 0]                                                   # (a) posterior by phase
rr = np.linspace(0.3, 1.8, 400)
a.plot(rr, stats.gamma.pdf(rr, A0, scale=1 / B0), color=INK, lw=1.4, ls=":", label="prior")
aa, bb = A0, B0
for (_, r), c in zip(log.iterrows(), [GOLD, SAGE]):
    aa, bb = aa + r.observed, bb + r.expected
    a.plot(rr, stats.gamma.pdf(rr, aa, scale=1 / bb), color=c, lw=2.2,
           label=f"after {r.phase}: P(RR < 1) = {r.P_benefit:.3f}")
a.axvline(1, color=RUST, ls="--", lw=1)
a.axvline(RR_TRUE, color=INK, lw=0.8)
a.set_ylim(0, a.get_ylim()[1] * 1.12)
a.text(RR_TRUE + 0.015, a.get_ylim()[1] * 0.97, "true 0.85", fontsize=7.8, va="top")
a.set(xlabel="Crash rate ratio with the program", ylabel="Posterior density",
      title="(a) Evidence accumulates across phases")
a.legend(fontsize=8, frameon=False, loc="upper right")
a = ax[0, 1]                                                   # (b) gate characteristics
a.plot(oc.true_RR, oc.P_full_deployment, "-o", color=SAGE, mec=INK, lw=2.2, ms=6,
       label="reaches full deployment")
a.plot(oc.true_RR, oc.P_go_at_pilot_closed, "--", color=GOLD, lw=1.8,
       label="advances at the first pilot look (closed form)")
a.plot(oc.true_RR, oc.P_go_at_pilot_sim, "o", color=GOLD, mec=INK, ms=5,
       label="same, simulated")
a.axvline(1, color=RUST, ls="--", lw=1)
a.set(xlabel="True crash rate ratio", ylabel="Probability",
      title="(b) Gates rarely approve a program that does not work")
a.legend(fontsize=8, frameon=False, loc="upper right")
a = ax[1, 0]                                                   # (c) efficacy to impact
labs = ["efficacy", "x adoption", "x reach", "x fidelity", "x maintenance"]
vals, lvl = [], 1.0
for _, f in steps:
    lvl *= f
    vals.append(lvl)
a.bar(range(5), vals, color=[RUST, GOLD, OLIVE, SAGE, SKY], edgecolor=INK, lw=0.6)
for i, v in enumerate(vals):
    a.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
a.set_xticks(range(5), labs, fontsize=8)
a.set(ylabel="Crash reduction in the population", ylim=(0, 0.29),
      title="(c) From 25% efficacy to 6.3% impact (RE-AIM)")
a = ax[1, 1]                                                   # (d) CUSUM surveillance
a2 = a.twinx()
show = months <= 84
a2.bar(months[show], counts[show], color=GRID, width=0.8, zorder=0)
a2.set_ylabel("Incidents per month", color=INK)
a2.spines["right"].set_visible(True)
a.set_zorder(a2.get_zorder() + 1); a.patch.set_visible(False)
a.plot(months[show], path[show], color=RUST, lw=2, label="CUSUM statistic")
a.axhline(H, color=INK, ls="--", lw=1.2, label=f"decision interval H = {H}")
a.axvline(60.5, color=SAGE, ls=":", lw=1.6, label="regression after month 60")
if signal:
    a.plot(signal, path[signal - 1], "o", color=INK, ms=7)
    a.text(signal + 2, path[signal - 1], f"signal, month {signal}", fontsize=8,
           bbox=dict(fc=PAPER, ec="none", pad=1))
a.set(xlabel="Month after deployment", ylabel="CUSUM statistic", ylim=(-1, max(3.2 * H, 1.08 * path[show].max())),
      title="(d) Surveillance catches a bad update")
a.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("chap12_fig02_implementation.png", dpi=200)
