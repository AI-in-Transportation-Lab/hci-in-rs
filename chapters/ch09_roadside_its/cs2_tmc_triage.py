"""Chapter 9 - Case Study 2: AI-Assisted Traffic Management Center Visualization for Safety Decisions.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/09a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch09` or `python scripts/run_all.py --chapter 9`;
outputs are written to outputs/ch09/.
"""
import heapq
import math
import numpy as np
import pandas as pd
from scipy import stats


# ---------- Closed-form waits for a TMC verification queue (minutes) ----------
def erlang_c(c, a):
    # probability that an alert must wait: M/M/c with offered load a = lambda / mu
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - a / c)
    return top / (s + top)


def wait_fcfs(lam, mu, c):
    return 60 * erlang_c(c, lam / mu) / (c * mu - lam)


def wait_priority(lams, mu, c):
    # non-preemptive priority with equal exponential service (Cobham), classes in order
    w0, out, s_prev = erlang_c(c, sum(lams) / mu) / (c * mu), [], 0.0
    for lam in lams:
        s = s_prev + lam / (c * mu)
        out.append(60 * w0 / ((1 - s_prev) * (1 - s)))
        s_prev = s
    return out


# ---------- Discrete-event simulation: c operators, any priority key ----------
def simulate(arr, svc, prio, c):
    n, start = len(arr), np.empty(len(arr))
    free, queue, i = [0.0] * c, [], 0
    for _ in range(n):
        t = heapq.heappop(free)              # the next operator to become free
        if not queue and arr[i] > t:
            t = arr[i]                       # idle until the next alert arrives
        while i < n and arr[i] <= t:
            heapq.heappush(queue, (prio[i], arr[i], i))
            i += 1
        _, _, j = heapq.heappop(queue)       # lowest key first, then earliest arrival
        start[j] = t
        heapq.heappush(free, t + svc[j])
    return 60 * (start - arr)


def batch_se(x, k=20):
    # standard error by batch means, because successive waits are correlated
    return np.std([b.mean() for b in np.array_split(x, k)], ddof=1) / np.sqrt(k)


# ---------- Cross-check: 24 alerts/h, 4-min checks, 2 operators ----------
rng = np.random.default_rng(91)
LAM, MU, C, n = 24.0, 15.0, 2, 200_000
arr, svc = np.cumsum(rng.exponential(1 / LAM, n)), rng.exponential(1 / MU, n)
high = rng.random(n) < 0.25
w_f = simulate(arr, svc, np.zeros(n), C)
w_p = simulate(arr, svc, np.where(high, 0, 1), C)
an = {"FCFS, all alerts": wait_fcfs(LAM, MU, C)}
an["priority, high class"], an["priority, low class"] = wait_priority(
    [0.25 * LAM, 0.75 * LAM], MU, C)
sim = {"FCFS, all alerts": w_f, "priority, high class": w_p[high],
       "priority, low class": w_p[~high]}
chk = pd.DataFrame([(k, an[k], sim[k].mean(), (sim[k].mean() - an[k]) / batch_se(sim[k]))
                    for k in an],
                   columns=["mean wait (min)", "closed_form", "simulation", "z"])
print(f"P(wait) = {erlang_c(C, LAM / MU):.4f}; utilization {LAM / (C * MU):.2f}")
print(chk.round(3).to_string(index=False))
cons_a = 0.25 * an["priority, high class"] + 0.75 * an["priority, low class"]
print(f"Conservation law: weighted priority wait {cons_a:.3f} (closed form), "
      f"\n{w_p.mean():.3f} (simulation); FCFS {an['FCFS, all alerts']:.3f}")

# ---------- Half a year of AI alerts at a regional TMC ----------
from scipy.special import expit, logit
DAYS, OPS, VERIFY = 180, 2, 4.0                  # days, operators, mean verification (min)
hours = np.arange(24)
peak = np.exp(-0.5 * ((hours - 8) / 1.5) ** 2) + np.exp(-0.5 * ((hours - 17) / 2.0) ** 2)
# incidents, non-incident events per h
INC, CAND = 1.0 + 2.0 * peak, 80.0 + 120.0 * peak
B_TRUE, B_FALSE = (5, 2), (2, 5)                 # beta distributions of the AI score
MAJOR, FLAG_OK = 0.25, 0.80                      # lane-blocking share; flag accuracy
# minutes until a missed incident is phoned in
CALL = 10.0
P_BASE, OR_MIN = {True: 0.08, False: 0.02}, 1.028   # secondary risk; odds per minute


def events(rate, true):
    k = rng.poisson(np.tile(rate, DAYS))
    t = np.repeat(np.arange(24 * DAYS), k) + rng.random(k.sum())     # hours
    s = rng.beta(*(B_TRUE if true else B_FALSE), k.sum())
    major = (rng.random(k.sum()) < MAJOR) & true
    flag = np.where(major, rng.random(k.sum()) < FLAG_OK, rng.random(k.sum()) > FLAG_OK)
    return t, s, major, flag


ti, si, mi, fi = events(INC, True)
tf, sf, mf, ff = events(CAND, False)
h_i, h_f = (ti % 24).astype(int), (tf % 24).astype(int)
# share of events that are incidents, by hour
prior = INC / (INC + CAND)
LR_FLAG = {True: (MAJOR * FLAG_OK + (1 - MAJOR) * (1 - FLAG_OK)) / (1 - FLAG_OK),
           False: (MAJOR * (1 - FLAG_OK) + (1 - MAJOR) * FLAG_OK) / FLAG_OK}


def posterior(s, flag, h):
    # calibrated probability that an alert is a real incident, from score, flag, and hour
    lr = stats.beta.pdf(s, *B_TRUE) / stats.beta.pdf(s, *B_FALSE)
    lr = lr * np.where(flag, LR_FLAG[True], LR_FLAG[False])
    return expit(logit(prior[h]) + np.log(lr))


def p_major(flag):
    hit = MAJOR * FLAG_OK / (MAJOR * FLAG_OK + (1 - MAJOR) * (1 - FLAG_OK))
    miss = MAJOR * (1 - FLAG_OK) / (MAJOR * (1 - FLAG_OK) + (1 - MAJOR) * FLAG_OK)
    return np.where(flag, hit, miss)


def load(tau):
    # closed form: alerts per hour by hour of day, and operator utilization
    alerts = CAND * stats.beta.sf(tau, *B_FALSE) + INC * stats.beta.sf(tau, *B_TRUE)
    rho = (alerts + INC * stats.beta.cdf(tau, *B_TRUE)).mean() * VERIFY / 60 / OPS
    return alerts, rho


def run(tau, policy):
    det, fa = si >= tau, sf >= tau               # incidents detected; false alerts raised
    miss = ~det
    crn = np.random.default_rng(int(round(tau * 1000)))    # same draws for every policy
    t_call = ti[miss] + crn.exponential(CALL, miss.sum()) / 60
    t = np.concatenate([ti[det], tf[fa], t_call])
    s = np.concatenate([si[det], sf[fa], np.ones(miss.sum())])
    fl = np.concatenate([fi[det], ff[fa], mi[miss]])
    h = np.concatenate([h_i[det], h_f[fa], h_i[miss]])
    called = np.repeat([False, True], [det.sum() + fa.sum(), miss.sum()])
    harm = np.where(called, 9.0, posterior(s, fl, h) * (1 + 4 * p_major(fl)))
    key = {"FCFS": np.where(called, -1.0, 0.0),
           "score": -np.where(called, 9.0, s), "harm": -harm}[policy]
    o, svc = np.argsort(t, kind="stable"), crn.exponential(VERIFY / 60, len(t))
    w = np.empty(len(t))
    w[o] = simulate(t[o], svc[o], key[o], OPS)
    nd, nf = det.sum(), fa.sum()
    delay = np.empty(len(ti))                    # minutes from incident to verification
    delay[det] = 1.0 + w[:nd] + 60 * svc[:nd]
    delay[miss] = (t_call - ti[miss]) * 60 + w[nd + nf:] + 60 * svc[nd + nf:]
    base = np.where(mi, P_BASE[True], P_BASE[False])
    p_sec = expit(logit(base) + np.log(OR_MIN) * delay)
    per10 = np.bincount((t[:nd + nf] * 6).astype(int), minlength=144 * DAYS) / OPS
    return {"tau": tau, "policy": policy, "rho": load(tau)[1], "per10": per10.mean(),
            "peak10": per10.reshape(DAYS, 24, 6).mean(axis=(0, 2)).max(),
            "detected": det.mean(), "wait_major": w[:nd][mi[det]].mean(),
            "sec_per_1000": 1000 * p_sec.mean()}


TAUS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70]
print(f"{len(ti):,} incidents and {len(tf):,} non-incident events in {DAYS} days")
for tau in TAUS:
    if load(tau)[1] >= 1:
        print(f"tau {tau:.2f}: utilization {load(tau)[1]:.3f} >= 1; unstable, not run")
grid = pd.DataFrame([run(tau, p) for tau in TAUS if load(tau)[1] < 1
                     for p in ["FCFS", "score", "harm"]])
print(grid.round(3).to_string(index=False))
a5 = load(0.5)[0]
s5 = grid[(grid.tau == 0.5) & (grid.policy == "FCFS")].iloc[0]
print(f"\nCross-check at tau 0.50, alerts per operator per 10 min: mean "
      f"{a5.mean() / 6 / OPS:.3f} vs {s5.per10:.3f}; \npeak hour "
      f"{a5.max() / 6 / OPS:.3f} vs {s5.peak10:.3f} (closed form vs simulation)")
best = grid.loc[grid.sec_per_1000.idxmin()]
print(f"Fewest secondary crashes: tau {best.tau:.2f}, {best.policy} policy, "
      f"{best.sec_per_1000:.2f} per 1,000 incidents")

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
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.45))
for tau, c in [(0.45, RUST), (0.50, GOLD), (0.55, SAGE), (0.60, SKY)]:   # (a) alarm rate
    ax[0].plot(hours, load(tau)[0] / 6 / OPS, color=c, lw=1.2, label=f"{tau:.2f}")
ax[0].axhspan(0, 1, color="#DCE5E1", zorder=0)
ax[0].axhline(2, color=RUST, ls="--", lw=0.6)
ax[0].text(10.2, 0.08, "about 1 per 10 min", fontsize=6)
ax[0].text(0.15, 1.93, "2 per 10 min", fontsize=6, color=RUST, va="top")
ax[0].set(xlabel="Hour of day", ylabel="Alerts per operator per 10 min", xlim=(0, 23), ylim=(0, 4.2),
          title="(a) Alert load vs. guidance")
ax[0].legend(fontsize=6, loc="upper center", ncol=2, title="threshold",
             title_fontsize=6)
x = np.arange(3)                                                         # (b) waits
for k, (tau, c) in enumerate([(0.45, RUST), (0.50, GOLD), (0.55, SAGE)]):
    r = grid[grid.tau == tau].set_index("policy").loc[["FCFS", "score", "harm"]]
    ax[1].bar(x + (k - 1) * 0.26, r.wait_major, 0.25, color=c, edgecolor=INK, lw=0.5,
              label=f"threshold {tau:.2f}")
ax[1].set_xticks(x, ["arrival\norder", "highest\nscore", "expected\nharm"], fontsize=6)
ax[1].set(ylabel="Mean wait, lane-blocking incidents (min)",
          title="(b) Triage order and waiting")
ax[1].legend(fontsize=6)
for p, c, m in [("FCFS", RUST, "o"), ("score", GOLD, "s"), ("harm", SAGE, "D")]:  # (c)
    r = grid[grid.policy == p]
    ax[2].plot(r.tau, r.sec_per_1000, "-" + m, color=c, mec=INK, ms=3, lw=1.2,
               label={"FCFS": "arrival order", "score": "highest score",
                      "harm": "expected harm"}[p])
ax[2].axvspan(0.38, 0.425, color="#F6DCC4", zorder=0)
ax[2].text(0.402, 52, "over\ncapacity", fontsize=6, ha="center")
ax[2].set(xlim=(0.38, 0.72), xlabel="Alert threshold on the AI score",
          ylabel="Secondary crashes per 1,000 incidents",
          title="(c) Too many alerts cost more")
ax[2].legend(fontsize=6, loc="upper right")
fig.tight_layout(); fig.savefig("chap09_fig02_tmc_triage.png", dpi=300)
