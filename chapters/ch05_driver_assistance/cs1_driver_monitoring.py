"""Chapter 5 - Case Study 1: Driver Monitoring and Engagement in SAE Level 2 Automation.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/05a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch05` or `python scripts/run_all.py --chapter 5`;
outputs are written to outputs/ch05/.
"""
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Thirty minutes of Level 2 driving: where are the driver's eyes? ----------
rng = np.random.default_rng(5)
T, N = 1800.0, 4000                    # drive length (s); drivers per strategy
ON_MED, ON_SD = 2.2, 0.60              # glances to the road: median (s), log-SD
OFF_MED, OFF_SD = 1.0, 0.55            # glances away, at the start of the drive
LAT = 0.3                              # camera detection latency (s)
# escalation: (threshold s, probability the driver looks back, reaction s)
LADDER = [(2.0, 0.5, 0.8), (3.5, 0.9, 0.7), (5.0, 1.0, 0.5)]
DRIFT = {"no monitoring": 0.6, "hands-on-wheel": 0.5, "camera + escalation": 0.2}


def escalate(g):
    out, alerts = g.copy(), np.zeros((len(g), 3), bool)
    done = np.zeros(len(g), bool)
    for lvl, (th, p_back, rt) in enumerate(LADDER):
        hit = ~done & (g > th + LAT)
        alerts[hit, lvl] = True
        back = hit & (rng.random(len(g)) < p_back)
        out[back] = np.minimum(g[back], th + LAT + rt)
        done |= back
    return out, alerts


def drive(strategy, n=N, drift=None, hazard=True):
    c = DRIFT[strategy] if drift is None else drift
    t = np.zeros(n)
    off_tot, long2, long4, count = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    alerts = np.zeros((n, 3))
    h = rng.uniform(600, T, n)                       # hazard time, after 10 minutes
    resid = np.full(n, np.nan)
    while (t < T).any():
        live = t < T
        on = ON_MED * np.exp(ON_SD * rng.normal(size=n))
        g = OFF_MED * (1 + c * np.minimum(t, T) / T) * np.exp(OFF_SD * rng.normal(size=n))
        if strategy == "camera + escalation":
            g, a = escalate(g)
            alerts += a * live[:, None]
        on_end, off_end = t + on, t + on + g
        inside = live & (off_end <= T)
        seen = np.clip(np.minimum(off_end, T) - np.minimum(on_end, T), 0, None)
        off_tot += np.where(live, seen, 0)
        long2 += inside & (g > 2)
        long4 += inside & (g > 4)
        count += inside
        r_on = np.isnan(resid) & (h >= t) & (h < on_end)
        r_off = np.isnan(resid) & (h >= on_end) & (h < off_end)
        resid[r_on] = 0.0
        resid[r_off] = off_end[r_off] - h[r_off]
        t = off_end
    return dict(off=off_tot / T, long2=long2 / (T / 3600), long4=long4 / (T / 3600),
                alerts=alerts / (T / 3600), resid=resid, glances=count)


res = {s: drive(s) for s in DRIFT}
rows = []
for s, r in res.items():
    rows.append((s, r["off"].mean(), r["long2"].mean(), r["long4"].mean(),
                 r["alerts"].sum(axis=1).mean(), (r["resid"] > 0).mean(),
                 np.nanmean(np.where(r["resid"] > 0, r["resid"], np.nan))))
tab = pd.DataFrame(rows, columns=["strategy", "eyes_off", "gt2s_per_h", "gt4s_per_h",
                                  "alerts_h", "P_off_hz", "wait_s"])
print(tab.round(4).to_string(index=False))

# ---------- Check the simulator against renewal theory (no drift) ----------
e_on, e_off = ON_MED * np.exp(ON_SD ** 2 / 2), OFF_MED * np.exp(OFF_SD ** 2 / 2)
chk = drive("no monitoring", n=6000, drift=0.0)
print(f"\nNo drift: share of time eyes off {e_off / (e_on + e_off):.4f} (renewal theory), "
      f"{chk['off'].mean():.4f} (simulation)")
lg = 3600 / (e_on + e_off) * stats.lognorm.sf(2, OFF_SD, scale=OFF_MED)
print(f"Long glances per hour: {lg:.3f} (closed form), "
      f"{chk['long2'].mean():.3f} (simulation)")

# ---------- Where should the first alert come? ----------
BASE = list(LADDER)
sweep = []
for th1 in [1.5, 2.0, 2.5, 3.0, 4.0]:
    LADDER[:] = [(th1, 0.5, 0.8), (th1 + 1.5, 0.9, 0.7), (th1 + 3.0, 1.0, 0.5)]
    r = drive("camera + escalation", n=3000)
    sweep.append((th1, r["alerts"].sum(axis=1).mean(), r["long4"].mean(),
                  (r["resid"] > 1).mean()))
    if th1 == 2.0:                                   # check against the closed form
        z = drive("camera + escalation", n=3000, drift=0.0)
        per_glance = (z["alerts"][:, 0] * (T / 3600)).sum() / z["glances"].sum()
        closed = stats.lognorm.sf(th1 + LAT, OFF_SD, scale=OFF_MED)
LADDER[:] = BASE
sweep = pd.DataFrame(sweep, columns=["first_alert_s", "alerts_per_h", "glances_gt4s_per_h",
                                     "P_wait_gt_1s"])
print(sweep.round(4).to_string(index=False))
print(f"\nNo drift, first alert at 2.0 s: alerts per glance {per_glance:.4f} (simulation),"
      f"\n  {closed:.4f} (closed form)")

# ---------- Can the camera be trusted? False alerts from frame errors ----------
# frames per second; frame error rate when attentive
FPS, Q = 2, 0.03
SE = 0.95                                # frames correctly flagged when eyes are off


def false_per_hour(k, p11, q=Q):
    p01 = q * (1 - p11) / (1 - q)        # Markov errors with stationary rate q
    return 3600 * FPS * (1 - q) * p01 * p11 ** (k - 1)


def false_sim(k, p11, hours=400, q=Q):
    n = int(hours * 3600 * FPS)
    p01 = q * (1 - p11) / (1 - q)
    u, e = rng.random(n), np.zeros(n, bool)
    e[0] = u[0] < q
    for i in range(1, n):
        e[i] = u[i] < (p11 if e[i - 1] else p01)
    run, alerts = 0, 0
    for x in e:
        run = run + 1 if x else 0
        alerts += run == k
    return alerts / hours


def delay_frames(k, p=SE):               # expected frames until k flags in a row
    return (1 - p ** k) / ((1 - p) * p ** k)


rows3 = []
for k in [2, 3, 4, 6]:
    rows3.append((k, k / FPS, false_per_hour(k, Q), false_per_hour(k, 0.6),
                  false_sim(k, 0.6, hours=150), delay_frames(k) / FPS))
fa = pd.DataFrame(rows3, columns=["k", "min_delay_s", "indep_per_h", "bursty_per_h",
                                  "bursty_sim", "delay_s"])
print(fa.round(4).to_string(index=False))
sims = []
for _ in range(20_000):
    run, i = 0, 0
    while run < 4:
        run = run + 1 if rng.random() < SE else 0
        i += 1
    sims.append(i)
print(f"\nFrames to detect a real glance away with 4 in a row: {delay_frames(4):.4f} "
      f"(closed form),\n  {np.mean(sims):.4f} (simulation)")

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
col = {"no monitoring": RUST, "hands-on-wheel": GOLD, "camera + escalation": SAGE}
a = ax[0, 0]                                                   # (a) long glances
x = np.arange(3)
a.bar(x - 0.2, tab.gt2s_per_h / 10, 0.38, color=SKY, edgecolor=INK, lw=0.5,
      label="over 2 s (tens per hour)")
a.bar(x + 0.2, tab.gt4s_per_h, 0.38, color=RUST, edgecolor=INK, lw=0.5,
      label="over 4 s (per hour)")
a.set_xticks(x, [d.replace(" + ", " +\n") for d in DRIFT], fontsize=6.5)
a.set(ylabel="Long glances away from the road",
      title="(a) Hands on the wheel is not eyes on the road", ylim=(0, 24))
a.legend(fontsize=6.5)
a = ax[0, 1]                                                   # (b) threshold trade-off
a.plot(sweep.alerts_per_h, sweep.glances_gt4s_per_h, "-o", color=SAGE, mec=INK, lw=1.3, ms=4.5)
for _, r in sweep.iterrows():
    a.annotate(f"{r.first_alert_s:.1f} s", (r.alerts_per_h, r.glances_gt4s_per_h),
               textcoords="offset points", xytext=(3.7, 3.7), fontsize=6.5)
a.set_xscale("log")
a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
a.set(xlabel="Alerts per hour (log scale)", ylabel="Glances over 4 s per hour",
      title="(b) Nuisance against protection")
# (c) readiness at the hazard
a = ax[1, 0]
w = np.linspace(0, 4, 300)
for s, r in res.items():
    rr = np.nan_to_num(r["resid"], nan=0.0)
    a.plot(w, [(rr > v).mean() for v in w], color=col[s], lw=1.4, label=s)
a.set(xlabel="Seconds until the eyes return to the road",
      ylabel="Share of drivers still looking away",
      title="(c) When the hazard appears", ylim=(0, 0.42))
a.legend(fontsize=6.5)
a = ax[1, 1]                                                   # (d) false alerts
a.semilogy(fa.k, np.maximum(fa.indep_per_h, 1e-4), "-o", color=GOLD,
           mec=INK, lw=1.3, label="independent errors")
a.semilogy(fa.k, fa.bursty_per_h, "-o", color=RUST, mec=INK, lw=1.3,
           label="bursty errors (closed form)")
a.semilogy(fa.k, fa.bursty_sim, "x", color=INK, ms=5.8, mew=1.3,
           label="bursts, simulated")
a.set(xlabel="Flagged frames in a row before alerting (2 per second)",
      ylabel="False alerts per hour, attentive driver",
      title="(d) Correlated errors defeat the filter")
a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
a.legend(fontsize=6, loc="center right", bbox_to_anchor=(1, 0.6))
fig.tight_layout(); fig.savefig("chap05_fig01_monitoring.png", dpi=300)
