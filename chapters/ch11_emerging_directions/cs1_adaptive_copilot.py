"""Chapter 11 - Case Study 1: Adaptive AI Copilots for Personalized Roadway Safety Interfaces.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/11a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch11` or `python scripts/run_all.py --chapter 11`;
outputs are written to outputs/ch11/.
"""
import numpy as np
import pandas as pd
from scipy import stats, integrate
from sklearn.metrics import roc_auc_score

# ---------- A drowsiness signal whose alert baseline differs between drivers ----------
rng = np.random.default_rng(11)
SB, SW, DELTA = 1.0, 0.5, 1.5        # between-driver SD, within-driver SD, drowsy shift
N_DRV, CAL, K, WPH = 2000, 12, 120, 12   # drivers; calibration and test windows; per hour
mu = rng.normal(0, SB, N_DRV)        # each driver's alert baseline (5-min window average)
base = mu + rng.normal(0, SW, (N_DRV, CAL)).mean(axis=1)    # estimated from first hour
alert = mu[:, None] + rng.normal(0, SW, (N_DRV, K))
drowsy = mu[:, None] + DELTA + rng.normal(0, SW, (N_DRV, K))
V = {"one threshold": SB ** 2 + SW ** 2, "personal baseline": SW ** 2 * (1 + 1 / CAL)}
score = {"one threshold": (alert, drowsy),
         "personal baseline": (alert - base[:, None], drowsy - base[:, None])}
z80 = stats.norm.ppf(0.2)                # threshold that detects 80% of drowsy windows
rows = []
for name, (a, d) in score.items():
    y = np.r_[np.zeros(a.size), np.ones(d.size)]
    tau = DELTA + z80 * np.sqrt(V[name])
    rows.append((name, stats.norm.cdf(DELTA / np.sqrt(2 * V[name])),
                 roc_auc_score(y, np.r_[a.ravel(), d.ravel()]), (d > tau).mean(),
                 WPH * stats.norm.sf(tau / np.sqrt(V[name])), WPH * (a > tau).mean()))
res = pd.DataFrame(rows, columns=["scoring", "AUC_closed", "AUC_sim", "detected",
                                  "FA_per_h_closed", "FA_per_h_sim"])
print(res.round(4).to_string(index=False))

# ---------- Who bears the errors of one threshold for everyone? ----------
tg = DELTA + z80 * np.sqrt(V["one threshold"])
tp = DELTA + z80 * np.sqrt(V["personal baseline"])
cut = SW * stats.norm.isf(2 / WPH)       # window false-alarm SD units for 2 per hour
fa_g = WPH * (alert > tg).mean(axis=1)
fa_p = WPH * (score["personal baseline"][0] > tp).mean(axis=1)
flood = {"one threshold": (stats.norm.sf((tg - cut) / SB), (fa_g > 2).mean()),
         "personal baseline": (stats.norm.cdf((cut - tp) / (SW / np.sqrt(CAL))),
                               (fa_p > 2).mean())}
missed = {"one threshold": (stats.norm.cdf((tg - DELTA) / SB),
                            ((drowsy > tg).mean(axis=1) < 0.5).mean())}
print("\nDrivers with over 2 false alarms per hour (closed form, simulated):")
for k, (c, s) in flood.items():
    print(f"  {k:<18} {c:.4f}  {s:.4f}")
m_c, m_s = missed["one threshold"]
print(f"Share of drivers whose drowsy windows are caught less than half the time, one "
      f"threshold:\n  {m_c:.4f} (closed form)  {m_s:.4f} (simulated)")

# ---------- The copilot's choice: nothing, ask, or act (mixed initiative) ----------
from scipy.special import logit, expit
from scipy.optimize import brentq
PI = 0.05                                # share of windows in which the driver is drowsy
B, C = 10.0, 1.0                         # act: benefit if drowsy, cost if alert
b, c = 6.0, 0.2                          # ask: benefit if drowsy, cost if alert
p_ask = c / (b + c)
p_act = (C - c) / ((B - b) + (C - c))
p_only = C / (B + C)                     # threshold if the copilot can only act
Vp = V["personal baseline"]


def posterior(s):
    return expit(logit(PI) + (DELTA * s - DELTA ** 2 / 2) / Vp)


def s_at(p):                             # score at which the posterior equals p
    return (Vp * (logit(p) - logit(PI)) + DELTA ** 2 / 2) / DELTA


root = {p: brentq(lambda s: posterior(s) - p, -10, 10) for p in [p_ask, p_act, p_only]}
print(f"Posterior thresholds: ask {p_ask:.4f}, act {p_act:.4f}, act-only {p_only:.4f}")
for lab, p in [("ask", p_ask), ("act", p_act), ("act-only", p_only)]:
    print(f"  score threshold, {lab:<8} closed form {s_at(p):.6f}   "
          f"root finding {root[p]:.6f}")


def eu_closed(s1, s2, sd):
    # utility per hour: ask in (s1, s2], act above s2; drowsy scores shift by DELTA
    def split(shift):
        act = stats.norm.sf((s2 - shift) / sd)
        return stats.norm.sf((s1 - shift) / sd) - act, act
    ask_d, act_d = split(DELTA)
    ask_a, act_a = split(0.0)
    return WPH * (PI * (B * act_d + b * ask_d) - (1 - PI) * (C * act_a + c * ask_a))


state = rng.random((N_DRV, K)) < PI      # which windows are drowsy
raw = mu[:, None] + DELTA * state + rng.normal(0, SW, (N_DRV, K))
pers = raw - base[:, None]
pol = {"one threshold, act only": (raw, tg, tg, np.sqrt(V["one threshold"])),
       "personal, act only": (pers, s_at(p_only), s_at(p_only), np.sqrt(Vp)),
       "personal, ask or act": (pers, s_at(p_ask), s_at(p_act), np.sqrt(Vp))}
out = []
for name, (s, s1, s2, sd) in pol.items():
    act, ask = s > s2, (s > s1) & (s <= s2)
    u = WPH * np.where(state, B * act + b * ask, -(C * act + c * ask))
    se = u.mean(axis=1).std(ddof=1) / np.sqrt(N_DRV)   # drivers are independent units
    eu = eu_closed(s1, s2, sd)
    out.append((name, eu, u.mean(), (u.mean() - eu) / se,
                (act | ask)[state].mean(), WPH * (act & ~state).mean(),
                WPH * (ask & ~state).mean()))
out = pd.DataFrame(out, columns=["policy", "U_closed", "U_sim", "z", "reached",
                                 "false_act_h", "false_ask_h"])
print(f"Drowsy share of simulated windows: {state.mean():.4f} (target {PI})")
print(out.round(4).to_string(index=False))

# ---------- Learning each driver's best alert modality (Thompson sampling) ----------
AH, BH, AV, BV, T = 7, 3, 6, 4, 60       # compliance ~ beta: haptic (7, 3), voice (6, 4)
ph, pv = rng.beta(AH, BH, N_DRV), rng.beta(AV, BV, N_DRV)
win, lose = np.ones((N_DRV, 2)), np.ones((N_DRV, 2))
got, rows_ = np.zeros((N_DRV, T)), np.arange(N_DRV)
for t in range(T):
    arm = rng.beta(win, lose).argmax(axis=1)  # sample each modality's posterior, pick best
    y = rng.random(N_DRV) < np.where(arm == 0, ph, pv)
    win[rows_, arm] += y
    lose[rows_, arm] += ~y
    got[:, t] = y
fixed = (rng.random((N_DRV, T)) < ph[:, None]).mean()
oracle = (rng.random((N_DRV, T)) < np.maximum(ph, pv)[:, None]).mean()


def e_max():                             # E[max(haptic, voice)] for independent betas
    def f(x):
        tail = AV / (AV + BV) * stats.beta.sf(x, AV + 1, BV)    # E[voice; voice > x]
        return stats.beta.pdf(x, AH, BH) * (x * stats.beta.cdf(x, AV, BV) + tail)
    return integrate.quad(f, 0, 1)[0]


print(f"Drivers who comply more with voice: {(pv > ph).mean():.3f}")
print(f"Haptic for everyone   expected {AH / (AH + BH):.4f}   simulated {fixed:.4f}")
print(f"Each driver's best    expected {e_max():.4f}   simulated {oracle:.4f}")
print(f"Thompson sampling     all {T} alerts {got.mean():.4f}   first 10 "
      f"{got[:, :10].mean():.4f}   last 20 {got[:, -20:].mean():.4f}")
learned = (win / (win + lose)).argmax(axis=1) == np.where(ph >= pv, 0, 1)
print(f"Share of drivers whose learned favorite is their true best: {learned.mean():.3f}")

# ---------- Figure ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
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
a = ax[0, 0]                                                   # (a) ROC
fpr = np.linspace(0, 1, 400)
for name, col in [("one threshold", RUST), ("personal baseline", SAGE)]:
    sd = np.sqrt(V[name])
    tpr = stats.norm.sf(stats.norm.isf(np.clip(fpr, 1e-9, 1)) - DELTA / sd)
    a.plot(fpr, tpr, color=col,
           lw=2.2, label=f"{name}: AUC {stats.norm.cdf(DELTA / np.sqrt(2 * V[name])):.3f}")
    al, dr = score[name]
    idx = rng.choice(al.size, 20_000, replace=False)       # windows from all drivers
    f_, t_, _ = roc_curve(np.r_[np.zeros(20_000), np.ones(20_000)],
                          np.r_[al.ravel()[idx], dr.ravel()[idx]])
    keep = np.searchsorted(f_, np.linspace(0.005, 0.95, 28))
    a.plot(f_[keep], t_[keep], "o", color=col, mec=INK, ms=3.5)
a.plot([0, 1], [0, 1], color=INK, lw=0.8, ls=":")
a.set(xlabel="False-alarm rate per alert window", ylabel="Drowsy windows detected",
      title="(a) Personal baselines sharpen detection")
a.legend(fontsize=8, frameon=False, loc="lower right")
a = ax[0, 1]                                                   # (b) who gets the alarms
bins = np.linspace(0, 12, 49)
a.hist(fa_g, bins, color=RUST, alpha=0.75, label="one threshold")
a.hist(fa_p, bins, color=SAGE, alpha=0.75, label="personal baseline")
a.axvline(2, color=INK, ls="--", lw=1)
a.set_yscale("log")
a.set(xlabel="False alarms per hour for each driver (at 80% detection)",
      ylabel="Drivers", title="(b) One threshold floods some drivers")
a.legend(fontsize=8, frameon=False, loc="upper right")
a = ax[1, 0]                                                   # (c) utility of each action
pp = np.linspace(0, 0.4, 400)
a.plot(pp, 0 * pp, color=INK, lw=1.6, label="do nothing")
a.plot(pp, pp * b - (1 - pp) * c, color=GOLD, lw=2.2, label="ask")
a.plot(pp, pp * B - (1 - pp) * C, color=RUST, lw=2.2, label="act")
for p_, lab in [(p_ask, "ask"), (p_act, "act")]:
    a.axvline(p_, color=INK, ls=":", lw=1)
    a.text(p_ + 0.004, 2.6, f"{lab} above\n{p_:.3f}", fontsize=7.6)
a.set(ylim=(-1.2, 3.2), xlabel="Probability the driver is drowsy",
      ylabel="Expected utility of the action", title="(c) When to ask and when to act")
a.legend(fontsize=8, frameon=False, loc="lower right")
a = ax[1, 1]                                                   # (d) learning a modality
tt = np.arange(1, T + 1)
a.plot(tt, got.mean(axis=0), color=SAGE, lw=2, label="Thompson sampling")
a.axhline(AH / (AH + BH), color=RUST, ls="--", lw=1.6, label="haptic for everyone")
a.axhline(e_max(), color=INK, ls=":", lw=1.6, label="each driver's best")
a.set(ylim=(0.6, 0.8), xlabel="Alert number", ylabel="Share of alerts complied with",
      title="(d) Learning each driver's best alert")
a.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("chap11_fig01_copilot.png", dpi=200)
