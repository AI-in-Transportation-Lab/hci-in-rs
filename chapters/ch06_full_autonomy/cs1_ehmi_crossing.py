"""Chapter 6 - Case Study 1: eHMI Intent Signaling for Pedestrian Interaction with Autonomous Vehicles.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/06a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch06` or `python scripts/run_all.py --chapter 6`;
outputs are written to outputs/ch06/.
"""
import numpy as np
import pandas as pd

# ---------- Synthetic VR crossing experiment: three between-subject groups ----------
rng = np.random.default_rng(6)
# approach speed (m/s), lane width (m), walk speed (m/s)
V0, LANE, WALK = 30 / 3.6, 3.5, 1.4
# deceleration-detection threshold (m), motor delay (s)
DELTA, MOTOR = 1.2, 0.35
GROUPS = {"K": "Kinematics only", "R": "Reliable eHMI", "F": "eHMI with two faults"}
# participants per group, trials, fault positions
N_PER, N_TRIALS, FAULTS = 40, 40, (24, 33)
A_UP, A_DOWN, W0 = 0.10, 0.70, 0.40           # reliance learning rates and initial reliance

rows = []
for g in GROUPS:
    for p in range(N_PER):
        delta = DELTA * np.exp(rng.normal(0, 0.25))      # individual detection threshold
        risk = rng.normal(0, 0.8)                        # individual gap-acceptance offset
        w, seen = W0, 0                                  # reliance on eHMI, eHMI exposures
        order = np.ones(N_TRIALS, int)                   # 1 = vehicle yields, 0 = does not
        free = [t for t in range(N_TRIALS) if t + 1 not in FAULTS]
        order[[f - 1 for f in FAULTS]] = 0
        # 24 yielding, 16 non-yielding trials
        order[rng.choice(free, 14, replace=False)] = 0
        for t in range(N_TRIALS):
            trial, yields = t + 1, order[t]
            tta = 4.0 if trial in FAULTS else rng.choice([3.0, 4.0, 5.0])
            fault = g == "F" and trial in FAULTS
            ehmi_on = (g != "K" and yields == 1) or fault          # 'yielding' signal shown
            relies = ehmi_on and rng.random() < w
            t_ehmi = 0.45 + 1.10 * np.exp(-seen / 4) + MOTOR       # comprehension improves
            # constant decel to stop 2 m short
            if yields:
                a = V0 ** 2 / (2 * (V0 * tta - 2.0))
                # deviation reaches threshold
                t_kin = np.sqrt(2 * delta / a) + MOTOR
                cit = (min(t_kin, t_ehmi) if relies else t_kin) + rng.normal(0, 0.20)
                crossed, cit = 1, max(cit, 0.30)
            else:
                if relies:                                         # trusts a false signal
                    crossed, cit = 1, t_ehmi + rng.normal(0, 0.15)
                else:                                              # ordinary gap acceptance
                    p_go = 1 / (1 + np.exp(-(-7.0 + 1.1 * tta + risk)))
                    crossed = int(rng.random() < p_go)
                    cit = 0.5 + abs(rng.normal(0, 0.2)) if crossed else np.nan
            margin = tta - cit - LANE / WALK if (crossed and not yields) else np.nan
            # experience updates reliance
            if ehmi_on:
                w += A_UP * (1 - w) if yields else A_DOWN * (0 - w)
                seen += 1
            trust = (np.clip(np.round(1 + 6 * w + rng.normal(0, 0.6)), 1, 7)
                     if g != "K" else np.nan)
            rows.append((g, f"{g}{p:02d}", trial, yields, tta, int(fault), crossed, cit,
                         margin, trust))

df = pd.DataFrame(rows, columns=["group", "pid", "trial", "yields", "tta", "fault",
                                 "crossed", "cit", "margin", "trust"])
print(df.groupby("group", sort=False).agg(participants=("pid", "nunique"),
      trials=("trial", "size"), yield_trials=("yields", "sum")).to_string())
yt = df[df.yields == 1]
print("\nMean crossing initiation time on yielding trials (s), by block of 10 trials:")
print(yt.assign(block=(yt.trial - 1) // 10 + 1).pivot_table(index="group", columns="block",
      values="cit", aggfunc="mean", sort=False).round(2).to_string())

# ---------- Efficiency: mixed model of crossing initiation time ----------
import statsmodels.formula.api as smf
from scipy.stats import norm

late = yt.assign(late=(yt.trial > 20).astype(int))
mm = smf.mixedlm("cit ~ C(group, Treatment('K')) * late", late, groups="pid").fit(reml=True)
print("Linear mixed model, CIT (s), random intercept per participant:")
for name, lab in [("Intercept", "Kinematics only, trials 1-20"),
                  ("C(group, Treatment('K'))[T.R]", "Reliable eHMI vs K, trials 1-20"),
                  ("C(group, Treatment('K'))[T.R]:late", "  extra effect, trials 21-40"),
                  ("C(group, Treatment('K'))[T.F]", "Faulty eHMI vs K, trials 1-20"),
                  ("C(group, Treatment('K'))[T.F]:late", "  extra effect, trials 21-40")]:
    lo, hi = mm.conf_int().loc[name]
    print(f"  {lab:<34}{mm.params[name]:7.2f}  [{lo:5.2f}, {hi:5.2f}]")

# ---------- Safety: crossings started in front of a non-yielding vehicle ----------
def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h

ny = df[df.yields == 0].assign(
    pos=lambda d: np.where(d.trial.isin(FAULTS), "fault", "other"))
print("\nFalse starts on non-yielding trials (share of trials, 95% Wilson interval):")
for g in GROUPS:
    for pos in ["other", "fault"]:
        s = ny[(ny.group == g) & (ny.pos == pos)]
        k, n = int(s.crossed.sum()), len(s)
        lo, hi = wilson(k, n)
        unsafe = int((s.margin < 1.0).sum())
        print(f"  {GROUPS[g]:<22}{pos:<7}{k:4d}/{n:<4d}= {k / n:5.3f}  [{lo:.3f}, {hi:.3f}]"
              f"   margin < 1 s: {unsafe}")

# ---------- Discrimination: signal detection measures per participant ----------
W_GO = 2.0                                     # 'go' = crossing started within 2 s
def sdt(d):
    hits = ((d.yields == 1) & (d.cit <= W_GO)).sum()
    fas = ((d.yields == 0) & (d.crossed == 1)).sum()
    h = (hits + 0.5) / ((d.yields == 1).sum() + 1)          # log-linear correction
    f = (fas + 0.5) / ((d.yields == 0).sum() + 1)
    return pd.Series({"H": h, "F": f, "dprime": norm.ppf(h) - norm.ppf(f),
                      "c": -0.5 * (norm.ppf(h) + norm.ppf(f))})

sd = df.groupby(["group", "pid"], sort=False)[["yields", "cit", "crossed"]].apply(sdt)
summary = sd.groupby("group", sort=False).agg(["mean", "sem"])
print(f"\nSignal detection (go = crossing within {W_GO:.0f} s), "
      "mean +/- s.e. over participants:")
for g in GROUPS:
    r = summary.loc[g]
    print(f"  {GROUPS[g]:<22}H {r[('H', 'mean')]:.3f}  F {r[('F', 'mean')]:.3f}  "
          f"d' {r[('dprime', 'mean')]:.2f} +/- {r[('dprime', 'sem')]:.2f}  "
          f"c {r[('c', 'mean')]:+.2f} +/- {r[('c', 'sem')]:.2f}")

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
col = {"K": SAGE, "R": GOLD, "F": RUST}
fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
for g in GROUPS:                                       # (a) learning curve, 5-trial window
    m = yt[yt.group == g].groupby("trial").cit.mean().rolling(5, center=True,
                                                              min_periods=3).mean()
    ax[0, 0].plot(m.index, m.values, color=col[g], lw=2, label=GROUPS[g])
for g in ["R", "F"]:                                   # (b) trust ratings after each trial
    m = df[df.group == g].groupby("trial").trust.mean()
    ax[0, 1].plot(m.index, m.values, "o-", ms=3, color=col[g], lw=1.4, label=GROUPS[g])
for a in ax[0]:
    for f in FAULTS:
        a.axvline(f, color=RUST, ls=":", lw=1.2)
ax[0, 0].set(xlabel="Trial", ylabel="Crossing initiation time (s)", ylim=(0.6, 2.1),
             title="(a) Yielding trials: time to start crossing")
ax[0, 1].set(xlabel="Trial", ylabel="Mean trust rating (1-7)", ylim=(1, 7),
             title="(b) Self-reported trust in the eHMI")
ax[0, 0].legend(fontsize=8, frameon=False, loc="lower left")
ax[0, 1].legend(fontsize=8, frameon=False, loc="lower right")
ax[0, 1].text(FAULTS[0] + 0.5, 6.6, "faults", color=RUST, fontsize=8)
xs = np.arange(3)                                      # (c) false starts by trial type
for j, pos in enumerate(["other", "fault"]):
    vals, errs = [], []
    for g in GROUPS:
        s = ny[(ny.group == g) & (ny.pos == pos)]
        k, n = int(s.crossed.sum()), len(s)
        lo, hi = wilson(k, n)
        vals.append(k / n); errs.append([k / n - lo, hi - k / n])
    ax[1, 0].bar(xs + (j - 0.5) * 0.36, vals, 0.34, yerr=np.array(errs).T, ecolor=INK,
                 color=[SKY, RUST][j], label=["other non-yielding trials",
                                              "fault positions (trials 24, 33)"][j])
ax[1, 0].set_xticks(xs, [GROUPS[g].replace(" with ", "\nwith ") for g in GROUPS],
                    fontsize=8)
ax[1, 0].set(ylabel="Share of trials with a false start", ylim=(0, 0.85),
             title="(c) Starting to cross before a non-yielding vehicle")
ax[1, 0].legend(fontsize=8, frameon=False, loc="upper left")
# (d) ROC space, isosensitivity curves
fgrid = np.linspace(0.001, 0.999, 300)
for g in GROUPS:
    d, c = summary.loc[g, ("dprime", "mean")], summary.loc[g, ("c", "mean")]
    ax[1, 1].plot(fgrid, norm.cdf(d + norm.ppf(fgrid)), color=col[g], lw=1.2, alpha=0.8)
    ax[1, 1].plot(norm.cdf(-d / 2 - c), norm.cdf(d / 2 - c), "o", ms=8, color=col[g],
                  mec=INK, label=f"{GROUPS[g]} (d' = {d:.2f}, c = {c:+.2f})")
ax[1, 1].plot([0, 1], [0, 1], "--", color=OLIVE, lw=1)
ax[1, 1].set(xlim=(0, 0.5), ylim=(0.5, 1.0), xlabel="False-alarm rate F",
             ylabel="Hit rate H", title="(d) Discrimination of yielding intent")
ax[1, 1].legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("chap06_fig01_ehmi_crossing.png", dpi=200)
