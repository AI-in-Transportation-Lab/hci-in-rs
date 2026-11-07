"""Chapter 4 - Case Study 2: Comparing Touch, Voice, Gesture, and Multimodal Interfaces for Driver Safety.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/04a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch04` or `python scripts/run_all.py --chapter 4`;
outputs are written to outputs/ch04/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# ---------- Five ways to choose one of 12 saved destinations while driving ----------
rng = np.random.default_rng(44)
FA, FB = 0.25, 0.14                        # Fitts: MT = a + b log2(D / W + 1)
HA, HB = 0.20, 0.15                        # Hick: decision = a + b log2(n + 1)
D_MM, W_MM, S_MOTOR, S_VIB = 180.0, 18.0, 3.0, 3.5   # reach, width, endpoint SDs (mm)
REAIM = 0.40                               # extra time to re-aim after a miss (s)
# attempt, recovery, error, glance
T_VOICE, R_VOICE, E_VOICE, CONFIRM = 2.7, 1.0, 0.12, 0.6
# per gesture, gestures, error, looking
GEST, N_GEST, E_GEST, LOOK_GEST = 0.9, 4, 0.10, 0.5
# per detent, select, overshoot, looking
KNOB, PUSH, OVER, LOOK_KNOB = 0.18, 0.3, 0.05, 0.7


def fitts(d, w):
    return FA + FB * np.log2(d / w + 1)


def p_hit(w, s=np.hypot(S_MOTOR, S_VIB)):
    return 2 * stats.norm.cdf(w / 2 / s) - 1


decide, move, q = HA + HB * np.log2(13), fitts(D_MM, W_MM), p_hit(W_MM)
closed = {"touch": decide + move + (1 / q - 1) * (REAIM + move),
          "rotary knob": 6.5 * KNOB + PUSH + OVER * 0.5,
          "voice": (T_VOICE + E_VOICE * R_VOICE) / (1 - E_VOICE),
          "gesture": N_GEST * GEST / (1 - E_GEST),
          "voice + touch": T_VOICE + E_VOICE * (0.3 + HA + HB * np.log2(4) + move)}
eor = {"touch": closed["touch"], "rotary knob": LOOK_KNOB * closed["rotary knob"],
       "voice": CONFIRM, "gesture": LOOK_GEST * closed["gesture"],
       "voice + touch": CONFIRM + E_VOICE * (HA + HB * np.log2(4) + move)}


def simulate(m, n=200_000):
    jit = lambda mean: mean * np.exp(rng.normal(-0.02, 0.2, n))   # CV about 0.2, mean kept
    if m == "touch":
        miss = rng.geometric(q, n) - 1                            # misses before a hit
        t = jit(decide) + jit(move) + miss * (REAIM + move)
        return t, t
    if m == "rotary knob":
        t = rng.integers(1, 13, n) * KNOB + PUSH + (rng.random(n) < OVER) * 0.5
        return t, LOOK_KNOB * t
    if m == "voice":
        fails = rng.geometric(1 - E_VOICE, n) - 1
        return jit(T_VOICE) * (fails + 1) + fails * R_VOICE, np.full(n, CONFIRM)
    if m == "gesture":
        tries = N_GEST + rng.negative_binomial(N_GEST, 1 - E_GEST, n)
        t = tries * jit(GEST)
        return t, LOOK_GEST * t
    err = rng.random(n) < E_VOICE
    extra = HA + HB * np.log2(4) + move
    return jit(T_VOICE) + err * (0.3 + extra), CONFIRM + err * extra


rows = []
for m in closed:
    t, e = simulate(m)
    rows.append((m, closed[m], t.mean(), eor[m], e.mean(), (e > 2).mean()))
cmp = pd.DataFrame(rows, columns=["modality", "time_closed", "time_sim", "eyes_off_closed",
                                  "eyes_off_sim", "P_total_eyes_off_gt_2s"])
print(f"Touch: hit probability {q:.4f} for an {W_MM:.0f}-mm target on a rough road")
print(cmp.round(4).to_string(index=False))

# ---------- Checks on the motor model: Fitts' law and touch accuracy ----------
dd = rng.choice([60, 120, 180, 240], 4000)
ww = rng.choice([8, 12, 18, 24], 4000)
mt = fitts(dd, ww) + rng.normal(0, 0.05, 4000)
b_hat, a_hat = np.polyfit(np.log2(dd / ww + 1), mt, 1)
print(f"Fitts coefficients recovered by regression: a = {a_hat:.4f}, b = {b_hat:.4f} "
      f"(true {FA}, {FB})")
s_tot = np.hypot(S_MOTOR, S_VIB)
for w in [12.0, 18.0, 24.0]:
    x = rng.normal(0, s_tot, 400_000)
    print(f"  {w:.0f}-mm target: P(hit) {p_hit(w):.4f} closed form, "
          f"{(np.abs(x) < w / 2).mean():.4f} simulated")
w99 = 2 * stats.norm.ppf(0.995) * s_tot
print(f"Width for 99% hits on a rough road: {w99:.2f} mm (closed form), "
      f"{brentq(lambda w: p_hit(w) - 0.99, 1, 100):.2f} mm (root finding)")
print(f"Width for 99% hits on a smooth road: {2 * stats.norm.ppf(0.995) * S_MOTOR:.2f} mm")

# ---------- Redundant alerts: an independent race, and Miller's bound ----------
MED = {"visual": 0.62, "auditory": 0.52, "tactile": 0.48}
SIG = 0.22
F = {k: (lambda t, m=m: stats.lognorm.cdf(t, SIG, scale=m)) for k, m in MED.items()}


def race_cdf(t, keys):
    return 1 - np.prod([1 - F[k](t) for k in keys], axis=0)


rt = {k: m * np.exp(SIG * rng.normal(size=400_000)) for k, m in MED.items()}
for keys in [("visual",), ("auditory",), ("tactile",), ("visual", "auditory"),
             ("visual", "auditory", "tactile")]:
    med_c = brentq(lambda t: race_cdf(t, keys) - 0.5, 0.05, 3)
    med_s = np.median(np.min([rt[k] for k in keys], axis=0))
    lab = " + ".join(keys)
    print(f"  median RT, {lab:<27} {med_c:.4f} s closed form, {med_s:.4f} s sim")
coact = 0.90 * np.minimum(rt["visual"], rt["auditory"])     # faster than any race allows
t10 = np.quantile(coact, 0.10)
bound = F["visual"](t10) + F["auditory"](t10)
print(f"Coactivation check at t = {t10:.4f} s: P(RT <= t) = 0.1000\n  against Miller's "
      f"bound {bound:.4f}: {'violated' if 0.10 > bound else 'not violated'}")

# ---------- When does voice lose its advantage? ----------
# glance to see what went wrong, per voice error
LOOK_ERR = 0.8


def voice_eyes(e):                          # expected eyes-off-road with error glances
    return CONFIRM + LOOK_ERR * e / (1 - e)


target = eor["touch"]
e_star = (target - CONFIRM) / (target - CONFIRM + LOOK_ERR)
e_root = brentq(lambda e: voice_eyes(e) - target, 0, 0.99)
print(f"Touch eyes-off-road {target:.4f} s; voice matches it at a recognition error rate")
print(f"of {e_star:.4f} (closed form), {e_root:.4f} (root finding)")
e_sim = []
for e in [0.05, 0.12, 0.30, 0.50]:
    fails = rng.geometric(1 - e, 200_000) - 1
    e_sim.append((e, voice_eyes(e), (CONFIRM + LOOK_ERR * fails).mean()))
print(pd.DataFrame(e_sim, columns=["error_rate", "eyes_off_closed", "eyes_off_sim"])
      .round(4).to_string(index=False))

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
a = ax[0, 0]                                                   # (a) time and eyes-off-road
x = np.arange(len(cmp))
a.bar(x - 0.2, cmp.time_sim, 0.38, color=SKY, edgecolor=INK, lw=0.5, label="task time")
a.bar(x + 0.2, cmp.eyes_off_sim, 0.38, color=RUST, edgecolor=INK, lw=0.5,
      label="eyes off the road")
a.plot(x - 0.2, cmp.time_closed, "D", color=INK, ms=2.6)
a.plot(x + 0.2, cmp.eyes_off_closed, "D", color=INK, ms=2.6, label="closed form")
a.set_xticks(x, [m.replace(" + ", "\n+ ") for m in cmp.modality], fontsize=6.5)
a.set(ylabel="Seconds per selection", title="(a) Fast is not the same as eyes on the road", ylim=(0, 5.4))
a.legend(fontsize=6.5, loc="upper left", ncol=2)
a = ax[0, 1]                                                   # (b) touch accuracy
ww_ = np.linspace(5, 35, 300)
ROADS = [(S_MOTOR, SAGE, "smooth road"), (np.hypot(S_MOTOR, S_VIB), RUST, "rough road")]
for s, c, lab in ROADS:
    a.plot(ww_, p_hit(ww_, s), color=c, lw=1.4, label=lab)
for w in [12.0, 18.0, 24.0]:
    xx = rng.normal(0, np.hypot(S_MOTOR, S_VIB), 100_000)
    a.plot(w, (np.abs(xx) < w / 2).mean(), "o", color=INK, ms=3.2)
a.axhline(0.99, color=INK, ls=":", lw=0.6)
a.axvline(w99, color=RUST, ls="--", lw=0.6)
a.text(w99 + 0.5, 0.6, f"{w99:.1f} mm for 99%\non a rough road", fontsize=6.5, color=RUST)
a.set(xlabel="Touch target width (mm)", ylabel="Probability the first touch hits",
      title="(b) Vibration widens the target a touch needs")
a.legend(fontsize=6.5, loc="lower right")
a = ax[1, 0]                                                   # (c) redundant alerts
tt = np.linspace(0.2, 1.2, 400)
a.plot(tt, F["visual"](tt), color=SKY, lw=1.3, label="visual alone")
a.plot(tt, F["auditory"](tt), color=GOLD, lw=1.3, label="auditory alone")
a.plot(tt, race_cdf(tt, ("visual", "auditory")), color=SAGE, lw=1.4,
       label="visual + auditory (race)")
a.plot(tt, np.minimum(1, F["visual"](tt) + F["auditory"](tt)), color=INK, ls="--", lw=0.8,
       label="Miller's bound")
a.plot(np.sort(coact)[::4000], np.linspace(0, 1, len(coact))[::4000], color=RUST, lw=1.2,
       label="coactivation (simulated)")
a.set(xlabel="Reaction time (s)", ylabel="Cumulative probability", xlim=(0.2, 1.0),
      title="(c) Two channels, a faster response")
a.legend(fontsize=6, loc="lower right")
a = ax[1, 1]                                                   # (d) voice break-even
ee = np.linspace(0, 0.7, 300)
a.plot(ee, voice_eyes(ee), color=SAGE, lw=1.4, label="voice, with a glance per error")
a.plot(e_sim and [r[0] for r in e_sim], [r[2] for r in e_sim], "o", color=INK, ms=3.2,
       label="simulated")
a.axhline(target, color=RUST, ls="--", lw=0.9, label="touch")
a.axvline(e_star, color=INK, ls=":", lw=0.6)
a.text(e_star + 0.01, 0.75, f"break-even\n{e_star:.2f}", fontsize=6.5)
a.set(xlabel="Voice recognition error rate", ylabel="Eyes off the road per selection (s)",
      ylim=(0.4, 2.6), title="(d) When touch beats voice")
a.legend(fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig("chap04_fig02_modalities.png", dpi=300)
