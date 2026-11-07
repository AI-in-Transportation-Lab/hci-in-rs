"""Chapter 2 - Case Study 2: Trust Calibration in Automated Driving Decision Support.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: code/ch02_case2_trust.py (shown in chapters/02a.tex).
Run from the repository root with `make ch02` or `python scripts/run_all.py --chapter 2`;
outputs are written to outputs/ch02/.

Case Study 2 -- Trust calibration in automated driving decision support.

A pedestrian-conflict alert operates on an urban arterial in two contexts that
differ in how well the sensor can actually see: an open crosswalk, and a
crosswalk partly occluded by a stopped bus.  The interface is the experimental
variable.  The status-quo version shows one symbol in both contexts; the
revised version also shows that the view is obstructed.

The same 4,000 encounters are replayed under both interfaces, so the two
conditions face an identical set of conflicts.

The revised interface is not intended to raise trust.  It is intended to move
reliance toward what the system can actually deliver in each context -- which,
where the bus blocks the view, means trusting its silence less.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(7742)
N = 4000

CONTEXT_P = {"open crosswalk": 0.70, "bus occlusion": 0.30}
CONFLICT_P = {"open crosswalk": 0.06, "bus occlusion": 0.11}
HIT = {"open crosswalk": 0.94, "bus occlusion": 0.55}   # P(alert | conflict)
FA = {"open crosswalk": 0.05, "bus occlusion": 0.09}    # P(alert | no conflict)

# P(protective action) when alerted, and when the system stays silent.
RELY_ALERT = {"status quo": {"open crosswalk": 0.88, "bus occlusion": 0.86},
              "with limits": {"open crosswalk": 0.89, "bus occlusion": 0.84}}
RELY_SILENT = {"status quo": {"open crosswalk": 0.22, "bus occlusion": 0.24},
               "with limits": {"open crosswalk": 0.21, "bus occlusion": 0.57}}
CONDITIONS, CONTEXTS = list(RELY_ALERT), list(CONTEXT_P)


def posterior(ctx, alerted):
    """P(conflict | alert) and P(conflict | silence), from the design values."""
    p, h, f = CONFLICT_P[ctx], HIT[ctx], FA[ctx]
    if alerted:
        return h * p / (h * p + f * (1 - p))
    return (1 - h) * p / ((1 - h) * p + (1 - f) * (1 - p))


# ---- one set of encounters, replayed under both interfaces ----------------
ctx = RNG.choice(CONTEXTS, N, p=[CONTEXT_P[c] for c in CONTEXTS])
conflict = RNG.random(N) < np.array([CONFLICT_P[c] for c in ctx])
alert = RNG.random(N) < np.where(conflict, [HIT[c] for c in ctx],
                                 [FA[c] for c in ctx])

# Common random numbers: the same driver draw is used under both interfaces,
# so any difference below comes from the interface, not from sampling noise.
u_act = RNG.random(N)

frames = []
for cond in CONDITIONS:
    p_act = np.where(alert, [RELY_ALERT[cond][c] for c in ctx],
                     [RELY_SILENT[cond][c] for c in ctx])
    frames.append(pd.DataFrame(dict(condition=cond, context=ctx,
                                    conflict=conflict, alert=alert,
                                    acted=u_act < p_act)))
df = pd.concat(frames, ignore_index=True)

# ------------------------------------------- 1. what a signal is worth
print("1. WHAT EACH SIGNAL IS WORTH, BY CONTEXT  (Bayes, from design values)")
print(f"   {'context':<17}{'P(conflict)':>12}{'hit':>7}{'FA':>7}"
      f"{'P(conf|alert)':>15}{'P(conf|silence)':>17}")
for c in CONTEXTS:
    print(f"   {c:<17}{CONFLICT_P[c]:>12.2f}{HIT[c]:>7.2f}{FA[c]:>7.2f}"
          f"{posterior(c, True):>15.3f}{posterior(c, False):>17.3f}")
ratio = posterior("bus occlusion", False) / posterior("open crosswalk", False)
print(f"\n   Silence hides a conflict {ratio:.0f} times more often behind the")
print("   bus than at the open crosswalk. The alert symbol is identical in")
print("   both, so the status-quo interface conveys none of this.")

# ------------------------------------------------------------ 2. reliance
print("\n2. WHAT DRIVERS DID")
print(f"   {'condition':<13}{'context':<17}{'acts|alert':>12}{'acts|silence':>14}")
for cond in CONDITIONS:
    for c in CONTEXTS:
        s = df[(df.condition == cond) & (df.context == c)]
        print(f"   {cond:<13}{c:<17}{s[s.alert]['acted'].mean():>12.2f}"
              f"{s[~s.alert]['acted'].mean():>14.2f}")
print("   Compliance with alerts is unchanged. Only the response to silence")
print("   in the occluded context moved, which is the only place it should.")

# ------------------------------------------------- 3. the outcome that counts
print("\n3. CONFLICTS THAT WENT UNPROTECTED  (identical encounters, n = 4000)")
print(f"   {'condition':<13}{'context':<17}{'conflicts':>10}"
      f"{'unprotected':>13}{'rate':>8}")
for cond in CONDITIONS:
    for c in CONTEXTS:
        s = df[(df.condition == cond) & (df.context == c) & df.conflict]
        miss = len(s) - int(s["acted"].sum())
        print(f"   {cond:<13}{c:<17}{len(s):>10d}{miss:>13d}"
              f"{100 * miss / len(s):>7.1f}%")
tot = {}
for cond in CONDITIONS:
    s = df[(df.condition == cond) & df.conflict]
    tot[cond] = len(s) - int(s["acted"].sum())
    print(f"   {cond:<13}{'ALL':<17}{len(s):>10d}{tot[cond]:>13d}"
          f"{100 * tot[cond] / len(s):>7.1f}%")
print(f"\n   Reduction in unprotected conflicts: "
      f"{tot['status quo'] - tot['with limits']} fewer "
      f"({100 * (1 - tot['with limits'] / tot['status quo']):.0f}%).")

# ---------------------------------------- 4. the measure that would mislead
print("\n4. WHAT A TRUST QUESTIONNAIRE WOULD HAVE REPORTED")
for cond in CONDITIONS:
    s = df[df.condition == cond]
    print(f"   {cond:<13} compliance with alerts = "
          f"{s[s.alert]['acted'].mean():.2f}   "
          f"overall action rate = {s['acted'].mean():.2f}")
print("   On compliance the two interfaces are indistinguishable. An")
print("   evaluation that measured trust or compliance alone would have")
print("   reported no effect, and the safety gain in Section 3 would have")
print("   been invisible.")

# ------------------------------------------------------------- 5. the figure
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
col = dict(zip(CONDITIONS, [GOLD, SAGE]))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

# (a) action rate when the system is silent, against the risk silence hides
w, xs = 0.38, np.arange(len(CONTEXTS))
for k, cond in enumerate(CONDITIONS):
    v = [df[(df.condition == cond) & (df.context == c) & ~df.alert]["acted"].mean()
         for c in CONTEXTS]
    a1.bar(xs + (k - 0.5) * w, v, w * 0.94, color=col[cond], edgecolor=INK, lw=0.5,
           label=cond)
a1b = a1.twinx()
a1b.plot(xs, [posterior(c, False) for c in CONTEXTS], "o-", color=RUST, mec=INK,
         lw=1.6, ms=5.4, label="P(conflict | silence), right")
a1b.set_ylabel("P(conflict | silence)", color=RUST)
a1b.tick_params(axis="y", colors=RUST)
a1b.spines["right"].set(visible=True, color=RUST)
a1b.set_ylim(0, 0.075); a1b.grid(False)
a1.set_xticks(xs, CONTEXTS)
a1.set(ylabel="P(protective action | silence)", ylim=(0, 0.75),
       title="(a) Reliance on silence vs. what it hides")
h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a1b.get_legend_handles_labels()
a1.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper left")

# (b) unprotected conflicts
for k, cond in enumerate(CONDITIONS):
    v = [100 * (1 - df[(df.condition == cond) & (df.context == c) &
                       df.conflict]["acted"].mean()) for c in CONTEXTS]
    b = a2.bar(xs + (k - 0.5) * w, v, w * 0.94, color=col[cond], edgecolor=INK,
               lw=0.5, label=cond)
    a2.bar_label(b, fmt="%.1f%%", fontsize=6.5, padding=1.6)
a2.set_xticks(xs, CONTEXTS)
a2.set(ylabel="Conflicts left unprotected (%)", ylim=(0, 56),
       title="(b) What the questionnaire could not see")
a2.legend(fontsize=6.5, loc="upper left")

fig.tight_layout()
fig.savefig("fig02_07_cs2_trust.png", dpi=300)
print("\n   figure written to fig02_07_cs2_trust.png")
