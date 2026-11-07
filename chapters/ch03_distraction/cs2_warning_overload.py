"""Chapter 3 - Case Study 2: Warning Overload and Attention Capture in Safety-Critical Alerts.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: code/ch03_case2_alerts.py (shown in chapters/03a.tex).
Run from the repository root with `make ch03` or `python scripts/run_all.py --chapter 3`;
outputs are written to outputs/ch03/.

Case Study 2 -- Warning overload and attention capture.

A fleet of 60 vehicles is driven for 80 hours each.  Five signal sources
share one cabin.  Three are warnings and, as built, they sound alike, so a
driver who cannot tell them apart experiences the pooled reliability of the
whole warning channel rather than the reliability of any one source.

Two interventions are evaluated against the same simulated driving:
raising the detection thresholds of the two low-value warnings, and adding
an arbitration rule that suppresses a lower-priority signal within 2.5 s of
a higher-priority one.

Common random numbers are used throughout.  Raising a threshold is modelled
as thinning: each genuine alert survives with one probability and each false
alarm with a lower one, so the cost of the change -- genuine detections lost
-- is visible rather than assumed away.

The outcome is not how many alerts were issued. It is how many genuine
forward-collision conflicts ended with the driver braking in time.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(90210)
DRIVERS, HOURS = 60, 80.0
FLEET_H = DRIVERS * HOURS
FLEET_S = FLEET_H * 3600

# source: signals/hour, PPV, priority (1 is highest)
AS_BUILT = {"forward collision": (0.40, 0.31, 1),
            "blind spot":        (2.20, 0.22, 2),
            "lane departure":    (1.80, 0.17, 3),
            "navigation":        (6.00, 0.95, 4),
            "phone/messaging":   (3.00, 0.90, 5)}
RETUNED = dict(AS_BUILT,
               **{"blind spot": (0.90, 0.48, 2),
                  "lane departure": (0.50, 0.45, 3)})

WARNINGS = ["forward collision", "blind spot", "lane departure"]
FLOOR = 0.20          # probability of acting on a warning of no value
ARB_WINDOW = 2.5      # s
STARTLE_COST = 0.35   # s added when a warning lands on another signal
TTC_BUDGET = 1.90     # s available to brake once the warning sounds
BASE_RT = 1.25        # s mean braking response to an attended warning


def split(rate, ppv):
    """Genuine and false signal rates per hour."""
    return rate * ppv, rate * (1 - ppv)


def thinning(name):
    """P(keep) for genuine and for false signals when the threshold rises."""
    g0, f0 = split(*AS_BUILT[name][:2])
    g1, f1 = split(*RETUNED[name][:2])
    return g1 / g0, f1 / f0


# ---- one stream of events, generated once and reused by every configuration
parts = []
for name, (rate, ppv, pri) in AS_BUILT.items():
    for genuine, r in zip((True, False), split(rate, ppv)):
        n = RNG.poisson(r * FLEET_H)
        parts.append(pd.DataFrame(dict(
            t=RNG.uniform(0, FLEET_S, n), source=name, priority=pri,
            genuine=genuine, u_keep=RNG.random(n))))
ev = pd.concat(parts, ignore_index=True).sort_values("t").reset_index(drop=True)

# the same driver draws are reused under every configuration
ev["u_act"] = RNG.random(len(ev))
ev["rt"] = RNG.gamma(9.0, BASE_RT / 9.0, len(ev))

print("0. WHAT RAISING A THRESHOLD COSTS")
print(f"   {'warning':<17}{'genuine kept':>14}{'false kept':>12}")
for n in ("blind spot", "lane departure"):
    kg, kf = thinning(n)
    print(f"   {n:<17}{100 * kg:>13.1f}%{100 * kf:>11.1f}%")
print("   Removing three false blind-spot alerts in four costs one genuine")
print("   detection in ten; removing four false lane-departure alerts in")
print("   five costs one in four. That trade is the intervention.")


def configure(retune, arbitrate):
    d = ev.copy()
    keep = np.ones(len(d), dtype=bool)
    if retune:
        for n in ("blind spot", "lane departure"):
            kg, kf = thinning(n)
            m = d.source == n
            keep &= ~m | (d.u_keep < np.where(d.genuine, kg, kf))
    d = d[keep].reset_index(drop=True)

    if arbitrate:
        shown = np.ones(len(d), dtype=bool)
        t, pri = d["t"].to_numpy(), d["priority"].to_numpy()
        lt, lp = -1e9, 99
        for i in range(len(d)):
            if t[i] - lt < ARB_WINDOW and pri[i] > lp:
                shown[i] = False
            else:
                lt, lp = t[i], pri[i]
        d = d[shown].reset_index(drop=True)
    return d


def evaluate(d, label):
    cfg = RETUNED if "retuned" in label else AS_BUILT
    wr = sum(cfg[n][0] for n in WARNINGS)
    v = sum(cfg[n][0] * cfg[n][1] for n in WARNINGS) / wr
    comply = FLOOR + (1 - FLOOR) * v

    t = d["t"].to_numpy()
    is_conf = ((d.source == "forward collision") & d.genuine).to_numpy()
    prev_gap = np.full(len(d), 1e9)
    prev_gap[1:] = t[1:] - t[:-1]
    startled = is_conf & (prev_gap < ARB_WINDOW)

    c = d[is_conf]
    rt = c["rt"].to_numpy() + startled[is_conf] * STARTLE_COST
    in_time = (c["u_act"].to_numpy() < comply) & (rt < TTC_BUDGET)
    return dict(label=label, signals_h=len(d) / FLEET_H,
                warn_h=int(d.source.isin(WARNINGS).sum()) / FLEET_H,
                ppv=v, comply=comply, conflicts=len(c),
                startled=int(startled.sum()), handled=int(in_time.sum()))


res = [evaluate(configure(False, False), "as built"),
       evaluate(configure(True, False), "retuned thresholds"),
       evaluate(configure(True, True), "retuned + arbitration")]

print("\n1. WHAT REACHES THE DRIVER  (60 drivers x 80 h = 4,800 vehicle-hours)")
print(f"   {'configuration':<24}{'signals/h':>11}{'warnings/h':>12}"
      f"{'pooled PPV':>12}{'compliance':>12}")
for r in res:
    print(f"   {r['label']:<24}{r['signals_h']:>11.2f}{r['warn_h']:>12.2f}"
          f"{r['ppv']:>12.2f}{r['comply']:>12.2f}")

print("\n2. GENUINE FORWARD-COLLISION CONFLICTS  (the same events each time)")
print(f"   {'configuration':<24}{'conflicts':>10}{'braked in time':>16}"
      f"{'rate':>8}")
for r in res:
    print(f"   {r['label']:<24}{r['conflicts']:>10d}{r['handled']:>16d}"
          f"{100 * r['handled'] / r['conflicts']:>7.1f}%")
b = res[0]
for r in res[1:]:
    print(f"   {r['label']:<24} "
          f"{100 * (r['handled'] - b['handled']) / b['conflicts']:+.1f} "
          f"percentage points, {r['handled'] - b['handled']:+d} conflicts")

print("\n3. HOW OFTEN DOES A WARNING ACTUALLY LAND ON ANOTHER SIGNAL?")
for r in res:
    print(f"   {r['label']:<24}{r['startled']:>4d} of {r['conflicts']} "
          f"({100 * r['startled'] / r['conflicts']:.1f}%)")
lam = res[0]["signals_h"] / 3600.0
print(f"   At {res[0]['signals_h']:.1f} signals an hour the chance that any "
      f"one of them")
print(f"   falls within {ARB_WINDOW:.1f} s of another is "
      f"{100 * (1 - np.exp(-lam * ARB_WINDOW)):.1f}%. Simultaneity is rare.")
need = -np.log(0.95) / ARB_WINDOW * 3600
print(f"   It would take {need:.0f} signals an hour to reach even 5%.")

print("\n4. THE MEASURE THAT WOULD MISLEAD")
print(f"   {'configuration':<24}{'alerts/h':>10}{'conflicts handled':>20}")
for r in res:
    print(f"   {r['label']:<24}{r['signals_h']:>10.2f}{r['handled']:>20d}")
print("   Arbitration changes nothing at all for the critical channel, and")
print("   the reason is structural: it suppresses a lower-priority signal")
print("   that arrives after a higher-priority one, so it can never protect")
print("   the highest-priority alert from a signal that preceded it. A")
print("   report that counted alerts issued would still have credited it.")

# ------------------------------------------------------------- the figure
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
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
lab = ["as built", "retuned\nthresholds", "retuned +\narbitration"]
x = np.arange(3)

bars = a1.bar(x, [r["warn_h"] for r in res], 0.46, color=GOLD, edgecolor=INK, lw=0.5,
              label="warnings per hour (left)")             # (a) fewer, better warnings
a1.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
a1b = a1.twinx()
l1, = a1b.plot(x, [r["ppv"] for r in res], "o-", color=RUST, mec=INK, lw=2, ms=8,
               label="pooled predictive value (right)")
l2, = a1b.plot(x, [r["comply"] for r in res], "s--", color=SAGE, mec=INK, lw=2,
               ms=7, label="compliance (right)")
for xi, r in enumerate(res):
    a1b.annotate(f"{r['ppv']:.2f}", (xi, r["ppv"]), xytext=(0, -15),
                 textcoords="offset points", ha="center", fontsize=8, color=RUST,
                 bbox=dict(fc=PAPER, ec="none", pad=0.6))
    a1b.annotate(f"{r['comply']:.2f}", (xi, r["comply"]), xytext=(0, 9),
                 textcoords="offset points", ha="center", fontsize=8, color=SAGE,
                 bbox=dict(fc=PAPER, ec="none", pad=0.6))
a1.set_xticks(x, lab, fontsize=9)
a1.set(ylabel="Warnings presented per hour", ylim=(0, 6.6),
       title="(a) Fewer warnings, each worth more")
a1b.set(ylabel="Predictive value / compliance", ylim=(0, 1.0))
a1b.spines["right"].set_visible(True); a1b.grid(False)
a1.legend(handles=[bars, l1, l2], fontsize=8, frameon=False, loc="upper right")

rate = [100 * r["handled"] / r["conflicts"] for r in res]      # (b) conflicts handled
bars = a2.bar(x, rate, 0.5, color=[RUST, GOLD, SAGE], edgecolor=INK, lw=0.5)
a2.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=3)
a2.set_xticks(x, lab, fontsize=9)
a2.set(ylabel="Genuine conflicts braked for in time (%)", ylim=(0, max(rate) * 1.25),
       title="(b) The outcome an alert count cannot see")

fig.tight_layout()
fig.savefig("fig03_07_cs2_alerts.png", dpi=200)
print("\n   figure written to fig03_07_cs2_alerts.png")
