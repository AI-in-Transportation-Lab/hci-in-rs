"""Chapter 3 - Case Study 1: Smartphone and Infotainment Distraction in Urban Driving.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: code/ch03_case1_interfaces.py (shown in chapters/03a.tex).
Run from the repository root with `make ch03` or `python scripts/run_all.py --chapter 3`;
outputs are written to outputs/ch03/.

Case Study 1 -- Four ways to enter a destination, in urban driving.

The same goal is implemented four ways.  Thirty-six drivers complete each
implementation on the same simulated urban corridor, so every comparison is
within-subject.  Data are synthetic, generated from the assumptions in
Table 3.4, so each number can be checked against a known process.

The question is not which interface is best.  It is that the four standard
ways of asking produce four different answers, and only one of them is about
safety.
"""
import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(31415)
N = 36
SPEED = 13.9          # m/s, 50 km/h urban corridor

# name, task duration (s), eyes-off share, mean glance (s), DRT cost (s)
DESIGNS = {
    "touch, free text":   dict(dur=17.0, eyes=0.55, glance=1.55, drt=0.16),
    "touch, predictive":  dict(dur=10.5, eyes=0.48, glance=1.20, drt=0.14),
    "voice, confirmed":   dict(dur=24.0, eyes=0.07, glance=0.85, drt=0.26),
    "voice, single shot": dict(dur=12.0, eyes=0.05, glance=0.80, drt=0.21),
}
NAMES = list(DESIGNS)

# Per-driver traits, shared across designs: the same person is slower or
# more cautious in every condition.
trait_speed = RNG.normal(0, 0.14, N)      # multiplies task duration
trait_glance = RNG.normal(0, 0.17, N)     # adds to mean glance (s)
trait_drt = RNG.normal(0, 0.045, N)       # adds to DRT cost (s)


def simulate():
    rows = []
    for d in range(N):
        for name in NAMES:
            s = DESIGNS[name]
            dur = max(3.0, s["dur"] * (1 + trait_speed[d]))
            teort = dur * s["eyes"] * float(RNG.normal(1.0, 0.10))
            gmean = max(0.25, s["glance"] + trait_glance[d])
            n_gl = max(1, int(round(teort / gmean)))
            glances = RNG.gamma(8.0, gmean / 8.0, n_gl)
            longest = float(glances.max()) if n_gl else 0.0
            drt = max(0.0, s["drt"] + trait_drt[d] + RNG.normal(0, 0.03))
            rows.append(dict(driver=d, design=name, duration=dur,
                             teort=teort, longest=longest, n_glances=n_gl,
                             mean_glance=float(glances.mean()),
                             drt_cost=drt))
    return pd.DataFrame(rows)


df = simulate()

# ------------------------------------------- 1. the visual-manual verdict
print("1. THE VISUAL-MANUAL VERDICT  (NHTSA-style acceptance test)")
print(f"   {'design':<20}{'mean TEORT':>12}{'85th longest':>14}"
      f"{'verdict':>10}")
vm = {}
for name in NAMES:
    s = df[df.design == name]
    p85 = float(np.percentile(s["longest"], 85))
    ok = "pass" if (p85 <= 2.0 and s["teort"].mean() <= 12.0) else "FAIL"
    vm[name] = (s["teort"].mean(), p85, ok)
    print(f"   {name:<20}{s['teort'].mean():>11.1f}s{p85:>13.2f}s{ok:>10}")
print("   Both voice designs pass trivially: they barely use the eyes.")

# ------------------------------------------------ 2. the cognitive verdict
print("\n2. THE COGNITIVE VERDICT  (detection-response task)")
print(f"   {'design':<20}{'DRT cost':>10}{'task length':>13}"
      f"{'rank':>7}")
order = sorted(NAMES, key=lambda n: df[df.design == n]["drt_cost"].mean())
for name in NAMES:
    s = df[df.design == name]
    print(f"   {name:<20}{1000 * s['drt_cost'].mean():>8.0f}ms"
          f"{s['duration'].mean():>12.1f}s{order.index(name) + 1:>7}")
print("   The ranking has reversed. Voice costs the most per response.")

# ----------------------------------------- 3. the measure that combines them
print("\n3. EXPOSURE-WEIGHTED DELAY  (Eq. 3.5; delay-seconds)")
print(f"   {'design':<20}{'visual':>9}{'cognitive':>11}{'total':>9}"
      f"{'per hazard':>12}{'rank':>6}")
tot, per = {}, {}
for name in NAMES:
    s = df[df.design == name]
    visual = float((s["teort"] * s["mean_glance"] / 2.0).mean())
    cognitive = float((s["duration"] * s["drt_cost"]).mean())
    tot[name] = visual + cognitive
    per[name] = tot[name] / float(s["duration"].mean())
rank = sorted(NAMES, key=lambda n: tot[n])
for name in NAMES:
    s = df[df.design == name]
    visual = float((s["teort"] * s["mean_glance"] / 2.0).mean())
    cognitive = float((s["duration"] * s["drt_cost"]).mean())
    print(f"   {name:<20}{visual:>8.2f}s{cognitive:>10.2f}s"
          f"{tot[name]:>8.2f}s{1000 * per[name]:>10.0f}ms"
          f"{rank.index(name) + 1:>6}")
print("   A delay-second is an index, not a distance: a delay multiplied")
print("   by the time over which it applies. The last column divides it")
print(f"   back out, so at {SPEED:.1f} m/s the worst design costs "
      f"{SPEED * max(per.values()):.1f} m")
print(f"   of stopping margin and the best costs "
      f"{SPEED * min(per.values()):.1f} m.")

best, worst = rank[0], rank[-1]
a = df[df.design == best].set_index("driver")
b = df[df.design == worst].set_index("driver")
da = a["teort"] * a["mean_glance"] / 2 + a["duration"] * a["drt_cost"]
dbb = b["teort"] * b["mean_glance"] / 2 + b["duration"] * b["drt_cost"]
t, p = stats.ttest_rel(dbb, da)
diff = dbb - da
ci = stats.t.interval(0.95, N - 1, loc=diff.mean(), scale=stats.sem(diff))
print(f"\n   '{worst}' minus '{best}': {diff.mean():.2f} delay-seconds")
print(f"   95% CI = [{ci[0]:.2f}, {ci[1]:.2f}],  t({N - 1}) = {t:.2f},"
      f"  p = {p:.1e}")

# --------------------------------------------- 4. the four rankings compared
print("\n4. FOUR QUESTIONS, FOUR ANSWERS")
pref = {"touch, free text": 2.9, "touch, predictive": 3.6,
        "voice, confirmed": 4.4, "voice, single shot": 4.1}
rk_vm = sorted(NAMES, key=lambda n: vm[n][1])
rk_drt = order
rk_tot = rank
rk_pref = sorted(NAMES, key=lambda n: -pref[n])
print(f"   {'design':<20}{'glance':>8}{'DRT':>6}{'delay-s':>9}"
      f"{'liking':>8}")
for name in NAMES:
    print(f"   {name:<20}{rk_vm.index(name) + 1:>8}{rk_drt.index(name) + 1:>6}"
          f"{rk_tot.index(name) + 1:>9}{rk_pref.index(name) + 1:>8}")
print("   Every column is a defensible reading of the same experiment.")
print("   'voice, confirmed' is the design drivers liked best and the second")
print("   cleanest on glance data, yet it finishes behind a touchscreen on")
print("   delay-seconds, because it occupies the driver twice as long.")

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
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.2, 2.7))

x = np.arange(len(NAMES))                                      # (a) two pathways
vis = [float((df[df.design == n]["teort"] *
              df[df.design == n]["mean_glance"] / 2).mean()) for n in NAMES]
cog = [float((df[df.design == n]["duration"] *
              df[df.design == n]["drt_cost"]).mean()) for n in NAMES]
a1.bar(x, vis, 0.56, color=SAGE, edgecolor=INK, lw=0.5, label="visual component")
a1.bar(x, cog, 0.56, bottom=vis, color=GOLD, edgecolor=INK, lw=0.5,
       label="cognitive component")
for xi, (v, c) in enumerate(zip(vis, cog)):
    a1.text(xi, v + c + 0.12, f"{v + c:.2f} s", ha="center", fontsize=6.5)
a1.set_xticks(x, [n.replace(", ", ",\n") for n in NAMES], fontsize=7)
a1.set(ylabel="Exposure-weighted delay (delay-seconds)",
       ylim=(0, max(np.array(vis) + np.array(cog)) * 1.22),
       title="(a) The two pathways, on one scale")
a1.legend(fontsize=6.5, loc="upper right")

rks = {"glance": rk_vm, "DRT": rk_drt, "delay-s": rk_tot, "liking": rk_pref}
for n, c in zip(NAMES, [RUST, GOLD, SAGE, SKY]):               # (b) rank by measure
    a2.plot(range(len(rks)), [rks[k].index(n) + 1 for k in rks], "o-", lw=1.7,
            ms=6.2, color=c, mec=INK, label=n)
a2.set_xticks(range(len(rks)), list(rks))
a2.set_yticks([1, 2, 3, 4], ["best", "2nd", "3rd", "worst"])
a2.set_ylim(5.5, 0.6)
a2.set(xlabel="Measure used to rank the designs",
       title="(b) The question asked decides the winner")
a2.legend(fontsize=6.5, loc="lower center", ncol=2)

fig.tight_layout()
fig.savefig("fig03_06_cs1_interfaces.png", dpi=300)
print("\n   figure written to fig03_06_cs1_interfaces.png")
