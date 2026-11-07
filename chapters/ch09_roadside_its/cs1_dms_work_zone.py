"""Chapter 9 - Case Study 1: Dynamic Message Signs and Work Zone Messaging for Driver Comprehension.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/09a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch09` or `python scripts/run_all.py --chapter 9`;
outputs are written to outputs/ch09/.
"""
import numpy as np
import pandas as pd
from scipy import stats, integrate

# ---------- Drivers approaching a portable message sign ----------
rng = np.random.default_rng(9)
N = 400_000
OLDER = 0.20                                   # share of drivers aged 65 and over
GROUP = {"under 65": (40.0, 6.0), "65 and over": (32.0, 6.0)}  # legibility, ft per inch
V_MU, V_SD = 65.0, 6.0                         # approach speed (mph)
# letter (in), lateral offset (ft), cone (deg)
H, OFFSET, CONE = 18.0, 12.0, 10.0
FPS = 5280 / 3600                              # ft/s per mph
LOSS = OFFSET / np.tan(np.radians(CONE))       # sign leaves the cone of vision (ft)
older = rng.random(N) < OLDER
li = np.where(older, rng.normal(*GROUP["65 and over"], N),
              rng.normal(*GROUP["under 65"], N))
v = rng.normal(V_MU, V_SD, N)
t_avail = (li * H - LOSS) / (v * FPS)          # seconds of legible exposure
print(f"Loss distance {LOSS:.1f} ft; median available reading time: "
      f"under 65 {np.median(t_avail[~older]):.2f} s, "
      f"65 and over {np.median(t_avail[older]):.2f} s")

# ---------- Candidate messages: words per phase; 1 s per word, 2 s minimum ----------
MSG = {"A": ("STOPPED TRAFFIC AHEAD", [3]),
       "B": ("STOPPED TRAFFIC / 2 MILES", [4]),
       "C": ("STOPPED TRAFFIC 2 MILES | BE PREPARED TO STOP", [4, 4]),
       "D": ("RIGHT LANE CLOSED | STOPPED TRAFFIC 2 MILES", [3, 4])}


def p_fixed(t, mu, sd):
    # P(H*LI - LOSS - FPS*V*t >= 0): a linear combination of independent normals
    m = H * mu - LOSS - FPS * V_MU * t
    s = np.sqrt((H * sd) ** 2 + (FPS * V_SD * t) ** 2)
    return stats.norm.cdf(m / s)


def p_analytic(phases, mu, sd):
    T = [max(2.0, w) for w in phases]
    if len(T) == 1:
        return p_fixed(T[0], mu, sd)
    # arrival is uniform over the cycle; wait for the next phase, then read C
    C = sum(T)
    return sum(integrate.quad(lambda r: p_fixed(C + r, mu, sd), 0, Tk)[0] for Tk in T) / C


def p_monte_carlo(phases):
    T = np.array([max(2.0, w) for w in phases])
    if len(T) == 1:
        need = np.full(N, T[0])
    else:
        C, u = T.sum(), rng.uniform(0, T.sum(), N)
        need = C + np.where(u < T[0], T[0] - u, C - u)
    return t_avail >= need


rows = []
for k, (text, ph) in MSG.items():
    ok = p_monte_carlo(ph)
    for g, mask in [("under 65", ~older), ("65 and over", older)]:
        a, m = p_analytic(ph, *GROUP[g]), ok[mask].mean()
        se = np.sqrt(max(m * (1 - m), 1e-12) / mask.sum())
        rows.append((k, sum(ph), len(ph), g, a, m, (m - a) / se))
read = pd.DataFrame(rows, columns=["msg", "words", "phases", "group", "analytic",
                                   "monte_carlo", "z"])
print(read.round(4).to_string(index=False))
P_ALL = {}
for k in MSG:
    a_u = read[(read.msg == k) & (read.group == "under 65")].analytic.iloc[0]
    a_o = read[(read.msg == k) & (read.group == "65 and over")].analytic.iloc[0]
    P_ALL[k] = (1 - OLDER) * a_u + OLDER * a_o
print("\nAll drivers (analytic):", {k: round(float(p), 3) for k, p in P_ALL.items()},
      f"\nLargest |z| between methods: {read.z.abs().max():.2f}")

# ---------- A queue hidden beyond a crest: who can stop in time? ----------
S = 500.0                                      # sight distance to the queue tail (ft)
A_MU, A_SD = 15.0, 3.0                         # braking deceleration (ft/s^2)
PRT = {"unwarned": (1.5, 0.35), "warned": (1.0, 0.35)}   # median (s) and log-sd
# speed reduction after reading a warning (mph)
DV_MU, DV_SD = 8.0, 3.0
CREDIBLE = 0.85                                # share of readers who act on the warning


def stop_dist(v_mph, prt, a):
    u = v_mph * FPS
    return u * prt + u ** 2 / (2 * a)


def p_fail_quad(S, v_mu, v_sd, prt_med, prt_sd, n=80):
    # Gauss-Hermite quadrature over speed and log reaction time; braking in closed form
    x, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    V, T = v_mu + v_sd * x[:, None], prt_med * np.exp(prt_sd * x[None, :])
    u = V * FPS
    room = S - u * T                           # distance left after the reaction time
    a_star = u ** 2 / (2 * np.maximum(room, 1e-9))
    p = np.where(room > 0, stats.norm.cdf((a_star - A_MU) / A_SD), 1.0)
    return float(w @ p @ w)


n = 400_000
v0, a = rng.normal(V_MU, V_SD, n), rng.normal(A_MU, A_SD, n)
t_u = PRT["unwarned"][0] * np.exp(PRT["unwarned"][1] * rng.normal(size=n))
t_w = PRT["warned"][0] * np.exp(PRT["warned"][1] * rng.normal(size=n))
v_w = v0 - rng.normal(DV_MU, DV_SD, n)
VW_MU, VW_SD = V_MU - DV_MU, np.hypot(V_SD, DV_SD)        # warned speeds are still normal
res = []
for label, d, args in [("unwarned", stop_dist(v0, t_u, a), (V_MU, V_SD, *PRT["unwarned"])),
                       ("warned", stop_dist(v_w, t_w, a), (VW_MU, VW_SD, *PRT["warned"]))]:
    mc, q = (d > S).mean(), p_fail_quad(S, *args)
    res.append((label, q, mc, (mc - q) / np.sqrt(q * (1 - q) / n)))
res = pd.DataFrame(res, columns=["drivers", "quadrature", "monte_carlo", "z"])
print(f"Share who cannot stop within {S:.0f} ft of sight distance:")
print(res.round(4).to_string(index=False))
respond = CREDIBLE * P_ALL["B"]
mix = respond * res.quadrature[1] + (1 - respond) * res.quadrature[0]
print(f"\nMessage B read by {P_ALL['B']:.3f}; acted on by {respond:.3f} of drivers")
print(f"Cannot stop: {res.quadrature[0]:.4f} without the sign, {mix:.4f} with it "
      f"({1 - mix / res.quadrature[0]:.1%} fewer)")

# ---------- Cross-check against the AASHTO stopping sight distance ----------
ssd = 1.47 * 65 * 2.5 + 1.075 * 65 ** 2 / 11.2
qa, qb = 1.075 / 11.2, 1.47 * 2.5
v_ok = (-qb + np.sqrt(qb ** 2 + 4 * qa * S)) / (2 * qa)
print(f"\nAASHTO stopping sight distance at 65 mph: {ssd:.1f} ft (tabulated 645 ft); "
      f"\nthe design driver can stop within {S:.0f} ft only from {v_ok:.1f} mph or less")

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
x = np.arange(len(MSG))                                              # (a) who can read
for j, (g, c) in enumerate([("under 65", SAGE), ("65 and over", GOLD)]):
    r = read[read.group == g]
    ax[0].bar(x + (j - 0.5) * 0.38, r.monte_carlo, 0.36, color=c, edgecolor=INK, lw=0.5,
              label=g)
    ax[0].plot(x + (j - 0.5) * 0.38, r.analytic, "D", color=INK, ms=2.5)
ax[0].plot([], [], "D", color=INK, ms=2.5, label="closed form")
ax[0].set_xticks(x, [f"{k}: {sum(MSG[k][1])} words, {len(MSG[k][1])} ph." for k in MSG],
                 fontsize=6, rotation=30, ha="right", rotation_mode="anchor")
ax[0].set(ylim=(0, 1.08), ylabel="Share who can read the whole message",
          title="(a) Who can read each message")
ax[0].legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 0.8),
             handlelength=1.2)
# (b) stopping distance
bins = np.linspace(0, 1100, 70)
ax[1].hist(stop_dist(v0, t_u, a), bins, density=True, histtype="step", lw=1.2, color=RUST,
           label="no warning")
ax[1].hist(stop_dist(v_w, t_w, a), bins, density=True, histtype="step", lw=1.2, color=SAGE,
           label="warned")
ax[1].axvline(S, color=INK, ls="--", lw=0.7)
ax[1].set_ylim(0, ax[1].get_ylim()[1] * 1.25)
ax[1].text(S - 10, ax[1].get_ylim()[1] * 0.92, "sight distance\n500 ft", fontsize=6,
           ha="right")
ax[1].axvline(ssd, color=GOLD, ls=":", lw=1)
ax[1].text(ssd + 10, ax[1].get_ylim()[1] * 0.92, "AASHTO SSD\n645 ft", fontsize=6)
ax[1].set(xlabel="Distance needed to stop (ft)", ylabel="Density",
          title="(b) Stopping at a hidden queue")
ax[1].legend(fontsize=6, loc="center right", bbox_to_anchor=(1, 0.55), handlelength=1.0,
             handletextpad=0.4, borderpad=0)
# (c) by sight distance
Ss = np.arange(300, 1001, 25)
pu = [p_fail_quad(s, V_MU, V_SD, *PRT["unwarned"]) for s in Ss]
pw = [p_fail_quad(s, VW_MU, VW_SD, *PRT["warned"]) for s in Ss]
pm = [respond * b + (1 - respond) * u for u, b in zip(pu, pw)]
for y, c, lab in [(pu, RUST, "no warning"), (pm, GOLD, "with message B"),
                  (pw, SAGE, "drivers who act")]:
    ax[2].plot(Ss, y, color=c, lw=1.2, label=lab)
for s in [400, 500, 600, 700, 800]:
    d_u, d_w = stop_dist(v0, t_u, a), stop_dist(v_w, t_w, a)
    ax[2].plot(s, (d_u > s).mean(), "o", color=RUST, mec=INK, ms=3)
    ax[2].plot(s, (d_w > s).mean(), "o", color=SAGE, mec=INK, ms=3)
ax[2].set_yscale("log")
ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax[2].set(ylim=(1e-4, 1.2), xlabel="Sight distance to the queue (ft)",
          ylabel="Share who cannot stop",
          title="(c) Quadrature vs. simulation")
ax[2].legend(fontsize=6, loc="lower left")
fig.tight_layout(); fig.savefig("chap09_fig01_dms_queue.png", dpi=300)
