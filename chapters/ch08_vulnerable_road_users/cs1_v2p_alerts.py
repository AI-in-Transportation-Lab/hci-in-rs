"""Chapter 8 - Case Study 1: V2P Safety Alerts for Pedestrians and Cyclists.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/08a.tex (3 listing blocks, concatenated in book order).
Run from the repository root with `make ch08` or `python scripts/run_all.py --chapter 8`;
outputs are written to outputs/ch08/.
"""
import numpy as np
import pandas as pd

# ---------- Encounters between crossing road users and turning vehicles ----------
rng = np.random.default_rng(8)
N = 200_000                                  # encounters at the moment of decision
T_DANGER = 3.0                               # dangerous if the conflict is under 3 s away
ttc = rng.gamma(4.0, 1.6, N)                 # true time to conflict (s)
speed = rng.uniform(6, 12, N)                # vehicle speed (m/s)
danger = ttc < T_DANGER
z_pos, z_rt = rng.normal(size=N), rng.normal(size=N)    # common random numbers
u_commit, u_comply = rng.random(N), rng.random(N)
print(f"Encounters {N:,}; dangerous share {danger.mean():.3f}; "
      f"median time to conflict {np.median(ttc):.1f} s")

# ---------- Users, alert modalities, and the response they allow ----------
GROUPS = {"Adult pedestrian": ((1.0, 2.5), 0.0),   # commitment window (s), extra RT (s)
          "Older pedestrian": ((1.5, 3.0), 0.4),
          "Cyclist": ((0.6, 1.4), 0.0)}
RT = {"Adult pedestrian": {"visual": 1.2, "auditory": 0.9, "haptic": 0.8},
      "Older pedestrian": {"visual": 1.2, "auditory": 0.9, "haptic": 0.8},
      "Cyclist": {"visual": 1.6, "auditory": 1.0, "haptic": 0.8}}   # median RT (s)
P_MAX, K_CRY = 0.95, 1.2                     # compliance, and its decay with false alarms


def run(tau, sigma=3.0, latency=0.3, group="Adult pedestrian", mode="haptic",
        cry_wolf=True):
    (c_lo, c_hi), slow = GROUPS[group]
    est = ttc + np.sqrt(2) * sigma / speed * z_pos           # two GNSS fixes
    alert = est < tau
    # share of alerts that are false
    fap = (alert & ~danger).sum() / max(alert.sum(), 1)
    comply = P_MAX * np.exp(-K_CRY * fap) if cry_wolf else P_MAX
    rt = (RT[group][mode] + slow) * np.exp(0.25 * z_rt)
    commit = c_lo + (c_hi - c_lo) * u_commit                 # time until the user commits
    saved = danger & alert & (latency + rt < commit) & (u_comply < comply)
    return {"tau": tau, "hit": alert[danger].mean(),
            "fa_per100": 100 * (alert & ~danger).mean(),
            "false_share": fap, "comply": comply, "prevented": saved.sum() / danger.sum()}


sweep = pd.DataFrame([run(t) for t in np.arange(2.0, 7.01, 0.5)])
sweep["no_crywolf"] = [run(t, cry_wolf=False)["prevented"] for t in sweep.tau]
print("\nAdult pedestrians, haptic alerts, 3-m positioning error, 0.3-s latency:")
print(sweep.round(3).to_string(index=False))
best = sweep.loc[sweep.prevented.idxmax()]
print(f"\nBest threshold {best.tau:.1f} s: prevents {best.prevented:.1%} of dangerous "
      f"encounters,\nwith {best.fa_per100:.1f} false alerts per 100 encounters")

# ---------- What positioning error and latency cost ----------
taus = np.arange(2.0, 7.01, 0.25)
grid = []
for sigma in [1.0, 3.0, 5.0]:
    for latency in [0.1, 0.5, 1.0]:
        res = [run(t, sigma=sigma, latency=latency) for t in taus]
        b = max(res, key=lambda r: r["prevented"])
        grid.append((sigma, latency, b["tau"], b["hit"], b["fa_per100"], b["prevented"]))
grid = pd.DataFrame(grid, columns=["gnss_m", "latency_s", "best_tau", "hit",
                                   "fa_per100", "prevented"])
print("Best achievable protection by positioning error and latency (adult, haptic):")
print(grid.round(3).to_string(index=False))

# ---------- Who benefits: user group by alert modality ----------
rows = []
for group in GROUPS:
    for mode in ["visual", "auditory", "haptic"]:
        res = [run(t, group=group, mode=mode) for t in taus]
        b = max(res, key=lambda r: r["prevented"])
        rows.append((group, mode, b["tau"], b["prevented"]))
who = pd.DataFrame(rows, columns=["group", "modality", "best_tau", "prevented"])
print("\nProtection by user group and modality (3-m error, 0.3-s latency):")
print(who.pivot(index="group", columns="modality", values="prevented")
      .loc[list(GROUPS), ["visual", "auditory", "haptic"]].round(3).to_string())

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
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
# (a) hit vs false alerts
for sigma, c in [(1.0, SAGE), (3.0, GOLD), (5.0, RUST)]:
    res = pd.DataFrame([run(t, sigma=sigma) for t in taus])
    ax[0].plot(res.fa_per100, res.hit, "-", color=c, lw=2, label=f"{sigma:.0f}-m error")
    r3 = res.iloc[np.argmin(np.abs(res.tau - 3.5))]
    ax[0].plot(r3.fa_per100, r3.hit, "o", color=c, mec=INK, ms=6)
ax[0].set(xlabel="False alerts per 100 encounters", ylabel="Share of dangerous "
          "encounters alerted", xlim=(0, 40), title="(a) Detection vs. false alerts")
ax[0].legend(fontsize=8, frameon=False, loc="lower right")
ax[0].text(12, 0.55, "dots: threshold 3.5 s", fontsize=7)
for latency, ls in [(0.3, "-"), (1.0, "--")]:                       # (b) protection vs tau
    for cw, c in [(True, RUST), (False, SAGE)]:
        p = [run(t, latency=latency, cry_wolf=cw)["prevented"] for t in taus]
        ax[1].plot(taus, p, ls, color=c, lw=2,
                   label=f"{'with' if cw else 'no'} cry-wolf, {latency:.1f}-s latency")
ax[1].set(xlabel="Alert threshold (s to conflict)",
          ylabel="Dangerous encounters prevented",
          ylim=(0, 1), title="(b) False alerts set the optimum")
ax[1].legend(fontsize=7, frameon=False, loc="upper right")
x = np.arange(3)                                                    # (c) who benefits
for k, (mode, c) in enumerate([("visual", SKY), ("auditory", GOLD), ("haptic", SAGE)]):
    vals = [who[(who.group == g) & (who.modality == mode)].prevented.iloc[0]
            for g in GROUPS]
    ax[2].bar(x + (k - 1) * 0.26, vals, 0.25, color=c, label=mode, edgecolor=INK, lw=0.5)
ax[2].set_xticks(x, [g.replace(" ", "\n") for g in GROUPS], fontsize=8)
ax[2].set(ylim=(0, 0.8), ylabel="Dangerous encounters prevented",
          title="(c) The best alert depends on the user")
ax[2].legend(fontsize=8, frameon=False, loc="upper left", ncol=3)
fig.tight_layout(); fig.savefig("chap08_fig01_v2p_alerts.png", dpi=200)
