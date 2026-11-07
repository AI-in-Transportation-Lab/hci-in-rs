"""Chapter 8 - Case Study 2: Inclusive Interface Design for Accessibility, Language, and Equity.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/08a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch08` or `python scripts/run_all.py --chapter 8`;
outputs are written to outputs/ch08/.
"""
import numpy as np
import pandas as pd
from itertools import combinations

# ---------- Comprehension test of three warning pictograms, by language group ----------
rng = np.random.default_rng(81)
LANG = ["English", "Spanish", "Vietnamese", "Arabic"]
TRUE = {"P1": [0.97, 0.96, 0.58, 0.93],       # true comprehension (unknown to the tester)
        "P2": [0.84, 0.82, 0.80, 0.81],
        "P3": [0.74, 0.70, 0.62, 0.66]}
n, CRIT = 50, 0.67                            # respondents per group; acceptance criterion


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - h, c + h


rows = []
for sym, rates in TRUE.items():
    k = rng.binomial(n, rates)
    for g, kk in list(zip(LANG, k)) + [("Pooled", k.sum())]:
        m = n * len(LANG) if g == "Pooled" else n
        lo, hi = wilson(kk, m)
        rows.append((sym, g, kk, m, kk / m, lo, hi))
test = pd.DataFrame(rows, columns=["symbol", "group", "correct", "n", "share", "lo", "hi"])
test["meets_67"] = np.where(test.share >= CRIT, "yes", "no")
test["lower_bound_67"] = np.where(test.lo >= CRIT, "yes", "no")
print(test.round(3).to_string(index=False))
by_group = test[test.group != "Pooled"].pivot(index="symbol", columns="group",
                                              values="share")
pooled = test[test.group == "Pooled"].set_index("symbol").share
CHOSEN = by_group.min(axis=1).idxmax()
print(f"\nHighest pooled score: {pooled.idxmax()} ({pooled.max():.3f}); "
      f"highest worst-group score: {CHOSEN} ({by_group.min(axis=1).max():.3f})")
COMP = by_group.loc[CHOSEN].to_dict()          # comprehension of the chosen symbol

# ---------- A corridor's pedestrians: what each can perceive and understand ----------
N = 50_000
pop = pd.DataFrame({"older": rng.random(N) < 0.18})
pop["blind_lv"] = rng.random(N) < np.where(pop.older, 0.06, 0.015)    # blind or low vision
# deaf or hard of hearing
pop["deaf_hoh"] = rng.random(N) < np.where(pop.older, 0.12, 0.02)
pop["lang"] = rng.choice(LANG, N, p=[0.88, 0.08, 0.02, 0.02])
pop["lep"] = (pop.lang != "English") & (rng.random(N) < 0.55)         # limited English
pop["low_lit"] = rng.random(N) < 0.10
pop["phone"] = rng.random(N) < np.where(pop.older, 0.76, 0.92)
pop["app"] = pop.phone & (rng.random(N) < 0.30)                       # alert app installed
reads_pictogram = rng.random(N) < pop.lang.map(COMP)
FEATURES = {"APS": 15, "Multilingual speech": 3, "Pictogram sign": 2, "Alert app": 5}  # $k
GROUPS = {"All": np.ones(N, bool), "Older adults": pop.older.to_numpy(),
          "Blind or low vision": pop.blind_lv.to_numpy(),
          "Deaf or hard of hearing": pop.deaf_hoh.to_numpy(),
          "Limited English": pop.lep.to_numpy(), "Low literacy": pop.low_lit.to_numpy(),
          "No smartphone": ~pop.phone.to_numpy()}


def reach(bundle):
    # True where a person receives BOTH the walk indication and the turning-vehicle warning
    sees, hears = ~pop.blind_lv, ~pop.deaf_hoh
    english = (pop.lang == "English") | ~pop.lep
    aps, ml = "APS" in bundle, "Multilingual speech" in bundle
    app = pop.app & ("Alert app" in bundle)
    # signal, tone, vibration
    walk = sees | (aps & hears) | (aps & pop.blind_lv) | app
    speech_ok = aps & hears & (english | ml)
    # text sign, speech, app
    warn = (sees & english & ~pop.low_lit) | speech_ok | app
    if "Pictogram sign" in bundle:
        warn = warn | (sees & reads_pictogram)
    return (walk & warn).to_numpy()


SHOW = [(), ("Pictogram sign",), ("Pictogram sign", "Alert app"),
        ("APS", "Pictogram sign"), ("APS", "Multilingual speech", "Pictogram sign")]
CODE = {"APS": "APS", "Multilingual speech": "ML", "Pictogram sign": "Pict",
        "Alert app": "App"}


def code(bundle):
    return "+".join(CODE[f] for f in bundle) or "Base"


table = pd.DataFrame({" + ".join(b) or "Baseline": {g: reach(b)[m].mean()
                      for g, m in GROUPS.items()} for b in SHOW}).T
print("Share of each group that receives both messages:")
print(table.set_axis([code(b) for b in SHOW]).T.round(3).to_string())
print("\nPeople in each group:")
print(pd.Series({g: int(m.sum()) for g, m in GROUPS.items()}).to_string())

# ---------- Choosing features under a budget: average reach or worst-off group ----------
BUDGET = 20
SHORT = {"All": "All", "Older adults": "Older", "Blind or low vision": "Blind/LV",
         "Deaf or hard of hearing": "Deaf/HoH", "Limited English": "Ltd English",
         "Low literacy": "Low literacy", "No smartphone": "No phone"}
cands = []
for r in range(len(FEATURES) + 1):
    for b in combinations(FEATURES, r):
        if "Multilingual speech" in b and "APS" not in b:
            continue                                 # speech needs an APS unit
        cost = sum(FEATURES[f] for f in b)
        if cost <= BUDGET:
            ok = reach(b)
            per = {g: ok[m].mean() for g, m in GROUPS.items()}
            cands.append((code(b), " + ".join(b) or "Baseline", cost, per["All"],
                          min(per.values()), SHORT[min(per, key=per.get)]))
cands = pd.DataFrame(cands, columns=["code", "bundle", "cost_k", "mean_reach",
                                     "min_group_reach", "worst_group"])
base = cands.mean_reach.min()
cands["reach_per_k"] = (cands.mean_reach - base) / cands.cost_k.replace(0, np.nan)
print(cands.drop(columns="bundle").sort_values("mean_reach", ascending=False)
      .round(3).to_string(index=False))
pick = {"Best value per $1k": cands.loc[cands.reach_per_k.idxmax()],
        "Highest mean reach": cands.loc[cands.mean_reach.idxmax()],
        "Highest worst-group reach": cands.loc[cands.min_group_reach.idxmax()]}
print()
for rule, row in pick.items():
    print(f"{rule:<27}{row.code:<13}cost {row.cost_k:>3.0f}k  mean {row.mean_reach:.3f}  "
          f"worst {row.min_group_reach:.3f} ({row.worst_group})")

# ---------- Figure: comprehension, reach by group, and the worst-off group ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
fig = plt.figure(figsize=(6.2, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1.3, 1], hspace=0.45,
                      wspace=0.25)
a0, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
a1 = fig.add_subplot(gs[1, :])
groups = LANG + ["Pooled"]                                       # (a) comprehension
for k, (sym, c) in enumerate([("P1", GOLD), ("P2", SAGE), ("P3", SKY)]):
    t = test[test.symbol == sym].set_index("group").loc[groups]
    x = np.arange(len(groups)) + (k - 1) * 0.25
    a0.errorbar(x, t.share, yerr=[t.share - t.lo, t.hi - t.share], fmt="o", color=c,
                mec=INK, ms=3.6, capsize=1.2, ecolor=INK, label=sym)
a0.axhline(CRIT, color=RUST, ls="--", lw=0.7, label="two-thirds criterion")
a0.set_xticks(np.arange(len(groups)), groups, fontsize=6.5)
a0.set(ylim=(0.15, 1.03), ylabel="Share who understood the symbol",
       title="(a) Pictogram comprehension, 95% Wilson intervals")
a0.legend(fontsize=6, loc="lower left", ncol=4, handlelength=1.5,
          columnspacing=1.0)
cm = LinearSegmentedColormap.from_list("reach", [RUST, GOLD, "#F6E7A6", SAGE])  # (b) reach
im = a1.imshow(table.to_numpy(), cmap=cm, vmin=0, vmax=1, aspect="auto")
for i in range(table.shape[0]):
    for j in range(table.shape[1]):
        a1.text(j, i, f"{table.iat[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
WRAP = {"Older adults": "Older\nadults", "Blind or low vision": "Blind or\nlow vision",
        "Deaf or hard of hearing": "Deaf or\nhard of hearing",
        "Limited English": "Limited\nEnglish", "Low literacy": "Low\nliteracy",
        "No smartphone": "No\nsmartphone"}
a1.set_xticks(range(table.shape[1]), [WRAP.get(c, c) for c in table.columns], fontsize=6.5)
a1.set_yticks(range(table.shape[0]), table.index, fontsize=6.5)
a1.grid(False)
a1.set_title("(c) Share of each group that receives both messages")
fig.colorbar(im, ax=a1, fraction=0.02, pad=0.01)
for _, r in cands.iterrows():                                    # (c) worst-off group
    a2.plot(r.cost_k, r.min_group_reach, "o", ms=4 + 7 * (r.mean_reach - 0.8) / 0.2,
            color=SAGE if "APS" in r.bundle else GOLD, mec=INK, alpha=0.9)
LABEL = {"Baseline": (-1, 7, "left"), "Pictogram sign": (7, -2, "left"),
         "Pictogram sign + Alert app": (0, 7, "center"),
         "APS + Pictogram sign": (-6, -9, "right"),
         "APS + Multilingual speech + Pictogram sign": (-6, 4, "right")}
for _, r in cands.iterrows():
    if r.bundle in LABEL:
        dx, dy, ha = LABEL[r.bundle]
        a2.annotate(r.bundle.replace("Multilingual speech", "ML speech"),
                    (r.cost_k, r.min_group_reach), xytext=(dx, dy), ha=ha,
                    textcoords="offset points", fontsize=6)
a2.set(xlabel="Cost per intersection ($ thousand)", ylabel="Reach of the worst-off group",
       xlim=(-1, 23), ylim=(-0.05, 1.08), title="(b) Only bundles with APS lift the floor")
fig.savefig("chap08_fig02_inclusive_design.png", dpi=300, bbox_inches="tight")
