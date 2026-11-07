"""Chapter 10 - Case Study 2: Translating Naturalistic Driving Evidence into Safety Claims.

Book: Human-Computer Interaction in Roadway Safety: Principles, Methods, and Applications
Source: chapters/10a.tex (4 listing blocks, concatenated in book order).
Run from the repository root with `make ch10` or `python scripts/run_all.py --chapter 10`;
outputs are written to outputs/ch10/.
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable

# ---------- A naturalistic study: 6-s epochs, tasks, crashes, near-crashes ----------
rng = np.random.default_rng(101)
M = 4_000_000                                   # driving epochs in the study population
teen = rng.random(M) < 0.25                     # younger drivers use phones more
TASK = np.array(["none", "visual-manual phone", "phone talk", "eating"])
task = np.where(teen, rng.choice(4, M, p=[0.80, 0.12, 0.04, 0.04]),
                rng.choice(4, M, p=[0.89, 0.05, 0.04, 0.02]))
LOG_OR = np.log([1.0, 3.5, 1.3, 1.6])           # true odds ratios against no task
p_event = expit(np.where(teen, -8.3, -9.0) + LOG_OR[task])
event = rng.random(M) < p_event
crash = event & (rng.random(M) < 0.15)          # the rest of the events are near-crashes
base = rng.choice(np.flatnonzero(~event), 10_000, replace=False)
cc = pd.DataFrame({"case": np.r_[np.ones(event.sum()), np.zeros(10_000)].astype(int),
                   "crash": np.r_[crash[event], np.zeros(10_000, bool)],
                   "teen": np.r_[teen[event], teen[base]].astype(int),
                   "task": np.r_[task[event], task[base]]})
print(f"{event.sum()} crash and near-crash events ({crash.sum()} crashes); "
      f"10,000 baseline epochs")


def woolf(a, b, c, d):
    # odds ratio and 95% CI from a 2x2 table: exposed cases a, unexposed cases b,
    # exposed controls c, unexposed controls d
    o, se = a * d / (b * c), np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return o, o * np.exp(-1.96 * se), o * np.exp(1.96 * se), se


pv = cc[cc.task.isin([0, 1])].assign(phone=lambda x: (x.task == 1).astype(int))
tab = pd.crosstab(pv.case, pv.phone)
a, b, c, d = tab.loc[1, 1], tab.loc[1, 0], tab.loc[0, 1], tab.loc[0, 0]
crude = woolf(a, b, c, d)
fit = smf.logit("case ~ phone", pv).fit(disp=0)
pop_or = (event[task == 1].mean() / (1 - event[task == 1].mean())) / \
    (event[task == 0].mean() / (1 - event[task == 0].mean()))
print(f"\nVisual-manual phone use vs no task: {a} of {a + b} events, "
      f"{c} of {c + d} baselines")
print(f"Crude OR in the whole population {pop_or:.3f}; true OR within each age group 3.5")
print(f"Crude OR {crude[0]:.4f} [{crude[1]:.3f}, {crude[2]:.3f}]; logistic regression OR "
      f"{np.exp(fit.params.phone):.4f};\nSE of log OR {crude[3]:.5f} (Woolf) vs "
      f"{fit.bse.phone:.5f} (logistic)")

# ---------- Age adjustment: Mantel-Haenszel by hand, statsmodels, regression ----------
strata = [pd.crosstab(g.case, g.phone).loc[[1, 0], [1, 0]].to_numpy()
          for _, g in pv.groupby("teen")]
num = sum(t[0, 0] * t[1, 1] / t.sum() for t in strata)
den = sum(t[0, 1] * t[1, 0] / t.sum() for t in strata)
st = StratifiedTable(strata)
fit2 = smf.logit("case ~ phone + teen", pv).fit(disp=0)
lo, hi = st.oddsratio_pooled_confint()
print(f"Mantel-Haenszel OR {num / den:.4f} (by hand), {st.oddsratio_pooled:.4f} "
      f"(statsmodels) [{lo:.3f}, {hi:.3f}]; regression {np.exp(fit2.params.phone):.4f}")
cr = pv[(pv.case == 0) | pv.crash]
t2 = pd.crosstab(cr.case, cr.phone)
co = woolf(t2.loc[1, 1], t2.loc[1, 0], t2.loc[0, 1], t2.loc[0, 0])
print(f"Crashes only: OR {co[0]:.3f} [{co[1]:.3f}, {co[2]:.3f}], interval "
      f"{co[2] / co[1]:.2f} times wide vs {crude[2] / crude[1]:.2f} with near-crashes")

# ---------- An AI video coder (LLaVA-type vision-language model) labels phones ----------
from sklearn.metrics import cohen_kappa_score
SE_AI, SP_AI = 0.85, 0.95                        # true sensitivity and specificity
truth = (cc.task == 1).to_numpy()
ai = np.where(truth, rng.random(len(cc)) < SE_AI, rng.random(len(cc)) > SP_AI)


def or_from(x):
    t = pd.crosstab(cc.case, x)
    return t.loc[1, True] * t.loc[0, False] / (t.loc[1, False] * t.loc[0, True])


def rogan_gladen(p_obs, se, sp):
    return (p_obs + sp - 1) / (se + sp - 1)


def corrected_or(p1, p0, se, sp):
    q1, q0 = rogan_gladen(p1, se, sp), rogan_gladen(p0, se, sp)
    return q1 / (1 - q1) / (q0 / (1 - q0))


case = cc.case.to_numpy() == 1
p1, p0 = ai[case].mean(), ai[~case].mean()
or_true, or_naive = or_from(truth), or_from(ai)
print(f"Phone use vs everything else: OR from human labels {or_true:.3f}; "
      f"from AI labels {or_naive:.3f}")
rows = []
for n_val in [400, 1500]:                        # epochs coded by people as well as the AI
    val = rng.choice(len(cc), n_val, replace=False)
    tv, av = truth[val], ai[val]
    se_hat, sp_hat = av[tv].mean(), (~av[~tv]).mean()
    po = (tv == av).mean()
    pe = tv.mean() * av.mean() + (1 - tv.mean()) * (1 - av.mean())
    boot = []
    for _ in range(2000):                        # resample validation and main samples
        j = rng.integers(0, n_val, n_val)
        s1, s0 = av[j][tv[j]].mean(), (~av[j][~tv[j]]).mean()
        b1 = rng.binomial(case.sum(), p1) / case.sum()
        b0 = rng.binomial((~case).sum(), p0) / (~case).sum()
        boot.append(corrected_or(b1, b0, s1, s0))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rows.append((n_val, tv.sum(), se_hat, sp_hat, (po - pe) / (1 - pe),
                 cohen_kappa_score(tv, av), corrected_or(p1, p0, se_hat, sp_hat), lo, hi))
vt = pd.DataFrame(rows, columns=["validated", "phone_epochs", "sens", "spec", "kappa",
                                 "kappa_sklearn", "corrected_OR", "lo", "hi"])
print(vt.round(3).to_string(index=False))

# ---------- From an odds ratio to a safety claim ----------
or_mh = st.oddsratio_pooled
pc = a / (a + b)                                 # exposure among events (phone vs no task)
p0c = c / (c + d)                                # exposure among baseline epochs
paf_miettinen = pc * (or_mh - 1) / or_mh
paf_levin = p0c * (crude[0] - 1) / (1 + p0c * (crude[0] - 1))
m = np.isin(task, [0, 1])                        # population, phone vs no task
risk = p_event[m]
cf = expit(np.where(teen[m], -8.3, -9.0))        # every phone epoch made a no-task epoch
paf_true = 1 - cf.sum() / risk.sum()
half = 1 - (risk.sum() - 0.5 * (risk - cf).sum()) / risk.sum()
print(f"Attributable fraction: true {paf_true:.4f}; Miettinen (adjusted OR) "
      f"{paf_miettinen:.4f}; Levin (crude OR) {paf_levin:.4f}")
bp = []
for _ in range(2000):
    ra, rb = rng.multinomial(a + b, [pc, 1 - pc])
    tb = [rng.multinomial(t.sum(), (t / t.sum()).ravel()).reshape(2, 2) for t in strata]
    o = StratifiedTable(tb).oddsratio_pooled
    bp.append(ra / (ra + rb) * (o - 1) / o)
lo, hi = np.percentile(bp, [2.5, 97.5])
print(f"Claim: halving visual-manual phone use would prevent {0.5 * paf_miettinen:.1%} of "
      f"crash and near-crash\nevents in these conditions [95% CI {0.5 * lo:.1%}, "
      f"{0.5 * hi:.1%}]; exact value in the population {half:.1%}")

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
fig = plt.figure(figsize=(6.2, 5.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.28)
a0 = fig.add_subplot(gs[0, :])
a1, a2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ci2 = np.exp(fit2.conf_int().loc["phone"])
naive_ci = woolf(*[pd.crosstab(cc.case, ai).loc[i, j] for i, j in
                   [(1, True), (1, False), (0, True), (0, False)]])
est = [("Crude, case-control sample", crude[0], crude[1], crude[2], RUST),
       ("Mantel-Haenszel, by age", st.oddsratio_pooled, *st.oddsratio_pooled_confint(),
        SAGE),
       ("Logistic regression with age", np.exp(fit2.params.phone), ci2[0], ci2[1], SAGE),
       ("Crashes only, crude", co[0], co[1], co[2], GOLD),
       ("AI labels, uncorrected", naive_ci[0], naive_ci[1], naive_ci[2], RUST),
       ("AI corrected, 400 validated", vt.corrected_OR[0], vt.lo[0], vt.hi[0], SKY),
       ("AI corrected, 1,500 validated", vt.corrected_OR[1], vt.lo[1], vt.hi[1], SKY)]
y = np.arange(len(est))[::-1].astype(float)
y[4:] -= 0.6
for yy, (lab, e, lo_, hi_, c) in zip(y, est):                     # (a) forest plot
    a0.plot([lo_, hi_], [yy, yy], color=c, lw=1.4, solid_capstyle="round")
    a0.plot(e, yy, "o", color=c, mec=INK, ms=4.2)
    a0.text(0.97, yy, lab, ha="right", va="center", fontsize=6.5)
a0.axvline(3.5, color=INK, ls="--", lw=0.7)
a0.text(3.42, y[0] + 0.55, "true OR, phone vs no task (3.5)", fontsize=6.5, ha="right")
a0.axvline(or_true, color=INK, ls=":", lw=0.7)
a0.text(or_true * 1.04, y[-1] - 0.75,
        f"human-coded OR, phone vs all other ({or_true:.2f})",
        fontsize=6.5, ha="left")
a0.set_xscale("log")
a0.set_xticks([1, 2, 3, 4, 6, 10, 16], ["1", "2", "3", "4", "6", "10", "16"])
a0.spines["left"].set_visible(False)
a0.set(xlim=(0.35, 20), ylim=(y[-1] - 1.1, y[0] + 1.0), yticks=[],
       xlabel="Odds ratio for visual-manual phone use (log scale)",
       title="(a) The same question, seven estimates")
sp_grid = np.linspace(0.80, 1.0, 200)                              # (b) attenuation
q1, q0 = truth[case].mean(), truth[~case].mean()
for se_, c in [(0.85, RUST), (0.95, SAGE)]:
    o1, o0 = se_ * q1 + (1 - sp_grid) * (1 - q1), se_ * q0 + (1 - sp_grid) * (1 - q0)
    a1.plot(sp_grid, o1 / (1 - o1) / (o0 / (1 - o0)), color=c, lw=1.2,
            label=f"sensitivity {se_:.2f}")
a1.plot(SP_AI, or_naive, "D", color=INK, ms=4.2, label="simulated AI coder")
a1.axhline(or_true, color=INK, ls=":", lw=0.7)
a1.set(xlabel="Specificity of the AI coder", ylabel="Uncorrected odds ratio",
       title="(b) Coding errors pull the OR toward 1")
a1.legend(fontsize=6.5, loc="center left")
a2.hist(np.array(bp) * 0.5 * 100, bins=40, color=SKY, edgecolor="white")  # (c) the claim
for v, c, lab in [(half, INK, "exact"), (0.5 * paf_miettinen, SAGE, "Miettinen, adj."),
                  (0.5 * paf_levin, RUST, "Levin, crude")]:
    a2.axvline(v * 100, ymax=0.74, color=c, lw=1.2, ls="--" if lab == "exact" else "-", label=lab)
a2.set(xlabel="Events prevented by halving phone use (%)", ylabel="Bootstrap replicates",
       title="(c) A claim with its uncertainty")
a2.set_ylim(0, a2.get_ylim()[1] * 1.3)
a2.legend(fontsize=6.5, loc="upper left")
fig.savefig("chap10_fig02_nds_claims.png", dpi=300, bbox_inches="tight")
