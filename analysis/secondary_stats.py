"""Inferential statistics for the secondary analyses (second revision).

Everything is computed from the stored predictions; no model is retrained.

  * transfer comparisons (exact McNemar test per seed, Holm correction within each seed and test set)
  * 95% bootstrap confidence intervals for accuracy and macro-F1 on every test set
  * in-domain versus cross-dataset accuracy (unpaired bootstrap interval of the difference)
  * calibration differences (paired bootstrap interval of the difference in expected calibration error)
  * monocyte recall (exact McNemar test restricted to monocyte images)
  * few-shot adaptation (descriptive: per-repetition differences with unpaired Newcombe intervals)

Usage: python secondary_stats.py <repo root> <out.json>
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import binomtest, norm

root, out = sys.argv[1], sys.argv[2]
B = 10000                      # bootstrap resamples
RNG_SEED = 20261126
SEEDS = [0, 1, 2]
TTA = {"none": [0], "6": [0, 4, 5, 1, 2, 3]}
CFG = {"M4": ("M4", "none"), "M5": ("M5", "none"), "M7": ("M5", "6"), "M8": ("M8", "none"), "final": ("M8", "6"),
       "X_noCLAHE": ("X_noCLAHE", "none"), "B_resnet50": ("B_resnet50", "none"), "LOO_ce": ("LOO_ce", "6")}


def load(run, stage, seed, split, tta):
    if run == "raabin":
        z = np.load(f"{root}/predictions/raabin/preds_RAABIN_{stage}_s{seed}.npz")["RAABIN"]
    else:
        z = np.load(f"{root}/predictions/{run}/preds_{stage}_s{seed}.npz")[split]
    return z[:, TTA[tta]].astype(np.float64).mean(1)


def labels(run, split):
    f = f"{root}/predictions/raabin/meta_RAABIN.csv" if run == "raabin" else f"{root}/predictions/{run}/meta_{split}.csv"
    return pd.read_csv(f).y.values


def prob(run, cfg, seed, split):
    stage, tta = CFG[cfg]
    return load(run, stage, seed, split, tta)


def mcnemar(ca, cb):
    """Exact McNemar test. ca, cb: boolean correctness of configurations a and b on the same images."""
    n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return dict(b_only=n01, a_only=n10, p=float(p))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj.tolist()


def macro_f1(y, yh):
    cm = np.bincount(y * 4 + yh, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm); den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def ece(p, y, bins=15):
    conf, ok = p.max(1), p.argmax(1) == y
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    n = np.bincount(idx, minlength=bins).astype(float)
    sc = np.bincount(idx, weights=conf, minlength=bins); so = np.bincount(idx, weights=ok.astype(float), minlength=bins)
    return float(np.abs(so - sc).sum() / len(y))


def scale(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T; z -= z.max(1, keepdims=True); q = np.exp(z)
    return q / q.sum(1, keepdims=True)


def fit_T(p, y):
    best = (1.0, np.inf)
    for T in np.arange(0.05, 5.0001, 0.01):
        nll = -np.mean(np.log(np.clip(scale(p, T)[np.arange(len(y)), y], 1e-12, 1)))
        if nll < best[1]: best = (round(float(T), 2), nll)
    return best[0]


def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


rng = np.random.default_rng(RNG_SEED)
res = {"bootstrap_resamples": B, "rng_seed": RNG_SEED}

# ---------------------------------------------------------------- 1. test sets and configurations
SETS = {"PBC->PBC": ("pbc", "TEST"), "PBC->Kaggle": ("pbc", "KTEST"), "PBC->Raabin": ("raabin", "RAABIN"),
        "Kaggle->Kaggle": ("kaggle", "TEST"), "Kaggle->PBC": ("kaggle", "EXT")}
Y = {k: labels(*v) for k, v in SETS.items()}
res["n"] = {k: int(len(v)) for k, v in Y.items()}

# ---------------------------------------------------------------- 2. bootstrap intervals for accuracy and macro-F1
res["ci"] = {}
for name, (run, split) in SETS.items():
    y = Y[name]; n = len(y)
    for cfg in ["M5", "M8", "X_noCLAHE", "final", "B_resnet50"]:
        for s in SEEDS:
            yh = prob(run, cfg, s, split).argmax(1)
            ix = rng.integers(0, n, (B, n), dtype=np.int32)
            acc = (yh[ix] == y[ix]).mean(1)
            f1 = np.array([macro_f1(y[i], yh[i]) for i in ix])
            res["ci"].setdefault(name, {}).setdefault(cfg, {})[s] = dict(
                acc=float((yh == y).mean()), acc_ci=ci(acc), f1=macro_f1(y, yh), f1_ci=ci(f1))

# ---------------------------------------------------------------- 3. transfer comparisons (exact McNemar, Holm per seed and test set)
PAIRS = [("X_noCLAHE", "M5"), ("B_resnet50", "M5"), ("M5", "M8"), ("M8", "final")]
res["transfer_mcnemar"] = {}
for name in ["PBC->Kaggle", "PBC->Raabin", "Kaggle->PBC"]:
    run, split = SETS[name]; y = Y[name]
    for s in SEEDS:
        rows = []
        for a, b in PAIRS:
            ca = prob(run, a, s, split).argmax(1) == y; cb = prob(run, b, s, split).argmax(1) == y
            r = mcnemar(ca, cb); r["diff_points"] = float((cb.mean() - ca.mean()) * 100); rows.append(r)
        for r, ph in zip(rows, holm([r["p"] for r in rows])): r["p_holm"] = ph
        for (a, b), r in zip(PAIRS, rows): res["transfer_mcnemar"].setdefault(name, {}).setdefault(f"{a}->{b}", {})[s] = r

# macro-F1 difference on Raabin-WBC, with and without CLAHE (paired bootstrap)
res["raabin_f1_diff"] = {}
y = Y["PBC->Raabin"]; n = len(y)
for s in SEEDS:
    ya = prob("raabin", "X_noCLAHE", s, "RAABIN").argmax(1); yb = prob("raabin", "M5", s, "RAABIN").argmax(1)
    ix = rng.integers(0, n, (B, n), dtype=np.int32)
    d = np.array([macro_f1(y[i], yb[i]) - macro_f1(y[i], ya[i]) for i in ix])
    res["raabin_f1_diff"][s] = dict(diff=macro_f1(y, yb) - macro_f1(y, ya), ci=ci(d))

# per-class recall of the final pipeline on Raabin-WBC
res["raabin_recall_ci"] = {}
for s in SEEDS:
    yh = prob("raabin", "final", s, "RAABIN").argmax(1)
    for c, cname in enumerate(["eos", "lym", "mon", "neu"]):
        m = np.where(y == c)[0]; ok = (yh[m] == c)
        bs = ok[rng.integers(0, len(m), (B, len(m)), dtype=np.int32)].mean(1)
        res["raabin_recall_ci"].setdefault(cname, {})[s] = dict(recall=float(ok.mean()), ci=ci(bs), n=int(len(m)))

# ---------------------------------------------------------------- 4. in-domain versus cross-dataset accuracy (unpaired bootstrap)
res["domain_gap"] = {}
for src, dst in [("PBC->PBC", "PBC->Kaggle"), ("PBC->PBC", "PBC->Raabin"), ("Kaggle->Kaggle", "Kaggle->PBC")]:
    for s in SEEDS:
        r1, s1 = SETS[src]; r2, s2 = SETS[dst]
        c1 = prob(r1, "final", s, s1).argmax(1) == Y[src]; c2 = prob(r2, "final", s, s2).argmax(1) == Y[dst]
        d = c1[rng.integers(0, len(c1), (B, len(c1)), dtype=np.int32)].mean(1) - c2[rng.integers(0, len(c2), (B, len(c2)), dtype=np.int32)].mean(1)
        res["domain_gap"].setdefault(f"{src} minus {dst}", {})[s] = dict(diff_points=float((c1.mean() - c2.mean()) * 100),
                                                                          ci_points=[x * 100 for x in ci(d)])

# ---------------------------------------------------------------- 5. calibration differences (paired bootstrap)
res["ece_diff"] = {}
def ece_pair(tag, pa, pb, y, s):
    n = len(y); ix = rng.integers(0, n, (B, n), dtype=np.int32)
    d = np.array([ece(pb[i], y[i]) - ece(pa[i], y[i]) for i in ix])
    res["ece_diff"].setdefault(tag, {})[s] = dict(a=ece(pa, y), b=ece(pb, y), diff=ece(pb, y) - ece(pa, y), ci=ci(d))
for run, tagp, split in [("pbc", "PBC", "TEST"), ("kaggle", "Kaggle", "TEST")]:
    y = labels(run, split); yv = labels(run, "VAL")
    for s in SEEDS:
        p4, p5, p7, pf = (prob(run, c, s, split) for c in ["M4", "M5", "M7", "final"])
        ece_pair(f"{tagp}: M4 -> M5 (focal loss)", p4, p5, y, s)
        ece_pair(f"{tagp}: M5 -> M7 (six-fold TTA)", p5, p7, y, s)
        T = fit_T(prob(run, "final", s, "VAL"), yv)
        ece_pair(f"{tagp}: final -> final after temperature scaling", pf, scale(pf, T), y, s)
        if run == "pbc":
            ece_pair("PBC: final -> final with cross-entropy", pf, prob(run, "LOO_ce", s, split), y, s)

# ---------------------------------------------------------------- 6. monocyte recall on PBC
res["monocyte_mcnemar"] = {}
y = Y["PBC->PBC"]; mono = y == 2
for a, b in [("M5", "M8"), ("M5", "final")]:
    for s in SEEDS:
        ca = (prob("pbc", a, s, "TEST").argmax(1) == y)[mono]; cb = (prob("pbc", b, s, "TEST").argmax(1) == y)[mono]
        r = mcnemar(ca, cb); r.update(recall_a=float(ca.mean()), recall_b=float(cb.mean()), n=int(mono.sum()))
        res["monocyte_mcnemar"].setdefault(f"{a}->{b}", {})[s] = r

# ---------------------------------------------------------------- 7. few-shot adaptation (descriptive)
def newcombe(p1, p2, n1, n2, z=norm.ppf(0.975)):
    """Newcombe hybrid-score interval for p1 - p2 (independent proportions)."""
    def wilson(p, n):
        c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return c - h, c + h
    l1, u1 = wilson(p1, n1); l2, u2 = wilson(p2, n2); d = p1 - p2
    return [float(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), float(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]
fs = pd.read_csv(f"{root}/results/fewshot_draws/fewshot_results.csv")
nK = res["n"]["Kaggle->Kaggle"]; res["fewshot"] = {}
for k in [10, 50]:
    for rep in SEEDS:
        g = fs[(fs.rep == rep)]
        a = float(g[(g.k == k) & (g.init == "pbc")].KTEST_none.iloc[0]); b = float(g[(g.k == k) & (g.init == "imagenet")].KTEST_none.iloc[0])
        z0 = float(g[(g.k == 0)].KTEST_none.iloc[0])
        full = float((prob("kaggle", "M8", rep, "TEST").argmax(1) == Y["Kaggle->Kaggle"]).mean())
        res["fewshot"].setdefault(str(k), {})[rep] = dict(
            pbc_start=a, imagenet_start=b, no_adaptation=z0, full_training_M8=full,
            pbc_minus_imagenet=dict(diff_points=(a - b) * 100, ci_points=[x * 100 for x in newcombe(a, b, nK, nK)]),
            pbc_minus_no_adaptation=dict(diff_points=(a - z0) * 100, ci_points=[x * 100 for x in newcombe(a, z0, nK, nK)]),
            pbc_minus_full_training=dict(diff_points=(a - full) * 100, ci_points=[x * 100 for x in newcombe(a, full, nK, nK)]))

# ---------------------------------------------------------------- 8. cluster-level resampling (added after the image-level analyses)
# Kaggle test images are augmented copies of a few original images (column "source"); Raabin-WBC Test-A file names start
# with a prefix (date and serial number) shared by many images, which we take to identify an acquisition session or smear.
# Here the bootstrap resamples these groups instead of single images. The random stream continues after the analyses above,
# so the earlier numbers are unchanged.
import os
kt = pd.read_csv(f"{root}/predictions/kaggle/meta_TEST.csv")
kp = pd.read_csv(f"{root}/predictions/pbc/meta_KTEST.csv")
assert (kp.file.map(os.path.basename).values == kt.file.values).all() and (kp.y.values == kt.y.values).all()   # same images, same order
GROUPS = {"PBC->Kaggle": kt.source.values, "Kaggle->Kaggle": kt.source.values,
          "PBC->Raabin": pd.read_csv(f"{root}/predictions/raabin/meta_RAABIN.csv").rel.map(lambda x: os.path.basename(x).split("_")[0]).values}
assert (kt.y.values == Y["PBC->Kaggle"]).all() and (kt.y.values == Y["Kaggle->Kaggle"]).all()
CLS = ["eos", "lym", "mon", "neu"]


def gsum(v, g):
    """Sum of v within each group; returns one row per group."""
    _, inv = np.unique(g, return_inverse=True)
    v = np.asarray(v, dtype=float)
    if v.ndim == 1: return np.bincount(inv, weights=v)
    return np.stack([np.bincount(inv, weights=v[:, j]) for j in range(v.shape[1])], 1)


def cluster_draws(g):
    G = len(np.unique(g)); return rng.integers(0, G, (B, G), dtype=np.int32)


cl = {"n_groups": {k: int(len(np.unique(v))) for k, v in GROUPS.items()},
      "groups_per_class": {k: {c: int(len(np.unique(v[Y[k] == i]))) for i, c in enumerate(CLS)} for k, v in GROUPS.items()}}
r = pd.read_csv(f"{root}/predictions/raabin/meta_RAABIN.csv"); gr = GROUPS["PBC->Raabin"]
top = pd.Series(gr[r.y.values == 0]).value_counts()
cl["raabin_eosinophils_in_largest_group"] = [int(top.iloc[0]), int((r.y.values == 0).sum())]

# accuracy and per-class recall of the final pipeline and paired accuracy differences, groups resampled
cl["acc"], cl["recall"], cl["paired_diff"] = {}, {}, {}
for name in ["PBC->Kaggle", "Kaggle->Kaggle", "PBC->Raabin"]:
    run, split = SETS[name]; y = Y[name]; g = GROUPS[name]; size = gsum(np.ones(len(y)), g)
    for s in SEEDS:
        yh = prob(run, "final", s, split).argmax(1); ok = yh == y
        ix = cluster_draws(g); k = gsum(ok, g)
        cl["acc"].setdefault(name, {})[s] = dict(acc=float(ok.mean()), ci=ci(k[ix].sum(1) / size[ix].sum(1)))
        for c, cname in enumerate(CLS):
            nc = gsum(y == c, g); kc = gsum(ok & (y == c), g); m = y == c
            den = nc[ix].sum(1); v = kc[ix].sum(1)[den > 0] / den[den > 0]
            bs = ok[m][rng.integers(0, m.sum(), (B, m.sum()), dtype=np.int32)].mean(1)
            cl["recall"].setdefault(name, {}).setdefault(cname, {})[s] = dict(
                recall=float(ok[m].mean()), n=int(m.sum()), groups=int((nc > 0).sum()), ci_image=ci(bs), ci_group=ci(v))
    if name == "Kaggle->Kaggle": continue
    for a, b in PAIRS:
        for s in SEEDS:
            ca = prob(run, a, s, split).argmax(1) == y; cb = prob(run, b, s, split).argmax(1) == y
            d = gsum(cb.astype(float) - ca.astype(float), g); ix = cluster_draws(g)
            cl["paired_diff"].setdefault(name, {}).setdefault(f"{a}->{b}", {})[s] = dict(
                diff_points=float((cb.mean() - ca.mean()) * 100), ci_points=[x * 100 for x in ci(d[ix].sum(1) / size[ix].sum(1))])

# in-domain versus cross-dataset accuracy, groups resampled on the Kaggle and Raabin-WBC test sets
cl["domain_gap"] = {}
def acc_draws(name, s):
    run, split = SETS[name]; ok = prob(run, "final", s, split).argmax(1) == Y[name]
    if name in GROUPS:
        g = GROUPS[name]; ix = cluster_draws(g); return ok.mean(), gsum(ok, g)[ix].sum(1) / gsum(np.ones(len(ok)), g)[ix].sum(1)
    return ok.mean(), ok[rng.integers(0, len(ok), (B, len(ok)), dtype=np.int32)].mean(1)
for src, dst in [("PBC->PBC", "PBC->Kaggle"), ("PBC->PBC", "PBC->Raabin"), ("Kaggle->Kaggle", "Kaggle->PBC")]:
    for s in SEEDS:
        m1, d1 = acc_draws(src, s); m2, d2 = acc_draws(dst, s)
        cl["domain_gap"].setdefault(f"{src} minus {dst}", {})[s] = dict(diff_points=float((m1 - m2) * 100), ci_points=[x * 100 for x in ci(d1 - d2)])

# calibration differences on the Kaggle test split, groups resampled
cl["ece_diff"] = {}
def ece_pair_cluster(tag, pa, pb, y, g, s):
    ids, inv = np.unique(g, return_inverse=True); members = [np.where(inv == j)[0] for j in range(len(ids))]
    d = np.empty(B)
    for b_, row in enumerate(cluster_draws(g)):
        i = np.concatenate([members[j] for j in row]); d[b_] = ece(pb[i], y[i]) - ece(pa[i], y[i])
    cl["ece_diff"].setdefault(tag, {})[s] = dict(diff=ece(pb, y) - ece(pa, y), ci=ci(d))
y = Y["Kaggle->Kaggle"]; yv = labels("kaggle", "VAL"); g = GROUPS["Kaggle->Kaggle"]
for s in SEEDS:
    p4, p5, p7, pf = (prob("kaggle", c, s, "TEST") for c in ["M4", "M5", "M7", "final"])
    ece_pair_cluster("Kaggle: M4 -> M5 (focal loss)", p4, p5, y, g, s)
    ece_pair_cluster("Kaggle: M5 -> M7 (six-fold TTA)", p5, p7, y, g, s)
    ece_pair_cluster("Kaggle: final -> final after temperature scaling", pf, scale(pf, fit_T(prob("kaggle", "final", s, "VAL"), yv)), y, g, s)

# macro-F1 difference on Raabin-WBC with and without CLAHE, groups resampled
cl["raabin_f1_diff"] = {}
y = Y["PBC->Raabin"]; g = GROUPS["PBC->Raabin"]; ids, inv = np.unique(g, return_inverse=True); members = [np.where(inv == j)[0] for j in range(len(ids))]
for s in SEEDS:
    ya = prob("raabin", "X_noCLAHE", s, "RAABIN").argmax(1); yb = prob("raabin", "M5", s, "RAABIN").argmax(1); d = np.empty(B)
    for b_, row in enumerate(cluster_draws(g)):
        i = np.concatenate([members[j] for j in row]); d[b_] = macro_f1(y[i], yb[i]) - macro_f1(y[i], ya[i])
    cl["raabin_f1_diff"][s] = dict(diff=macro_f1(y, yb) - macro_f1(y, ya), ci=ci(d))
res["cluster"] = cl

# ---------------------------------------------------------------- 9. frozen reference (M1) on Raabin-WBC against the final pipeline
res["raabin_m1"] = {}
size = gsum(np.ones(len(y)), g)
for s in SEEDS:
    y1 = np.load(f"{root}/predictions/raabin/preds_RAABIN_M1_s{s}.npz")["RAABIN"][:, 0].astype(np.float64).argmax(1)
    yf = prob("raabin", "final", s, "RAABIN").argmax(1)
    rr = mcnemar(y1 == y, yf == y); d = gsum((yf == y).astype(float) - (y1 == y).astype(float), g); ix = cluster_draws(g)
    rr.update(acc_m1=float((y1 == y).mean()), acc_final=float((yf == y).mean()), f1_m1=macro_f1(y, y1), f1_final=macro_f1(y, yf),
              diff_points=float(((yf == y).mean() - (y1 == y).mean()) * 100), ci_group_points=[x * 100 for x in ci(d[ix].sum(1) / size[ix].sum(1))],
              recall_m1={c: float((y1[y == i] == i).mean()) for i, c in enumerate(CLS)})
    res["raabin_m1"][s] = rr
# accuracy of M1 on every test set and macro-F1 on Raabin-WBC for every configuration (mean and standard deviation over seeds)
def msd(v): return [float(np.mean(v)), float(np.std(v, ddof=1))]
res["m1_accuracy"] = {}
for name, (run, split) in SETS.items():
    if name == "PBC->Raabin": continue
    res["m1_accuracy"][name] = msd([(np.load(f"{root}/predictions/{run}/preds_M1_s{s}.npz")[split][:, 0].astype(np.float64).argmax(1) == Y[name]).mean() for s in SEEDS])
res["m1_accuracy"]["PBC->Raabin"] = msd([res["raabin_m1"][s]["acc_m1"] for s in SEEDS])
res["raabin_f1"] = {c: msd([macro_f1(y, prob("raabin", c, s, "RAABIN").argmax(1)) for s in SEEDS]) for c in ["M5", "M8", "X_noCLAHE", "final", "B_resnet50"]}
res["raabin_f1"]["M1"] = msd([res["raabin_m1"][s]["f1_m1"] for s in SEEDS])
res["raabin_majority_class_accuracy"] = float(np.bincount(y).max() / len(y))

# ---------------------------------------------------------------- 10. few-shot: adapted network against the mean of M8 trained on the full split
full = [float((prob("kaggle", "M8", s, "TEST").argmax(1) == Y["Kaggle->Kaggle"]).mean()) for s in SEEDS]
res["fewshot_vs_full_mean"] = dict(full_mean=float(np.mean(full)),
    diff_points=[float((res["fewshot"]["50"][s]["pbc_start"] - np.mean(full)) * 100) for s in SEEDS],
    all_pairings_points=[float(min((res["fewshot"]["50"][a]["pbc_start"] - f) * 100 for a in SEEDS for f in full)),
                         float(max((res["fewshot"]["50"][a]["pbc_start"] - f) * 100 for a in SEEDS for f in full))])

json.dump(res, open(out, "w"), indent=1)
print("written", out)
