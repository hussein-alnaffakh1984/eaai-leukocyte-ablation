"""Compute every number used in the revised manuscript from stored predictions.

Usage: python analyse.py <dataset_tag P|K> <dir with preds_*.npz and meta_*.csv> <out.json>
P = PBC run (splits VAL, TEST, KTEST = Kaggle test as external)
K = Kaggle run (splits VAL, TEST, EXT = PBC as external)
"""
import sys, json, glob, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, roc_auc_score

tag, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
EXT = "KTEST" if tag == "P" else "EXT"
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
TTA = {"none": [0], "4": [0, 4, 5, 2], "6": [0, 4, 5, 1, 2, 3], "8": list(range(8))}
# configuration name -> (trained stage, TTA)
CONFIGS = {"M1": ("M1", "none"), "M2": ("M2", "none"), "M3": ("M3", "none"), "M4": ("M4", "none"),
           "M5": ("M5", "none"), "M6": ("M5", "4"), "M7": ("M5", "6"), "M7_D4": ("M5", "8"),
           "M8": ("M8", "none"), "M8_4": ("M8", "4"), "final": ("M8", "6"), "M8_8": ("M8", "8"),
           "X_noCLAHE": ("X_noCLAHE", "none"), "B_resnet50": ("B_resnet50", "none"),
           "LOO_noCLAHE": ("LOO_noCLAHE", "6"), "LOO_noFT": ("LOO_noFT", "6"), "LOO_mobilenet": ("LOO_mobilenet", "6"),
           "LOO_ce": ("LOO_ce", "6"), "LOO_nobal": ("M5", "6"), "LOO_noTTA": ("M8", "none")}
PAIRS = [("M1", "M2"), ("M2", "M3"), ("M3", "M4"), ("M4", "M5"), ("M5", "M6"), ("M6", "M7"),
         ("M5", "M7"), ("M5", "M8"), ("M7", "final"), ("X_noCLAHE", "M5"), ("B_resnet50", "M5"), ("M5", "final")]
LOO_PAIRS = [("LOO_noCLAHE", "final"), ("LOO_noFT", "final"), ("LOO_mobilenet", "final"), ("LOO_ce", "final"), ("LOO_nobal", "final"), ("LOO_noTTA", "final")]

Y = {s: pd.read_csv(f"{d}/meta_{s}.csv").y.values for s in ["VAL", "TEST", EXT] if os.path.exists(f"{d}/meta_{s}.csv")}
seeds = sorted({int(m.group(1)) for f in glob.glob(f"{d}/preds_M1_s*.npz") for m in [re.search(r"_s(\d+)\.npz", f)]})

def ece(p, y, bins=15):
    conf, ok = p.max(1), p.argmax(1) == y
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)

def brier(p, y): return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, 1)))

def scale(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T; z -= z.max(1, keepdims=True); q = np.exp(z)
    return q / q.sum(1, keepdims=True)

def fit_T(p, y):
    best = (1.0, np.inf)
    for T in np.arange(0.05, 5.0001, 0.01):
        nll = -np.mean(np.log(np.clip(scale(p, T)[np.arange(len(y)), y], 1e-12, 1)))
        if nll < best[1]: best = (round(float(T), 2), nll)
    return best[0]

def reliability(p, y, bins=15):
    conf, ok = p.max(1), p.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append([float(lo), float(hi), int(m.sum()), float(conf[m].mean()) if m.any() else None,
                     float(ok[m].mean()) if m.any() else None])
    return rows

P = {}
for name, (stage, t) in CONFIGS.items():
    for s in seeds:
        f = f"{d}/preds_{stage}_s{s}.npz"
        if not os.path.exists(f): continue
        z = np.load(f)
        P[(name, s)] = {k: z[k][:, TTA[t]].astype(np.float64).mean(1) for k in z.files if k in Y}

res = {"seeds": seeds, "per_seed": {}, "mean": {}, "sd": {}, "mcnemar": {}, "confusion": {}, "reliability": {}}
for (name, s), pr in P.items():
    pt, yt = pr["TEST"], Y["TEST"]; yh = pt.argmax(1)
    T = fit_T(pr["VAL"], Y["VAL"]); pts = scale(pt, T)
    r = dict(acc=float((yh == yt).mean()), f1=float(f1_score(yt, yh, average="macro")),
             bal=float(balanced_accuracy_score(yt, yh)),
             auc=float(roc_auc_score(np.eye(4)[yt], pt, average="macro")),
             ece=ece(pt, yt), brier=brier(pt, yt), T=T, ece_ts=ece(pts, yt), brier_ts=brier(pts, yt),
             val_acc=float((pr["VAL"].argmax(1) == Y["VAL"]).mean()))
    for c, v in zip(CLASSES, recall_score(yt, yh, average=None, labels=range(4))): r[f"rec_{c[:3].lower()}"] = float(v)
    if EXT in pr:
        pe = pr[EXT]; r["ext_acc"] = float((pe.argmax(1) == Y[EXT]).mean())
        r["ext_f1"] = float(f1_score(Y[EXT], pe.argmax(1), average="macro"))
    res["per_seed"].setdefault(name, {})[s] = r
    if s == seeds[0] and name in ("M4", "M5", "M7", "final"):
        res["reliability"][name] = {"raw": reliability(pt, yt), "ts": reliability(pts, yt)}
    if name in ("final", "M1"):
        cm = pd.crosstab(pd.Series(yt, name="t"), pd.Series(yh, name="p")).reindex(index=range(4), columns=range(4), fill_value=0)
        res["confusion"].setdefault(name, {})[s] = cm.values.tolist()
        if EXT in pr:
            cme = pd.crosstab(pd.Series(Y[EXT]), pd.Series(pr[EXT].argmax(1))).reindex(index=range(4), columns=range(4), fill_value=0)
            res["confusion"].setdefault(name + "_ext", {})[s] = cme.values.tolist()

for name, d_ in res["per_seed"].items():
    keys = list(next(iter(d_.values())).keys())
    res["mean"][name] = {k: float(np.mean([d_[s][k] for s in d_])) for k in keys}
    res["sd"][name] = {k: float(np.std([d_[s][k] for s in d_], ddof=1)) if len(d_) > 1 else None for k in keys}

for a, b in PAIRS:
    for s in seeds:
        if (a, s) in P and (b, s) in P:
            ca = P[(a, s)]["TEST"].argmax(1) == Y["TEST"]; cb = P[(b, s)]["TEST"].argmax(1) == Y["TEST"]
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            res["mcnemar"].setdefault(f"{a}->{b}", {})[s] = dict(b_better=n01, a_better=n10, p=float(p))

for a_, b_ in LOO_PAIRS:
    for s in seeds:
        if (a_, s) in P and (b_, s) in P:
            ca = P[(a_, s)]["TEST"].argmax(1) == Y["TEST"]; cb = P[(b_, s)]["TEST"].argmax(1) == Y["TEST"]
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            res.setdefault("mcnemar_loo", {}).setdefault(f"{a_}->{b_}", {})[s] = dict(b_better=n01, a_better=n10, p=float(p))
for k_, d_ in res.get("mcnemar_loo", {}).items():
    pass
# Holm correction across the planned comparisons, separately for each seed
for s in seeds:
    keys = [k for k in res["mcnemar"] if s in res["mcnemar"][k]]
    ps = sorted((res["mcnemar"][k][s]["p"], k) for k in keys); m_ = len(ps); run_max = 0.0
    for i, (p, k) in enumerate(ps):
        run_max = max(run_max, min(1.0, (m_ - i) * p)); res["mcnemar"][k][s]["p_holm"] = run_max

# compute cost from logs
cost = {}
for f in glob.glob(f"{d}/log_*_s*.json"):
    i = json.load(open(f))["info"]
    cost.setdefault(i["stage"], []).append(i)
def gpu_runs(v):
    """Drop runs that fell back to CPU (no-TTA latency > 3x the fastest run of the same stage)."""
    lo = min(x["latency_ms_per_image"]["none"] for x in v)
    return [x for x in v if x["latency_ms_per_image"]["none"] <= 3 * lo]
res["cost"] = {}
for k, v in cost.items():
    g = gpu_runs(v)
    res["cost"][k] = dict(params=v[0]["params_total"], trainable=v[0]["params_trainable"],
                          train_min=float(np.median([x["train_seconds"] for x in g])) / 60,
                          lat={t: float(np.median([x["latency_ms_per_image"][t] for x in g])) for t in g[0]["latency_ms_per_image"]},
                          n_runs=len(g), n_excluded=len(v) - len(g), mem_mb=[x.get("peak_gpu_mb_train") for x in g])
res["n"] = {k: int(len(v)) for k, v in Y.items()}
if "mcnemar_loo" in res:                                  # Holm within each seed across the leave-one-out family
    for s in seeds:
        ks = [k for k in res["mcnemar_loo"] if s in res["mcnemar_loo"][k]]
        ps = sorted((res["mcnemar_loo"][k][s]["p"], k) for k in ks); run_max = 0.0
        for i, (p, k) in enumerate(ps):
            run_max = max(run_max, min(1.0, (len(ps) - i) * p)); res["mcnemar_loo"][k][s]["p_holm"] = run_max
json.dump(res, open(out, "w"), indent=1)
print("seeds", seeds, "configs", sorted(res["mean"]))
for k in ["M1", "M4", "M5", "M7", "final", "X_noCLAHE", "B_resnet50"]:
    if k in res["mean"]:
        m = res["mean"][k]; print(k, round(m["acc"] * 100, 2), "ext", round(m.get("ext_acc", np.nan) * 100, 2), "ece", round(m["ece"], 3), "T", m["T"])
