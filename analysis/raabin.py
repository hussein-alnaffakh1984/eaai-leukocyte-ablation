"""Accuracy of PBC-trained networks on Raabin-WBC. Usage: python raabin.py <dir with raabin/meta_RAABIN.csv and preds_RAABIN_*.npz> out.json"""
import sys, glob, re, json, numpy as np, pandas as pd
from sklearn.metrics import recall_score, f1_score
d, out = sys.argv[1:3]
y = pd.read_csv(f"{d}/meta_RAABIN.csv").y.values
TTA = {"none": [0], "6": [0, 4, 5, 1, 2, 3]}
res = {"n": int(len(y)), "counts": np.bincount(y, minlength=4).tolist(), "per_seed": {}, "mean": {}, "sd": {}}
for f in sorted(glob.glob(f"{d}/preds_RAABIN_*_s*.npz")):
    stage, s = re.search(r"preds_RAABIN_(.+)_s(\d+)\.npz", f).groups()
    P = np.load(f)["RAABIN"].astype(np.float64)
    for t, ix in TTA.items():
        name = {"M8": {"none": "M8", "6": "final"}}.get(stage, {}).get(t, stage if t == "none" else stage + "_6")
        yh = P[:, ix].mean(1).argmax(1)
        r = dict(acc=float((yh == y).mean()), f1=float(f1_score(y, yh, average="macro")))
        for c, v in zip(["eos", "lym", "mon", "neu"], recall_score(y, yh, average=None, labels=range(4))): r["rec_" + c] = float(v)
        res["per_seed"].setdefault(name, {})[s] = r
for n, ds in res["per_seed"].items():
    ks = next(iter(ds.values())).keys()
    res["mean"][n] = {k: float(np.mean([v[k] for v in ds.values()])) for k in ks}
    res["sd"][n] = {k: (float(np.std([v[k] for v in ds.values()], ddof=1)) if len(ds) > 1 else None) for k in ks}
json.dump(res, open(out, "w"), indent=1)
for n in res["mean"]: print(n, round(res["mean"][n]["acc"] * 100, 1), "±", round((res["sd"][n]["acc"] or 0) * 100, 1))
