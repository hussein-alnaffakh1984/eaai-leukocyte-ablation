# %% [markdown]
# # EAAI-26-12471 — Step 5: Ablation benchmark on PBC (primary dataset) + cross-dataset test on Kaggle
# **Inputs:** `unclesamulus/blood-cells-image-dataset` (PBC, primary) and `paultimothymooney/blood-cells` (Kaggle, external).
# **Accelerator:** GPU. **Run with:** Save Version -> Save & Run All (Commit). Re-run safe (finished runs are skipped).
#
# Why PBC is primary: it contains real, non-augmented single-cell images from a clinical laboratory, so images are
# independent observations. The Kaggle "Blood Cell Images" set is ~410 originals expanded by augmentation whose
# copy-to-original mapping cannot be recovered from file names; it is therefore used only as an external test.
# Protocol: exact-duplicate removal, stratified 70/15/15 train/validation/test split with a fixed seed, all model
# selection on validation, test touched once; 3 training seeds; predictions stored for all 8 D4 transforms.

# %%
import os, re, glob, json, time, zipfile, hashlib
import numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf, keras
from keras import layers
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest

E = os.environ.get
ROOT    = E("BM_ROOT", "/kaggle/input")
OUT     = E("BM_OUT", "/kaggle/working")
WEIGHTS = None if E("BM_WEIGHTS", "imagenet") == "None" else E("BM_WEIGHTS", "imagenet")
SEEDS   = [int(s) for s in E("BM_SEEDS", "0,1,2").split(",")]
EP1     = int(E("BM_EP1", 10))
EP2     = int(E("BM_EP2", 20))
BS      = int(E("BM_BS", 32))
PBS     = int(E("BM_PBS", 64))
IMG     = 224
N_UNFREEZE = 30
SPLIT_SEED = 42
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
os.makedirs(OUT, exist_ok=True)
print("TF", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"), "| seeds:", SEEDS)

# %% [markdown]
# ## 1. Index PBC (primary) and Kaggle (external) images

# %%
allimg = [p for p in glob.glob(f"{ROOT}/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
pbc_rows, k_rows, seen_k = [], [], set()
for p in allimg:
    parts = p.split(os.sep)
    if "dataset2-master" in p:
        if parts[-3] in ("TEST", "TEST_SIMPLE") and parts[-2] in CLASSES and (parts[-3], parts[-2], parts[-1]) not in seen_k:
            seen_k.add((parts[-3], parts[-2], parts[-1]))
            k_rows.append(dict(path=p, split="KTEST" if parts[-3] == "TEST" else "KSIMPLE", cls=parts[-2], file=parts[-1]))
    elif "dataset-master" not in p and parts[-2].upper() in CLASSES:
        pbc_rows.append(dict(path=p, cls=parts[-2].upper(), file=parts[-1]))
pbc = pd.DataFrame(pbc_rows).sort_values("path").reset_index(drop=True)
pbc["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in pbc.path]
n0 = len(pbc); pbc = pbc[~pbc.md5.duplicated()].reset_index(drop=True)
print(f"PBC images: {n0}, exact duplicates removed: {n0 - len(pbc)}")
print(pbc.cls.value_counts().reindex(CLASSES).to_string())
assert len(pbc) > 0, "PBC not found - attach unclesamulus/blood-cells-image-dataset"

tr, rest = train_test_split(pbc, test_size=0.30, stratify=pbc.cls, random_state=SPLIT_SEED)
va, te = train_test_split(rest, test_size=0.50, stratify=rest.cls, random_state=SPLIT_SEED)
kag = pd.DataFrame(k_rows, columns=["path", "split", "cls", "file"]).sort_values("path").reset_index(drop=True)
SPLITS = {"TRAIN": tr.reset_index(drop=True), "VAL": va.reset_index(drop=True), "TEST": te.reset_index(drop=True)}
for s in ["KTEST", "KSIMPLE"]:
    if (kag.split == s).any(): SPLITS[s] = kag[kag.split == s].reset_index(drop=True)
for s, d in SPLITS.items():
    d["y"] = d.cls.map(CLASSES.index)
    d.drop(columns=["path"]).to_csv(f"{OUT}/meta_{s}.csv", index=False)
print({s: d.cls.value_counts().reindex(CLASSES).tolist() for s, d in SPLITS.items()})

# %% [markdown]
# ## 2. Load images (raw and luminance-CLAHE), 224x224 RGB

# %%
def load(p, clahe):
    im = cv2.imread(p, cv2.IMREAD_COLOR)
    if clahe:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        im = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (IMG, IMG), interpolation=cv2.INTER_AREA)

def load_all(paths, clahe):
    with ThreadPoolExecutor(8) as ex:
        return np.stack(list(ex.map(lambda p: load(p, clahe), paths))).astype(np.uint8)

t0 = time.time()
X = {(v, s): load_all(d.path.tolist(), v == "clahe") for v in ["raw", "clahe"] for s, d in SPLITS.items()}
Y = {s: d.y.values for s, d in SPLITS.items()}
print("Loaded in %.0fs:" % (time.time() - t0), {k: v.shape for k, v in X.items()})

# %% [markdown]
# ## 3. Model, losses, data pipeline

# %%
class RandomBrightnessMul(layers.Layer):
    def __init__(self, lo=0.85, hi=1.15, **kw):
        super().__init__(**kw); self.lo, self.hi = lo, hi
    def call(self, x, training=None):
        if not training:
            return x
        f = tf.random.uniform((tf.shape(x)[0], 1, 1, 1), self.lo, self.hi)
        return tf.clip_by_value(x * f, 0.0, 255.0)

class StainJitter(layers.Layer):
    """Random hue/saturation/value perturbation (simulates staining and scanner colour differences)."""
    def __init__(self, h=0.04, s=0.25, v=0.15, **kw):
        super().__init__(**kw); self.h, self.s, self.v = h, s, v
    def call(self, x, training=None):
        if not training:
            return x
        hsv = tf.image.rgb_to_hsv(tf.clip_by_value(x, 0.0, 255.0) / 255.0); b = tf.shape(x)[0]
        u = lambda a: tf.random.uniform((b, 1, 1), -a, a)
        hh = tf.math.floormod(hsv[..., 0] + u(self.h), 1.0)
        ss = tf.clip_by_value(hsv[..., 1] * (1 + u(self.s)), 0.0, 1.0)
        vv = tf.clip_by_value(hsv[..., 2] * (1 + u(self.v)), 0.0, 1.0)
        return tf.image.hsv_to_rgb(tf.stack([hh, ss, vv], -1)) * 255.0

def build(backbone, seed, bn_train=False, strong_aug=False):
    inp = keras.Input((IMG, IMG, 3))
    if strong_aug:   # full rotation invariance + stain/colour jitter
        x = layers.RandomRotation(0.5, fill_mode="constant", seed=seed)(inp)
        x = layers.RandomZoom(0.2, fill_mode="constant", seed=seed + 1)(x)
        x = layers.RandomTranslation(0.12, 0.12, fill_mode="constant", seed=seed + 2)(x)
        x = layers.RandomFlip("horizontal_and_vertical", seed=seed + 3)(x)
        x = RandomBrightnessMul()(x)
        x = StainJitter()(x)
    else:            # augmentation as in the original manuscript
        x = layers.RandomRotation(20 / 360, fill_mode="constant", seed=seed)(inp)
        x = layers.RandomZoom(0.15, fill_mode="constant", seed=seed + 1)(x)
        x = layers.RandomTranslation(0.08, 0.08, fill_mode="constant", seed=seed + 2)(x)
        x = layers.RandomFlip("horizontal", seed=seed + 3)(x)
        x = RandomBrightnessMul()(x)
    if backbone == "mobilenetv2":
        x = layers.Rescaling(1 / 127.5, offset=-1)(x)
        base = keras.applications.MobileNetV2(include_top=False, weights=WEIGHTS, input_shape=(IMG, IMG, 3))
    elif backbone == "resnet50":
        x = layers.Lambda(keras.applications.resnet50.preprocess_input)(x)
        base = keras.applications.ResNet50(include_top=False, weights=WEIGHTS, input_shape=(IMG, IMG, 3))
    elif backbone == "densenet121":
        x = layers.Lambda(keras.applications.densenet.preprocess_input)(x)
        base = keras.applications.DenseNet121(include_top=False, weights=WEIGHTS, input_shape=(IMG, IMG, 3))
    else:  # EfficientNetB0 includes its own input rescaling/normalisation
        base = keras.applications.EfficientNetB0(include_top=False, weights=WEIGHTS, input_shape=(IMG, IMG, 3))
    base.trainable = False
    # default: BatchNorm kept in inference mode; bn_train=True reproduces the common Keras-tutorial behaviour
    feat = base(x) if bn_train else base(x, training=False)
    h = layers.GlobalAveragePooling2D()(feat)
    h = layers.Dense(256, activation="relu")(h)
    h = layers.Dropout(0.4, seed=seed + 4)(h)
    out = layers.Dense(len(CLASSES), activation="softmax")(h)
    return keras.Model(inp, out), keras.Model(inp, [feat, out]), base

def focal_loss(y, p):                        # FL = -alpha (1-p_t)^gamma log p_t, alpha=0.25, gamma=2
    p = tf.clip_by_value(p, 1e-7, 1 - 1e-7)
    return tf.reduce_sum(-0.25 * y * tf.pow(1 - p, 2.0) * tf.math.log(p), axis=-1)

def dataset(Xa, ya, shuffle, seed, sample_p=None):
    """sample_p: optional per-image sampling probabilities (source-cell balanced sampling, with replacement)."""
    state = {"epoch": 0}
    def gen():
        idx = np.arange(len(Xa))
        if shuffle:
            rng = np.random.default_rng(seed * 1000 + state["epoch"]); state["epoch"] += 1
            if sample_p is None: rng.shuffle(idx)
            else: idx = rng.choice(len(Xa), size=len(Xa), replace=True, p=sample_p)
        for i in range(0, len(idx), BS):
            b = idx[i:i + BS]
            yield Xa[b].astype(np.float32), np.eye(len(CLASSES), dtype=np.float32)[ya[b]]
    sig = (tf.TensorSpec((None, IMG, IMG, 3), tf.float32), tf.TensorSpec((None, len(CLASSES)), tf.float32))
    n_batches = int(np.ceil(len(Xa) / BS))
    return (tf.data.Dataset.from_generator(gen, output_signature=sig)
            .apply(tf.data.experimental.assert_cardinality(n_batches)).prefetch(2))

# D4 transforms: I, R90, R180, R270, H(flip), V(flip), T(transpose), AT(anti-transpose)
D4 = ["I", "R90", "R180", "R270", "H", "V", "T", "AT"]
def d4(x, name):
    return {"I": x, "R90": np.rot90(x, 1, (1, 2)), "R180": np.rot90(x, 2, (1, 2)), "R270": np.rot90(x, 3, (1, 2)),
            "H": x[:, :, ::-1], "V": x[:, ::-1], "T": np.transpose(x, (0, 2, 1, 3)),
            "AT": np.rot90(np.transpose(x, (0, 2, 1, 3)), 2, (1, 2))}[name]
TTA = {"none": ["I"], "4-fold": ["I", "H", "V", "R180"], "6-fold": ["I", "H", "V", "R90", "R180", "R270"], "8-fold (D4)": D4}

def predict_d4(model, Xa, chunk=512):
    out = np.zeros((len(Xa), len(D4), len(CLASSES)), np.float32)
    for i in range(0, len(Xa), chunk):
        xb = Xa[i:i + chunk]
        for j, n in enumerate(D4):
            out[i:i + chunk, j] = model.predict(np.ascontiguousarray(d4(xb, n)).astype(np.float32), batch_size=PBS, verbose=0)
    return out

# %% [markdown]
# ## 4. Ablation stages (same definitions as the manuscript; M6/M7 = M5 with 4-/6-fold TTA)

# %%
STAGES = [
    dict(name="M1", clahe=False, backbone="mobilenetv2",    finetune=False, loss="ce"),
    dict(name="M2", clahe=True,  backbone="mobilenetv2",    finetune=False, loss="ce"),
    dict(name="M3", clahe=True,  backbone="mobilenetv2",    finetune=True,  loss="ce"),
    dict(name="M4", clahe=True,  backbone="efficientnetb0", finetune=True,  loss="ce"),
    dict(name="M5", clahe=True,  backbone="efficientnetb0", finetune=True,  loss="focal"),
    dict(name="M8", clahe=True,  backbone="efficientnetb0", finetune=True,  loss="focal", balanced=True),  # R2: class imbalance
    dict(name="X_noCLAHE", clahe=False, backbone="efficientnetb0", finetune=True, loss="focal"),       # single-factor control
    dict(name="B_resnet50", clahe=True, backbone="resnet50", finetune=True, loss="focal"),              # R4: backbone comparison
]
ONLY = [s for s in E("BM_ONLY", "").split(",") if s]
if ONLY: STAGES = [s for s in STAGES if s["name"] in ONLY]
TTA_STAGES = ("M5", "M8")

def gpu_peak_mb():
    try: return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    except Exception: return float("nan")
def gpu_reset():
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass

def latency(model, Xa, n=256):
    xb = Xa[:n]; res = {}
    model.predict(xb[:32].astype(np.float32), verbose=0)
    for k, names in TTA.items():
        t0 = time.time()
        for nm in names:
            model.predict(np.ascontiguousarray(d4(xb, nm)).astype(np.float32), batch_size=PBS, verbose=0)
        res[k] = 1000 * (time.time() - t0) / len(xb)
    return res

def run(stage, seed):
    tag = f"{stage['name']}_s{seed}"
    if os.path.exists(f"{OUT}/preds_{tag}.npz"):
        print("skip", tag); return
    keras.utils.set_random_seed(seed); keras.backend.clear_session(); gpu_reset()
    v = "clahe" if stage["clahe"] else "raw"
    model, gc_model, base = build(stage["backbone"], seed)
    sp = None
    if stage.get("balanced"):                      # each class equally likely per draw
        cnt = np.bincount(Y["TRAIN"], minlength=len(CLASSES)); sp = 1.0 / cnt[Y["TRAIN"]]; sp = sp / sp.sum()
    loss = focal_loss if stage["loss"] == "focal" else keras.losses.CategoricalCrossentropy()
    dtr = dataset(X[(v, "TRAIN")], Y["TRAIN"], True, seed, sp)
    dva = dataset(X[(v, "VAL")], Y["VAL"], False, seed)
    hist, t0 = {}, time.time()
    model.compile(keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    hist["phase1"] = model.fit(dtr, validation_data=dva, epochs=EP1, verbose=2).history
    if stage["finetune"]:
        base.trainable = True
        for l in base.layers[:-N_UNFREEZE]: l.trainable = False
        for l in base.layers[-N_UNFREEZE:]:
            if isinstance(l, layers.BatchNormalization): l.trainable = False
        model.compile(keras.optimizers.Adam(1e-5), loss=loss, metrics=["accuracy"])
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]
        hist["phase2"] = model.fit(dtr, validation_data=dva, epochs=EP2, callbacks=cb, verbose=2).history
    train_s = time.time() - t0
    info = dict(stage=stage["name"], seed=seed, train_seconds=train_s, peak_gpu_mb_train=gpu_peak_mb(),
                params_total=int(model.count_params()),
                params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                epochs_phase2=len(hist.get("phase2", {}).get("loss", [])),
                latency_ms_per_image=latency(model, X[(v, "TEST")]))
    preds = {s: predict_d4(model, X[(v, s)]) for s in SPLITS if s != "TRAIN"}
    np.savez_compressed(f"{OUT}/preds_{tag}.npz", **preds)
    model.save_weights(f"{OUT}/weights_{tag}.weights.h5")
    json.dump(dict(info=info, history={k: {kk: [float(x) for x in vv] for kk, vv in d.items()} for k, d in hist.items()}),
              open(f"{OUT}/log_{tag}.json", "w"), indent=1)
    print(tag, "done in %.0fs | PBC test acc (no TTA) = %.4f" % (train_s, (preds["TEST"][:, 0].argmax(1) == Y["TEST"]).mean()))
    if stage["name"] in TTA_STAGES and seed == SEEDS[0]:
        gradcam_figure(gc_model, X[(v, "TEST")], Y["TEST"], preds["TEST"][:, 0], f"PBC_{stage['name']}")
        if "KTEST" in SPLITS: gradcam_figure(gc_model, X[(v, "KTEST")], Y["KTEST"], preds["KTEST"][:, 0], f"KAGGLE_{stage['name']}")

# %% [markdown]
# ## 5. Grad-CAM (final model, first seed)

# %%
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def gradcam(gc_model, img):
    x = tf.convert_to_tensor(img[None].astype(np.float32))
    with tf.GradientTape() as tape:
        conv, pred = gc_model(x, training=False)
        c = int(tf.argmax(pred[0])); score = pred[:, c]
    g = tape.gradient(score, conv)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    cam = cv2.resize(cam, (IMG, IMG)); cam = cam / (cam.max() + 1e-8)
    return cam, c

def gradcam_figure(gc_model, Xa, ya, p, name, per_class=2):
    pred = p.argmax(1); rng = np.random.default_rng(0); picks = []
    for c in range(len(CLASSES)):
        ok = np.where((ya == c) & (pred == c))[0]; bad = np.where((ya == c) & (pred != c))[0]
        picks += list(rng.choice(ok, min(per_class, len(ok)), replace=False)) if len(ok) else []
        picks += list(rng.choice(bad, 1, replace=False)) if len(bad) else []
    if not picks: return
    fig, ax = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.8), squeeze=False)
    for j, i in enumerate(picks):
        cam, c = gradcam(gc_model, Xa[i])
        ax[0, j].imshow(Xa[i]); ax[1, j].imshow(Xa[i]); ax[1, j].imshow(cam, cmap="jet", alpha=0.4)
        ax[0, j].set_title(f"T:{CLASSES[ya[i]][:4]} P:{CLASSES[c][:4]}", fontsize=8, color="green" if c == ya[i] else "red")
        ax[0, j].axis("off"); ax[1, j].axis("off")
    fig.suptitle(f"Grad-CAM, {name}"); fig.tight_layout()
    fig.savefig(f"{OUT}/gradcam_{name}.png", dpi=130); plt.close(fig)

# %% [markdown]
# ## 6. Train all stages x seeds

# %%
for seed in SEEDS:
    for st in STAGES:
        run(st, seed)

# %% [markdown]
# ## 7. Summary: accuracy, F1, calibration (before/after TTA and temperature scaling), cross-dataset, McNemar

# %%
def ece(p, y, bins=15):
    conf, ok = p.max(1), (p.argmax(1) == y)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

def brier(p, y):
    return float(np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1)))

def fit_temperature(p_val, y_val):
    """Post-hoc temperature scaling fitted on the validation set (NLL grid search)."""
    lg = np.log(np.clip(p_val, 1e-12, 1)); best = (1.0, np.inf)
    for T in np.arange(0.25, 5.01, 0.05):
        z = lg / T; z = z - z.max(1, keepdims=True); q = np.exp(z); q /= q.sum(1, keepdims=True)
        nll = -np.mean(np.log(np.clip(q[np.arange(len(y_val)), y_val], 1e-12, 1)))
        if nll < best[1]: best = (float(T), nll)
    return best[0]

def apply_temperature(p, T):
    z = np.log(np.clip(p, 1e-12, 1)) / T; z = z - z.max(1, keepdims=True); q = np.exp(z)
    return q / q.sum(1, keepdims=True)

def metrics(p, y):
    return dict(acc=float((p.argmax(1) == y).mean()), macro_f1=float(f1_score(y, p.argmax(1), average="macro")),
                ece=float(ece(p, y)), brier=brier(p, y))


def label_of(stage, tname):
    if stage == "M5": return {"none": "M5", "4-fold": "M6", "6-fold": "M7", "8-fold (D4)": "M7_D4"}[tname]
    return stage if tname == "none" else f"{stage}_{tname.split()[0]}"

PRED = {}
rows = []
for seed in SEEDS:
    for st in STAGES:
        f = f"{OUT}/preds_{st['name']}_s{seed}.npz"
        if not os.path.exists(f): continue
        P = np.load(f)
        for tname, names in (TTA.items() if st["name"] in TTA_STAGES else [("none", ["I"])]):
            sel = [D4.index(n) for n in names]; lab = label_of(st["name"], tname)
            pt = P["TEST"][:, sel].mean(1); PRED[(lab, seed)] = pt
            yt = Y["TEST"]; yhat = pt.argmax(1)
            r = dict(stage=lab, seed=seed, tta=tname, **{f"test_{k}": v for k, v in metrics(pt, yt).items()})
            r["test_bal_acc"] = float(balanced_accuracy_score(yt, yhat))
            r["test_macro_auc"] = float(roc_auc_score(np.eye(len(CLASSES))[yt], pt, average="macro"))
            for c, rec in zip(CLASSES, recall_score(yt, yhat, average=None, labels=range(len(CLASSES)))):
                r[f"recall_{c[:4].lower()}"] = float(rec)
            T = fit_temperature(P["VAL"][:, sel].mean(1), Y["VAL"]); pts = apply_temperature(pt, T)
            r["temperature"] = T; r["test_ece_ts"] = float(ece(pts, yt)); r["test_brier_ts"] = brier(pts, yt)
            r["val_acc"] = float((P["VAL"][:, sel].mean(1).argmax(1) == Y["VAL"]).mean())
            for s in ["KTEST", "KSIMPLE"]:
                if s in P.files:
                    pk = P[s][:, sel].mean(1)
                    r[f"{s.lower()}_acc"] = float((pk.argmax(1) == Y[s]).mean())
                    r[f"{s.lower()}_macro_f1"] = float(f1_score(Y[s], pk.argmax(1), average="macro"))
            rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results_all_seeds.csv", index=False)
num = [c for c in res.columns if c not in ("stage", "seed", "tta")]
order = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M7_D4", "M8", "M8_4-fold", "M8_6-fold", "M8_8-fold", "X_noCLAHE", "B_resnet50"]
summ = res.groupby("stage")[num].agg(["mean", "std"]).reindex([o for o in order if o in set(res.stage)])
summ.to_csv(f"{OUT}/results_summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("MEAN over seeds\n", summ.xs("mean", axis=1, level=1).round(4).to_string())
print("\nSTD over seeds\n", summ.xs("std", axis=1, level=1).round(4).to_string())
print("\nPER-SEED\n" + res.round(4).to_csv(index=False))

# McNemar (exact binomial on discordant pairs) between consecutive stages, per seed; test images are independent
chain = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
mc = []
for seed in SEEDS:
    for a, b in zip(chain[:-1], chain[1:]):
        if (a, seed) in PRED and (b, seed) in PRED:
            ca = PRED[(a, seed)].argmax(1) == Y["TEST"]; cb = PRED[(b, seed)].argmax(1) == Y["TEST"]
            n01, n10 = int((~ca & cb).sum()), int((ca & ~cb).sum())
            p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            mc.append(dict(seed=seed, pair=f"{a}->{b}", gained=n01, lost=n10, p_value=p))
mc = pd.DataFrame(mc); mc.to_csv(f"{OUT}/mcnemar.csv", index=False)
print("\nMcNemar (exact)\n", mc.to_string(index=False))

for mname in TTA_STAGES:
    f = f"{OUT}/preds_{mname}_s{SEEDS[0]}.npz"
    if not os.path.exists(f): continue
    P = np.load(f)
    for s in [k for k in ["TEST", "KTEST"] if k in P.files]:
        for tname in ["none", "4-fold"]:
            sel = [D4.index(n) for n in TTA[tname]]
            cm = pd.DataFrame(confusion_matrix(Y[s], P[s][:, sel].mean(1).argmax(1), labels=range(len(CLASSES))),
                              index=CLASSES, columns=CLASSES)
            cm.to_csv(f"{OUT}/confusion_{mname}_{tname}_{s}.csv"); print(f"\n{s} | {mname} {tname}\n", cm)

# %% [markdown]
# ## 8. Bundle (send `pbc_bundle.zip` back; weights stay in the output for the code release)

# %%
with zipfile.ZipFile(f"{OUT}/pbc_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".npz", ".csv", ".json", ".png")):
            z.write(f"{OUT}/{f}", f)
print("pbc_bundle.zip: %.1f MB" % (os.path.getsize(f"{OUT}/pbc_bundle.zip") / 2**20))
