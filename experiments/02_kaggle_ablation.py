# %% [markdown]
# # EAAI-26-12471 — Step 2: Full re-implemented ablation benchmark
# **Inputs to attach:** `paultimothymooney/blood-cells` (main) and `unclesamulus/blood-cells-image-dataset` (external test, PBC).
# **Accelerator:** GPU. **Run with:** *Save Version → Save & Run All (Commit)* so it runs in the background (several hours for 3 seeds).
# Re-run safe: finished (stage, seed) runs are skipped if their prediction file already exists.
#
# Design (fixes found in the audit):
# * validation split is **grouped by source cell** (file name `_A_B.jpeg`, A = source cell index within split/class) — no augmented copy of a validation cell is ever trained on;
# * predictions are stored for all 8 dihedral (D4) transforms, so any test-time augmentation subset, calibration, temperature scaling and cluster (per-source-cell) statistics can be recomputed offline;
# * the external PBC dataset (different laboratory/scanner) is evaluated with the same four classes.

# %%
import os, re, glob, json, time, zipfile
import numpy as np, pandas as pd, cv2
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf, keras
from keras import layers
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

E = os.environ.get
ROOT    = E("BM_ROOT", "/kaggle/input")
OUT     = E("BM_OUT", "/kaggle/working")
WEIGHTS = None if E("BM_WEIGHTS", "imagenet") == "None" else E("BM_WEIGHTS", "imagenet")
SEEDS   = [int(s) for s in E("BM_SEEDS", "0,1,2").split(",")]
EP1     = int(E("BM_EP1", 10))     # phase 1 (head only), as in the manuscript
EP2     = int(E("BM_EP2", 20))     # phase 2 (fine-tuning), max epochs
BS      = int(E("BM_BS", 32))
PBS     = int(E("BM_PBS", 64))   # prediction batch size
IMG     = 224
N_UNFREEZE = 30
SPLIT_SEED = 42                    # validation split fixed across seeds/stages
CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
os.makedirs(OUT, exist_ok=True)
print("TF", tf.__version__, "| Keras", keras.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))
print("Seeds:", SEEDS, "| EP1/EP2:", EP1, EP2, "| BS:", BS)

# %% [markdown]
# ## 1. Index files

# %%
allimg = [p for p in glob.glob(f"{ROOT}/**/*", recursive=True) if p.lower().endswith((".jpg", ".jpeg", ".png"))]
rows = []
for p in allimg:
    parts = p.split(os.sep)
    if "dataset2-master" in p and parts[-3] in ("TRAIN", "TEST", "TEST_SIMPLE") and parts[-2] in CLASSES:
        m = re.match(r"^_?(\d+)_(\d+)", parts[-1])
        rows.append(dict(path=p, split=parts[-3], cls=parts[-2],
                         source=f"{parts[-3]}|{parts[-2]}|{m.group(1) if m else parts[-1]}"))
meta = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
meta["y"] = meta.cls.map(CLASSES.index)
print(meta.groupby(["split", "cls"]).agg(images=("path", "size"), source_cells=("source", "nunique")).to_string())

# grouped train/validation split, stratified by class (20% of source cells per class)
tr_all = meta[meta.split == "TRAIN"]
val_src = []
for c, d in tr_all.groupby("cls"):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SPLIT_SEED)
    _, vi = next(gss.split(d, groups=d.source))
    val_src += list(d.iloc[vi].source.unique())
meta.loc[(meta.split == "TRAIN") & meta.source.isin(val_src), "split"] = "VAL"
assert not set(meta[meta.split == "TRAIN"].source) & set(meta[meta.split == "VAL"].source)
print("\nAfter grouped split:\n", meta.groupby(["split", "cls"]).agg(images=("path", "size"), cells=("source", "nunique")).to_string())

# external dataset (PBC): any folder named like one of the 4 classes outside the Mooney dataset
ext_rows, seen = [], set()
for p in allimg:
    if "dataset2-master" in p or "dataset-master" in p:
        continue
    c = os.path.basename(os.path.dirname(p)).upper()
    if c in CLASSES and os.path.basename(p) not in seen:
        seen.add(os.path.basename(p)); ext_rows.append(dict(path=p, split="EXT", cls=c, source=f"EXT|{os.path.basename(p)}"))
ext = pd.DataFrame(ext_rows, columns=["path", "split", "cls", "source"]).sort_values("path").reset_index(drop=True)
ext["y"] = ext.cls.map(CLASSES.index)
print("\nExternal (PBC) images per class:", ext.cls.value_counts().to_dict() if len(ext) else "NOT FOUND - external test skipped")

for s, d in list(meta.groupby("split")) + ([("EXT", ext)] if len(ext) else []):
    d.drop(columns=["path"]).assign(file=d.path.map(os.path.basename)).to_csv(f"{OUT}/meta_{s}.csv", index=False)

# %% [markdown]
# ## 2. Load images (raw and luminance-CLAHE versions), 224×224 RGB

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
SPLITS = {s: meta[meta.split == s].reset_index(drop=True) for s in ["TRAIN", "VAL", "TEST", "TEST_SIMPLE"]}
if len(ext): SPLITS["EXT"] = ext
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
# ## 4. Ablation stages
# M1–M5 are trained; M6/M7 are M5 with 4-/6-fold TTA (computed from stored D4 predictions).
# X_noCLAHE is an extra single-factor control: the final configuration without CLAHE.

# %%
STAGES = [
    dict(name="M1", clahe=False, backbone="mobilenetv2",   finetune=False, loss="ce"),
    dict(name="M2", clahe=True,  backbone="mobilenetv2",   finetune=False, loss="ce"),
    dict(name="M3", clahe=True,  backbone="mobilenetv2",   finetune=True,  loss="ce"),
    dict(name="M4", clahe=True,  backbone="efficientnetb0", finetune=True, loss="ce"),
    dict(name="M5", clahe=True,  backbone="efficientnetb0", finetune=True, loss="focal"),
    dict(name="X_noCLAHE", clahe=False, backbone="efficientnetb0", finetune=True, loss="focal"),
    # "paper-faithful" variant of M5: BGR input, BatchNorm unfrozen in phase 2, random image-level validation split
    dict(name="P_faithful", clahe=True, backbone="efficientnetb0", finetune=True, loss="focal",
         bgr=True, bn_train=True, val_split="random"),
    # reviewer-driven extensions of the ablation (selected on the grouped validation set, test untouched)
    dict(name="M8", clahe=True, backbone="efficientnetb0", finetune=True, loss="focal", balanced=True),    # R2: long-tailed source cells
    dict(name="M9", clahe=True, backbone="efficientnetb0", finetune=True, loss="focal", balanced=True,
         strong_aug=True),                                                                               # R2: domain shift (stain/rotation)
    # same-protocol backbone comparison (R4), M5 configuration
    dict(name="B_resnet50", clahe=True, backbone="resnet50", finetune=True, loss="focal"),
    dict(name="B_densenet121", clahe=True, backbone="densenet121", finetune=True, loss="focal"),
]
ONLY = [s for s in E("BM_ONLY", "").split(",") if s]
if ONLY: STAGES = [s for s in STAGES if s["name"] in ONLY]

def gpu_peak_mb():
    try: return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20
    except Exception: return float("nan")
def gpu_reset():
    try: tf.config.experimental.reset_memory_stats("GPU:0")
    except Exception: pass

def latency(model, Xa, n=256):
    xb = Xa[:n]; res = {}
    model.predict(xb[:32].astype(np.float32), verbose=0)          # warm-up
    for k, names in TTA.items():
        t0 = time.time()
        for nm in names:
            model.predict(np.ascontiguousarray(d4(xb, nm)).astype(np.float32), batch_size=PBS, verbose=0)
        res[k] = 1000 * (time.time() - t0) / len(xb)
    return res  # ms per image

def run(stage, seed):
    tag = f"{stage['name']}_s{seed}"
    if os.path.exists(f"{OUT}/preds_{tag}.npz"):
        print("skip", tag); return
    keras.utils.set_random_seed(seed); keras.backend.clear_session(); gpu_reset()
    v = "clahe" if stage["clahe"] else "raw"
    bgr, bn_train = stage.get("bgr", False), stage.get("bn_train", False)
    ch = (lambda a: a[..., ::-1]) if bgr else (lambda a: a)
    model, gc_model, base = build(stage["backbone"], seed, bn_train, stage.get("strong_aug", False))
    sp = None
    if stage.get("balanced"):   # each class equally likely, each source cell equally likely within its class
        d = SPLITS["TRAIN"]; n_img = d.groupby("source").path.transform("size").values
        n_src = d.groupby("cls").source.transform("nunique").values
        sp = 1.0 / (len(CLASSES) * n_src * n_img); sp = sp / sp.sum()
    loss = focal_loss if stage["loss"] == "focal" else keras.losses.CategoricalCrossentropy()
    if stage.get("val_split") == "random":       # image-level random split of the whole TRAIN folder (leaky, as in v1)
        Xa = np.concatenate([X[(v, "TRAIN")], X[(v, "VAL")]]); ya = np.concatenate([Y["TRAIN"], Y["VAL"]])
        perm = np.random.default_rng(SPLIT_SEED).permutation(len(Xa)); nv = int(0.2 * len(Xa))
        dtr = dataset(ch(Xa[perm[nv:]]), ya[perm[nv:]], True, seed); dva = dataset(ch(Xa[perm[:nv]]), ya[perm[:nv]], False, seed)
    else:
        dtr, dva = dataset(ch(X[(v, "TRAIN")]), Y["TRAIN"], True, seed, sp), dataset(ch(X[(v, "VAL")]), Y["VAL"], False, seed)
    hist, t0 = {}, time.time()
    model.compile(keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
    h1 = model.fit(dtr, validation_data=dva, epochs=EP1, verbose=2)
    hist["phase1"] = h1.history
    if stage["finetune"]:
        base.trainable = True
        for l in base.layers[:-N_UNFREEZE]: l.trainable = False
        for l in base.layers[-N_UNFREEZE:]:
            if isinstance(l, layers.BatchNormalization) and not bn_train: l.trainable = False
        model.compile(keras.optimizers.Adam(1e-5), loss=loss, metrics=["accuracy"])
        cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
              keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]
        h2 = model.fit(dtr, validation_data=dva, epochs=EP2, callbacks=cb, verbose=2)
        hist["phase2"] = {k: [float(x) for x in vals] for k, vals in h2.history.items()}
    train_s = time.time() - t0
    info = dict(stage=stage["name"], seed=seed, train_seconds=train_s, peak_gpu_mb_train=gpu_peak_mb(),
                params_total=int(model.count_params()),
                params_trainable=int(sum(np.prod(w.shape) for w in model.trainable_weights)),
                backbone_layers=len(base.layers), latency_ms_per_image=latency(model, ch(X[(v, "TEST")])))
    preds = {s: predict_d4(model, ch(X[(v, s)])) for s in SPLITS if s != "TRAIN"}
    np.savez_compressed(f"{OUT}/preds_{tag}.npz", **preds)
    model.save_weights(f"{OUT}/weights_{tag}.weights.h5")
    json.dump(dict(info=info, history={k: {kk: [float(x) for x in vv] for kk, vv in d.items()} for k, d in hist.items()}),
              open(f"{OUT}/log_{tag}.json", "w"), indent=1)
    print(tag, "done in %.0fs | test acc (no TTA) = %.4f" % (train_s, (preds["TEST"][:, 0].argmax(1) == Y["TEST"]).mean()))
    if stage["name"] in ("M5", "M9") and seed == SEEDS[0]:
        gradcam_figure(gc_model, ch(X[(v, "TEST")]), Y["TEST"], preds["TEST"][:, 0], f"TEST_{stage['name']}")
        if "EXT" in SPLITS: gradcam_figure(gc_model, ch(X[(v, "EXT")]), Y["EXT"], preds["EXT"][:, 0], f"EXT_{stage['name']}")

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
# ## 6. Train all stages × seeds

# %%
for seed in SEEDS:
    for st in STAGES:
        run(st, seed)

# %% [markdown]
# ## 7. Summary metrics (image level, source-cell level, calibration, external)

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

rows = []
test_src = SPLITS["TEST"].source.values
for seed in SEEDS:
    for st in STAGES:
        f = f"{OUT}/preds_{st['name']}_s{seed}.npz"
        if not os.path.exists(f): continue
        P = np.load(f)
        ttas = TTA.items() if st["name"] in ("M5", "P_faithful", "M8", "M9") else [("none", ["I"])]
        for tname, names in ttas:
            sel = [D4.index(n) for n in names]
            if st["name"] == "M5":
                label = {"none": "M5", "4-fold": "M6", "6-fold": "M7", "8-fold (D4)": "M7_D4"}[tname]
            elif st["name"] in ("P_faithful", "M8", "M9"):
                label = st["name"] if tname == "none" else f"{st['name']}_{tname.split()[0]}"
            else:
                label = st["name"]
            pt = P["TEST"][:, sel].mean(1)
            r = dict(stage=label, seed=seed, tta=tname, **{f"test_{k}": v for k, v in metrics(pt, Y["TEST"]).items()})
            ps = pd.DataFrame(pt).groupby(test_src).mean()            # source-cell level (mean prob over augmented copies)
            ys = SPLITS["TEST"].groupby("source").y.first().loc[ps.index].values
            r["test_cell_acc"] = float((ps.values.argmax(1) == ys).mean()); r["n_test_cells"] = len(ys)
            r["test_simple_acc"] = float((P["TEST_SIMPLE"][:, sel].mean(1).argmax(1) == Y["TEST_SIMPLE"]).mean())
            r["val_acc"] = float((P["VAL"][:, sel].mean(1).argmax(1) == Y["VAL"]).mean())
            T = fit_temperature(P["VAL"][:, sel].mean(1), Y["VAL"]); pts = apply_temperature(pt, T)
            r["temperature"] = T; r["test_ece_ts"] = float(ece(pts, Y["TEST"])); r["test_brier_ts"] = brier(pts, Y["TEST"])
            if "EXT" in P.files:
                r.update({f"ext_{k}": v for k, v in metrics(P["EXT"][:, sel].mean(1), Y["EXT"]).items()})
            rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_all_seeds.csv", index=False)
num = [c for c in res.columns if c not in ("stage", "seed", "tta")]
order = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M7_D4", "X_noCLAHE",
         "P_faithful", "P_faithful_4-fold", "P_faithful_6-fold", "P_faithful_8-fold",
         "M8", "M8_4-fold", "M8_6-fold", "M8_8-fold", "M9", "M9_4-fold", "M9_6-fold", "M9_8-fold",
         "B_resnet50", "B_densenet121"]
summ = res.groupby("stage")[num].agg(["mean", "std"]).reindex([o for o in order if o in set(res.stage)])
summ.to_csv(f"{OUT}/results_summary.csv")
pd.set_option("display.width", 250)
print("MEAN over seeds\n", summ.xs("mean", axis=1, level=1).round(4).to_string())
print("\nSTD over seeds\n", summ.xs("std", axis=1, level=1).round(4).to_string())
print("\nPER-SEED (copy of results_all_seeds.csv)\n" + res.round(4).to_csv(index=False))

# confusion matrices of the final model, first seed
for mname in ["M5", "P_faithful", "M9"]:
    f = f"{OUT}/preds_{mname}_s{SEEDS[0]}.npz"
    if not os.path.exists(f): continue
    P = np.load(f)
    for s in [k for k in ["TEST", "EXT"] if k in P.files]:
        for tname in ["none", "6-fold"]:
            sel = [D4.index(n) for n in TTA[tname]]
            cm = confusion_matrix(Y[s], P[s][:, sel].mean(1).argmax(1), labels=range(len(CLASSES)))
            pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(f"{OUT}/confusion_{mname}_{tname}_{s}.csv")
            print(f"\n{s} | {mname} {tname}\n", pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

# %% [markdown]
# ## 8. Bundle results (send `results_bundle.zip` back; weights are kept separately for the code release)

# %%
with zipfile.ZipFile(f"{OUT}/results_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        if f.endswith((".npz", ".csv", ".json", ".png")):
            z.write(f"{OUT}/{f}", f)
print("results_bundle.zip: %.1f MB" % (os.path.getsize(f"{OUT}/results_bundle.zip") / 2**20))
