# %% [markdown]
# # EAAI-26-12471 — Step 3: rotation-dense test-time augmentation (no retraining)
# Re-uses the weights saved by Step 2. Attach as inputs: the Step-2 notebook output (`bloodblood`),
# `paultimothymooney/blood-cells`, `unclesamulus/blood-cells-image-dataset`. GPU. Run with Save & Run All (Commit).
# Rationale: test images are arbitrary rotations/shifts of 71 source cells; image-level accuracy (~73%) is far below
# cell-level accuracy (~98%), i.e. predictions are not rotation-invariant. Averaging over dense rotations at inference
# is the principled extension of the manuscript's TTA component. All view sets are pre-specified below.

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


WDIR = os.path.dirname(sorted(glob.glob(f"{ROOT}/**/weights_M5_s0.weights.h5", recursive=True))[0])
print("Step-2 outputs found in:", WDIR)
MODELS = [m for m in E("BM_MODELS", "M5,M8,B_resnet50").split(",") if m]
BACKBONE = {"M5": "efficientnetb0", "M8": "efficientnetb0", "M9": "efficientnetb0", "B_resnet50": "resnet50"}
ANGLES = list(range(0, 360, int(E("BM_STEP", 15))))
VIEWS = [(a, f) for f in (0, 1) for a in ANGLES]            # (angle in degrees, horizontal flip)
print("Views:", len(VIEWS))

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
SPLITS = {s: meta[meta.split == s].reset_index(drop=True) for s in ["VAL", "TEST", "TEST_SIMPLE"]}
if len(ext): SPLITS["EXT"] = ext
X = {s: load_all(d.path.tolist(), True) for s, d in SPLITS.items()}   # all evaluated models use CLAHE
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


def view(xb, ang, flip):
    out = np.empty_like(xb)
    M = cv2.getRotationMatrix2D((IMG / 2 - 0.5, IMG / 2 - 0.5), ang, 1.0)
    for i in range(len(xb)):
        im = xb[i][:, ::-1] if flip else xb[i]
        out[i] = im if ang == 0 else cv2.warpAffine(np.ascontiguousarray(im), M, (IMG, IMG), flags=cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out

def predict_views(model, Xa, chunk=256):
    out = np.zeros((len(Xa), len(VIEWS), len(CLASSES)), np.float16)
    for i in range(0, len(Xa), chunk):
        xb = Xa[i:i + chunk]
        for j, (a, f) in enumerate(VIEWS):
            out[i:i + chunk, j] = model.predict(view(xb, a, f).astype(np.float32), batch_size=PBS, verbose=0)
    return out

# %%
t_all = time.time()
for name in MODELS:
    for seed in SEEDS:
        tag = f"{name}_s{seed}"; fout = f"{OUT}/dense_{tag}.npz"
        if os.path.exists(fout): print("skip", tag); continue
        keras.backend.clear_session()
        model, _, _ = build(BACKBONE[name], seed, False, name == "M9")
        model.load_weights(f"{WDIR}/weights_{tag}.weights.h5")
        old = np.load(f"{WDIR}/preds_{tag}.npz")
        chk = model.predict(X["TEST"][:256].astype(np.float32), batch_size=PBS, verbose=0)
        diff = float(np.abs(chk - old["TEST"][:256, 0]).max())
        print(tag, "sanity check vs Step-2 predictions, max |diff| =", round(diff, 5))
        assert diff < 1e-2, "weights/data mismatch with Step 2"
        t0 = time.time()
        preds = {s: predict_views(model, X[s]) for s in X}
        np.savez_compressed(fout, **preds)
        print(tag, "done in %.0fs" % (time.time() - t0))
print("total %.0f min" % ((time.time() - t_all) / 60))

# %% [markdown]
# ## Summary: accuracy vs number of views (pre-specified view sets), per model and ensembles

# %%
def view_idx(step, flips):
    return [j for j, (a, f) in enumerate(VIEWS) if a % step == 0 and f in flips]
VIEWSETS = {"1 (none)": view_idx(360, (0,)), "4 rot": view_idx(90, (0,)), "8 (D4)": view_idx(90, (0, 1)),
            "12 rot (30deg)": view_idx(30, (0,)), "24 (30deg x flip)": view_idx(30, (0, 1)),
            f"{len(ANGLES)} rot": view_idx(ANGLES[1] if len(ANGLES) > 1 else 360, (0,)), f"{len(VIEWS)} (all)": list(range(len(VIEWS)))}
ys = {s: SPLITS[s].y.values for s in X}
src = SPLITS["TEST"].source.values
rows = []
ENSEMBLES = [[m] for m in MODELS] + ([MODELS] if len(MODELS) > 1 else [])
for ens in ENSEMBLES:
    D = {s: [np.load(f"{OUT}/dense_{m}_s{seed}.npz")[s].astype(np.float32) for m in ens for seed in SEEDS] for s in X}
    for vname, vi in VIEWSETS.items():
        r = dict(models="+".join(ens), views=vname, n_views=len(vi))
        for s in X:
            p = np.mean([d[:, vi].mean(1) for d in D[s]], 0)
            r[f"{s.lower()}_acc"] = float((p.argmax(1) == ys[s]).mean())
            if s == "TEST":
                r["test_macro_f1"] = float(f1_score(ys[s], p.argmax(1), average="macro"))
                pc = pd.DataFrame(p).groupby(src).mean(); yc = SPLITS["TEST"].groupby("source").y.first().loc[pc.index].values
                r["test_cell_acc"] = float((pc.values.argmax(1) == yc).mean())
        rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/dense_tta_summary.csv", index=False)
pd.set_option("display.width", 250); print(res.round(4).to_string(index=False))

with zipfile.ZipFile(f"{OUT}/dense_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        if f.startswith("dense_") and f.endswith((".npz", ".csv")): z.write(f"{OUT}/{f}", f)
print("dense_bundle.zip: %.1f MB" % (os.path.getsize(f"{OUT}/dense_bundle.zip") / 2**20))
