"""CPU latency and FLOPs for the inference networks (architecture only; latency does not depend on weight values)."""
import os, time, json, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np, tensorflow as tf, keras
from keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
IMG = 224
def build(backbone):
    inp = keras.Input((IMG, IMG, 3)); x = inp
    if backbone == "mobilenetv2":
        x = layers.Rescaling(1 / 127.5, offset=-1)(x); base = keras.applications.MobileNetV2(include_top=False, weights=None, input_shape=(IMG, IMG, 3))
    elif backbone == "resnet50":
        x = layers.Lambda(keras.applications.resnet50.preprocess_input)(x); base = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(IMG, IMG, 3))
    else:
        base = keras.applications.EfficientNetB0(include_top=False, weights=None, input_shape=(IMG, IMG, 3))
    h = layers.GlobalAveragePooling2D()(base(x, training=False)); h = layers.Dense(256, activation="relu")(h); h = layers.Dropout(0.4)(h)
    return keras.Model(inp, layers.Dense(4, activation="softmax")(h))
def flops(model):
    f = tf.function(lambda x: model(x, training=False)).get_concrete_function(tf.TensorSpec([1, IMG, IMG, 3], tf.float32))
    g = convert_variables_to_constants_v2(f).graph
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(); opts["output"] = "none"
    return tf.compat.v1.profiler.profile(graph=g, options=opts).total_float_ops
def lat(model, bs, views, reps=5):
    x = np.random.randint(0, 255, (bs, IMG, IMG, 3)).astype(np.float32)
    f = tf.function(lambda t: model(t, training=False)); f(x)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(views): f(x).numpy()
        ts.append((time.perf_counter() - t0) / bs * 1000)
    return float(np.median(ts))
res = {"cpu": platform.processor() or platform.machine(), "threads": os.cpu_count(), "tf": tf.__version__}
for name, bb in [("MobileNetV2", "mobilenetv2"), ("EfficientNetB0", "efficientnetb0"), ("ResNet50", "resnet50")]:
    m = build(bb)
    r = dict(params=m.count_params(), gflops=flops(m) / 2e9)  # multiply-accumulate = 2 FLOPs; report GMAC-equivalent below
    r["gflops_total"] = flops(m) / 1e9
    r["ms_bs1"] = lat(m, 1, 1, reps=20); r["ms_bs16"] = lat(m, 16, 1); r["ms_bs16_6view"] = lat(m, 16, 6)
    res[name] = r; print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
json.dump(res, open("cpu_bench.json", "w"), indent=1)
