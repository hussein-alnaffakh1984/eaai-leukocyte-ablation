"""Figures for the revised manuscript. Usage: python figures.py res_P.json res_K.json outdir"""
import sys, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fp, fk, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
R = {"P": json.load(open(fp)), "K": json.load(open(fk))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Fig. 2 flowchart ----------
fig, ax = plt.subplots(figsize=(7.4, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off")
def box(x, y, w, h, t, fc="#eef4fc", ec=BLUE, fs=7.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
box(0.5, 46, 19, 12, "PBC dataset\n9,072 images\n4 classes\n70/15/15 split")
box(0.5, 29, 19, 12, "Kaggle dataset\n12,444 images\npredefined\ntest split")
box(24, 38, 15, 12, "Preprocessing\n\nluminance\nCLAHE on/off")
box(43.5, 38, 26, 12, "Backbone and training\n\nMobileNetV2, EfficientNetB0\nor ResNet50;\nfrozen or fine-tuned")
box(74, 38, 25.5, 12, "Loss and sampling\n\ncross-entropy or focal loss;\nuniform or class-balanced")
box(74, 20, 25.5, 12, "Stored predictions\n\nvalidation and test sets,\nall 8 D4 views,\n" + (f"{len(R['P']['seeds'])} seeds" if len(R["P"]["seeds"]) == len(R["K"]["seeds"]) else f"{len(R['P']['seeds'])} (PBC) / {len(R['K']['seeds'])} (Kaggle) seeds"), fc="#fff1ea", ec=ORANGE)
box(43.5, 20, 26, 12, "Inference\n\nTTA: none, 4, 6 or 8 views;\ntemperature scaling", fc="#fff1ea", ec=ORANGE)
box(0.5, 18.5, 19, 7.5, "Raabin-WBC\n4,250 images\nexternal test only", fc="#f4f4f4", ec=INK2)
box(24, 18.5, 15.5, 12.5, "Additional\nanalyses\n\nleave-one-out;\nfew-shot (M8)", fc="#f4f4f4", ec=INK2)
arrow(10, 18.2, 10, 15.3); arrow(31.75, 18.2, 31.75, 15.3)
box(0.5, 1, 99, 14, "Evaluation\naccuracy, macro-F1, balanced accuracy, AUC, per-class recall  ·  ECE, Brier score, reliability diagrams\n"
    "exact McNemar test per seed with Holm correction  ·  cross-dataset transfer in both directions\nexternal test on Raabin-WBC  ·  parameters, training and inference time  ·  Grad-CAM saliency maps", fc="#ebf8f2", ec=AQUA)
arrow(19.8, 52, 23.7, 46); arrow(19.8, 35, 23.7, 42); arrow(39.3, 44, 43.2, 44); arrow(69.8, 44, 73.7, 44)
arrow(86.75, 37.7, 86.75, 32.3); arrow(73.7, 26, 69.8, 26); arrow(56.5, 19.7, 56.5, 15.3)
ax.text(0.5, 61.5, "One factor is changed per stage (M1 to M8, final pipeline and two controls); all other settings are fixed.", fontsize=7.6, color=INK2)
fig.savefig(f"{od}/fig2_framework.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 3 ablation ----------
ST = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "final", "X_noCLAHE", "B_resnet50"]
LB = ["M1", "M2\n+CLAHE", "M3\n+FT", "M4\nEffNet", "M5\n+focal", "M6\n4-TTA", "M7\n6-TTA", "M8\n+bal.", "Final", "No\nCLAHE", "ResNet\n50"]
PREV = {"M2": "M1", "M3": "M2", "M4": "M3", "M5": "M4", "M6": "M5", "M7": "M6", "M8": "M5", "final": "M7", "X_noCLAHE": "M5", "B_resnet50": "M5"}
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharey=False)
for ax, ds, col, title in [(axs[0], "P", BLUE, "PBC (hospital laboratory, single-cell crops)"), (axs[1], "K", ORANGE, "Kaggle (augmented smear fields)")]:
    mu = np.array([R[ds]["mean"][s]["acc"] * 100 for s in ST]); sd = np.array([(R[ds]["sd"][s]["acc"] or 0) * 100 for s in ST])
    x = np.arange(len(ST)); colors = [col] * 9 + ["#a8a7a2"] * 2
    ax.bar(x, mu, width=0.72, color=colors, yerr=sd, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), zorder=2)
    for i, s in enumerate(ST):
        a = PREV.get(s)
        if a:
            key = f"{s}->{a}" if s in ("X_noCLAHE", "B_resnet50") else f"{a}->{s}"
            ps = [v["p"] for v in R[ds]["mcnemar"].get(key, {}).values()]
            if ps and all(p < 0.05 for p in ps): ax.text(i, mu[i] + sd[i] + 0.6, "*", ha="center", fontsize=10, color=INK)
    lo = max(0, np.floor((mu - sd).min() / 5) * 5 - 5); ax.set_ylim(lo, 100)
    ax.set_xticks(x); ax.set_xticklabels(LB, fontsize=7); ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title, fontsize=8.5, color=INK, loc="left"); ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
fig.text(0.01, -0.04, f"Bars: mean over seeds (PBC: {len(R['P']['seeds'])}, Kaggle: {len(R['K']['seeds'])}); error bars: standard deviation; * significant (exact McNemar, p < 0.05) against the reference stage in every seed; grey: controls.",
         fontsize=6.6, color=INK2)
fig.tight_layout(); fig.savefig(f"{od}/fig3_ablation.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 4 confusion ----------
if "final" in R["P"]["confusion"]:
    CL = ["Eos", "Lym", "Mon", "Neu"]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, (ds, key, title) in zip(axs, [("P", "final", "Final pipeline, PBC test"), ("P", "final_ext", "Trained on PBC, Kaggle test")]):
        cm = np.array(list(R[ds]["confusion"][key].values())[0]); rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8, color="white" if rn[i, j] > 0.55 else INK)
        ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(CL); ax.set_yticklabels(CL)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=8.5, loc="left")
        for s in ax.spines.values(): s.set_visible(False)
    fig.text(0.01, -0.03, "Seed 0; cell text = counts, shading = row-normalised recall.", fontsize=6.6, color=INK2)
    fig.tight_layout(); fig.savefig(f"{od}/fig4_confusion.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ---------- Fig. 5 reliability ----------
if "final" in R["P"]["reliability"]:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, (ds, cfg, title) in zip(axs, [("P", "M4", "PBC, M4 (cross-entropy)"), ("P", "M5", "PBC, M5 (focal loss)"), ("P", "final", "PBC, final pipeline")]):
        for kind, col, lab in [("raw", ORANGE, "before scaling"), ("ts", BLUE, "after temperature scaling")]:
            rows = [r for r in R[ds]["reliability"][cfg][kind] if r[2] >= 10]
            conf = [r[3] for r in rows]; acc = [r[4] for r in rows]
            ax.plot(conf, acc, "-o", color=col, lw=2.6 if kind == "raw" else 1.4, ms=4.5 if kind == "raw" else 3, label=lab, zorder=2 if kind == "raw" else 3)
        ax.plot([0, 1], [0, 1], "--", color=GRID, lw=1); ax.set_xlim(0.3, 1.01); ax.set_ylim(0.3, 1.02)
        ax.set_title(title, fontsize=8, loc="left"); ax.set_xlabel("Confidence"); ax.yaxis.grid(True, color=GRID, lw=0.5)
    axs[0].set_ylabel("Accuracy"); axs[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.text(0.01, -0.05, "Seed 0, PBC test set, 15 equal-width bins; bins with fewer than 10 images omitted; dashed line = perfect calibration. For M4 the fitted temperature is close to 1, so the two curves almost coincide.", fontsize=6.6, color=INK2)
    fig.tight_layout(); fig.savefig(f"{od}/fig5_reliability.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("figures written to", od)
