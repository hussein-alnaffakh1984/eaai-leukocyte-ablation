"""Fill template.md with numbers from res_P.json (PBC) and res_K.json (Kaggle).
Usage: python render.py template.md res_P.json res_K.json manual.json out.md
manual.json holds hand-written passages (Grad-CAM description, literature comparison) and the code URL.
"""
import sys, json, re
import numpy as np

tpl, fp, fk, fman, out = sys.argv[1:6]
R = {"P": json.load(open(fp)), "K": json.load(open(fk))}
MAN = json.load(open(fman))
S = {k: len(v["seeds"]) for k, v in R.items()}

def m(ds, cfg, k): return R[ds]["mean"][cfg][k]
def sd(ds, cfg, k): return R[ds]["sd"][cfg][k]

def pct(ds, cfg, k="acc", sdv=True):
    v = m(ds, cfg, k) * 100
    s = sd(ds, cfg, k)
    return f"{v:.1f} ± {s * 100:.1f}%" if (sdv and s is not None) else f"{v:.1f}%"

def num(ds, cfg, k, nd=3, sdv=True):
    v = m(ds, cfg, k); s = sd(ds, cfg, k)
    return f"{v:.{nd}f} ± {s:.{nd}f}" if (sdv and s is not None) else f"{v:.{nd}f}"

def pp(ds, a, b, k="acc"): return (m(ds, b, k) - m(ds, a, k)) * 100

def ptxt(p):
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"

def sig(ds, a, b):
    """Summarise exact McNemar p-values over seeds for transition a -> b."""
    d = R[ds]["mcnemar"].get(f"{a}->{b}", {})
    ps = [v["p"] for v in d.values()]
    if not ps: return "test pending"
    k = sum(p < 0.05 for p in ps); n = len(ps)
    if n == 1: return ptxt(ps[0])
    dirs = {v["b_better"] > v["a_better"] for v in d.values() if v["p"] < 0.05}
    if len(dirs) > 1:
        v_ = lambda p: "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
        return f"significant in {k} of {n} seeds but in opposite directions, p from {v_(min(ps))} to {v_(max(ps))}"
    if k == n: return ("significant in both seeds, " if n == 2 else f"significant in all {n} seeds, ") + (f"all p < 0.001" if max(ps) < 0.001 else f"largest {ptxt(max(ps))}")
    if k == 0: return f"not significant in any seed, smallest {ptxt(min(ps))}"
    v = lambda p: "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
    return f"significant in {k} of {n} seeds, p from {v(min(ps))} to {v(max(ps))}"

def pcell(ds, a, b):
    d = R[ds]["mcnemar"].get(f"{a}->{b}", {})
    ps = [v["p"] for v in d.values()]
    if not ps: return "—"
    f = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    lo, hi = f(min(ps)), f(max(ps))
    return lo if lo == hi else f"{lo}–{hi}"

def signed(x): return f"+{x:.1f}" if x >= 0 else f"−{abs(x):.1f}"

def table(header, rows, caption):
    h = "| " + " | ".join(header) + " |\n| " + " | ".join("---" for _ in header) + " |\n"
    return f"**{caption}**\n\n" + h + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)

LBL = {"M1": "M1 MobileNetV2, frozen", "M2": "M2 + CLAHE", "M3": "M3 + fine-tuning", "M4": "M4 EfficientNetB0",
       "M5": "M5 + focal loss", "M6": "M6 + 4-fold TTA", "M7": "M7 + 6-fold TTA", "M8": "M8 M5 + balanced sampling",
       "final": "Final: M8 + 6-fold TTA", "X_noCLAHE": "Control: M5 without CLAHE", "B_resnet50": "Control: M5 with ResNet50",
       "M7_D4": "M5 + 8-fold TTA", "M8_4": "M8 + 4-fold TTA", "M8_8": "M8 + 8-fold TTA",
       "LOO_noCLAHE": "Final without CLAHE", "LOO_noFT": "Final without fine-tuning (frozen backbone)", "LOO_mobilenet": "Final with MobileNetV2",
       "LOO_ce": "Final with cross-entropy instead of focal loss", "LOO_nobal": "Final without class-balanced sampling", "LOO_noTTA": "Final without TTA"}
LOO_ORDER = ["LOO_noFT", "LOO_noTTA", "LOO_noCLAHE", "LOO_mobilenet", "LOO_nobal", "LOO_ce"]
PREV = {"M2": "M1", "M3": "M2", "M4": "M3", "M5": "M4", "M6": "M5", "M7": "M6", "M8": "M5", "final": "M7",
        "X_noCLAHE": "M5", "B_resnet50": "M5"}

def pair_key(cfg):
    a = PREV.get(cfg)
    if a is None: return None
    return (cfg, a) if cfg in ("X_noCLAHE", "B_resnet50") else (a, cfg)

T = {}
T["TABLE1"] = table(["Stage", "Backbone", "CLAHE", "Fine-tuning", "Loss", "Sampling", "TTA"], [
    ["M1", "MobileNetV2", "No", "No (frozen)", "Cross-entropy", "Uniform", "None"],
    ["M2", "MobileNetV2", "Yes", "No (frozen)", "Cross-entropy", "Uniform", "None"],
    ["M3", "MobileNetV2", "Yes", "Last 30 layers", "Cross-entropy", "Uniform", "None"],
    ["M4", "EfficientNetB0", "Yes", "Last 30 layers", "Cross-entropy", "Uniform", "None"],
    ["M5", "EfficientNetB0", "Yes", "Last 30 layers", "Focal", "Uniform", "None"],
    ["M6", "EfficientNetB0", "Yes", "Last 30 layers", "Focal", "Uniform", "4-fold"],
    ["M7", "EfficientNetB0", "Yes", "Last 30 layers", "Focal", "Uniform", "6-fold"],
    ["M8", "EfficientNetB0", "Yes", "Last 30 layers", "Focal", "Class-balanced", "None"],
    ["Final", "EfficientNetB0", "Yes", "Last 30 layers", "Focal", "Class-balanced", "6-fold"],
    ["Control A", "EfficientNetB0", "No", "Last 30 layers", "Focal", "Uniform", "None"],
    ["Control B", "ResNet50", "Yes", "Last 30 layers", "Focal", "Uniform", "None"]],
    "Table 1. Ablation stages. Each transition from M1 to M7 changes one factor; M8 changes the sampling of M5.")
T["TABLE2"] = table(["Parameter", "Setting"], [
    ["Input size", "224 × 224 × 3"], ["Classification head", "Global average pooling, dense 256 (ReLU), dropout 0.4, softmax 4"],
    ["Phase 1", "Backbone frozen, 10 epochs, Adam, learning rate 10⁻³"],
    ["Phase 2", "Last 30 backbone layers unfrozen (batch normalisation frozen), up to 20 epochs, learning rate 10⁻⁵"],
    ["Early stopping / learning-rate reduction", "Patience 4 (best weights restored) / factor 0.3, patience 2, minimum 10⁻⁷"],
    ["Batch size", "32"], ["Focal loss", "γ = 2.0, α = 0.25"], ["CLAHE", "L\\* channel, clip limit 2.0, 8 × 8 tiles"],
    ["Training augmentation", "Rotation ±20°, zoom ±15%, translation ±8%, horizontal flip, brightness ±15%"],
    ["Seeds", ("0, 1, 2" if S["P"] == S["K"] else "PBC: " + ", ".join(map(str, R["P"]["seeds"])) + "; Kaggle: " + ", ".join(map(str, R["K"]["seeds"]))) + " (data splits fixed with seed 42)"]], "Table 2. Hyper-parameters, identical for all stages.")

def ablation_rows(ds, cfgs, cols):
    rows = []
    for c in cfgs:
        if c not in R[ds]["mean"]: continue
        pk = pair_key(c)
        row = [LBL[c], pct(ds, c)] + [num(ds, c, k, 4 if k == "auc" else 3) for k in cols]
        row.append(pcell(ds, *pk) if pk else "—")
        rows.append(row)
    return rows

CFG_MAIN = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "final", "X_noCLAHE", "B_resnet50"]
T["TABLE3"] = table(["Configuration", "Accuracy", "Macro-F1", "Balanced accuracy", "Macro AUC", "McNemar p vs reference"],
                    ablation_rows("P", CFG_MAIN, ["f1", "bal", "auc"]),
                    f"Table 3. Ablation on the PBC test set (n = {R['P']['n']['TEST']}), mean ± standard deviation over {S['P']} seed{'s' if S['P'] > 1 else ''}. "
                    "McNemar p (exact, range over seeds) compares each stage with the previous one; M8 with M5; the final pipeline with M7; controls with M5. "
                    "Leave-one-out rows (all with six-fold TTA except the row without TTA; the rows without TTA and without class-balanced sampling are identical to M8 and M7) are compared with the final pipeline; in brackets, the number of seeds in which the difference stays significant after Holm correction within the leave-one-out family.")
def loo_cell(a):
    d = R["P"].get("mcnemar_loo", {}).get(f"{a}->final", {})
    if not d: return "—"
    f = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    ps = [v["p"] for v in d.values()]; k = sum(v["p_holm"] < 0.05 for v in d.values())
    return (f(min(ps)) if f(min(ps)) == f(max(ps)) else f"{f(min(ps))}–{f(max(ps))}") + f" ({k}/{len(d)} after Holm)"
if "mcnemar_loo" in R["P"]:
    _t3 = T["TABLE3"].rstrip("\n").split("\n")
    _nc = _t3[-1].count("|") - 1
    _t3.append("| *Leave-one-out from the final pipeline* |" + " |" * (_nc - 1))
    for c in LOO_ORDER:
        if c in R["P"]["mean"]:
            _t3.append("| " + " | ".join([LBL[c], pct("P", c), num("P", c, "f1"), num("P", c, "bal"), num("P", c, "auc", 4), loo_cell(c)]) + " |")
    T["TABLE3"] = "\n".join(_t3) + "\n"
T["TABLE4"] = table(["Configuration", "Accuracy", "Macro-F1", "McNemar p vs reference"],
                    [[r[0], r[1], r[2], r[3]] for r in ablation_rows("K", CFG_MAIN, ["f1"])],
                    f"Table 4. Ablation on the Kaggle test split (n = {R['K']['n']['TEST']}), mean ± standard deviation over {S['K']} seed{'s' if S['K'] > 1 else ''}. "
                    "McNemar p as in Table 3; because Kaggle test images include augmented copies of the same original, these p-values are descriptive.")

REC = [("rec_eos", "Eosinophil"), ("rec_lym", "Lymphocyte"), ("rec_mon", "Monocyte"), ("rec_neu", "Neutrophil")]
T["TABLE5"] = table(["Configuration"] + [n for _, n in REC], [[LBL[c]] + [pct("P", c, k) for k, _ in REC] for c in ["M5", "M7", "M8", "final"]],
                    f"Table 5. Per-class recall on the PBC test set, mean ± standard deviation over {S['P']} seed{'s' if S['P'] > 1 else ''}.")

def cal_rows(ds, name):
    return [[name, LBL[c], num(ds, c, "ece"), num(ds, c, "brier"), f"{m(ds, c, 'T') + 1e-9:.2f}", num(ds, c, "ece_ts"), num(ds, c, "brier_ts")]
            for c in ["M4", "M5", "M7", "final"]]
T["TABLE6"] = table(["Dataset", "Configuration", "ECE", "Brier", "Temperature", "ECE after scaling", "Brier after scaling"],
                    cal_rows("P", "PBC") + cal_rows("K", "Kaggle"),
                    "Table 6. Calibration on the test sets before and after temperature scaling fitted on the validation set (mean ± standard deviation; temperature is the mean).")

import os as _os
RB = json.load(open("raabin.json")) if _os.path.exists("raabin.json") else None
def rbv(c, k="acc", sdv=True):
    if not RB or c not in RB["mean"]: return "—"
    m_, s_ = RB["mean"][c][k], RB["sd"][c][k]
    if k == "f1": return f"{m_:.3f}" + (f" ± {s_:.3f}" if sdv and s_ is not None else "")
    return f"{m_*100:.1f}" + (f" ± {s_*100:.1f}%" if sdv and s_ is not None else "%")
def cross_rows():
    rows = []
    for c in ["M5", "M8", "X_noCLAHE", "final", "B_resnet50"]:
        rows.append([LBL[c], pct("P", c), pct("P", c, "ext_acc"), rbv(c), pct("K", c), pct("K", c, "ext_acc")])
    return rows
T["TABLE7"] = table(["Configuration", "PBC → PBC", "PBC → Kaggle", "PBC → Raabin", "Kaggle → Kaggle", "Kaggle → PBC"], cross_rows(),
                    f"Table 7. Accuracy within and across datasets (trained on → tested on), mean ± standard deviation. Test sets: PBC → PBC n = {R['P']['n']['TEST']}; PBC → Kaggle n = {R['P']['n']['KTEST']}; Kaggle → Kaggle n = {R['K']['n']['TEST']}; Kaggle → PBC n = {R['K']['n']['EXT']} (all PBC images of the four classes); PBC → Raabin n = {RB['n'] if RB else '—'} (Raabin-WBC Test-A).")

def cost_row(c, lbl):
    k = R["K"]["cost"].get(c); p = R["P"]["cost"].get(c)
    ref = p or k
    lat = ref["lat"]
    return [lbl, f"{ref['params'] / 1e6:.2f} M ({ref['trainable'] / 1e6:.2f} M)",
            f"{p['train_min']:.1f}" if p else "—", f"{k['train_min']:.1f}" if k else "—",
            f"{lat['none']:.1f} / {lat['4-fold']:.1f} / {lat['6-fold']:.1f}"]
_ex = sorted(k for k, v in R["P"]["cost"].items() if v.get("n_excluded"))
EXCL = (f"Runs that fell back to the CPU ({', '.join(_ex)}; one PBC seed each) are excluded from timing." if _ex else "")
T["TABLE8"] = table(["Configuration", "Parameters (trainable)", "Training time, PBC (min)", "Training time, Kaggle (min)", "Inference time per image, no TTA / 4-fold / 6-fold (ms)"],
                    [cost_row("M1", "M1 MobileNetV2, frozen"), cost_row("M3", "M3 MobileNetV2, fine-tuned"),
                     cost_row("M5", "M5 EfficientNetB0, fine-tuned"), cost_row("M8", "M8 / final pipeline"), cost_row("B_resnet50", "ResNet50, fine-tuned")],
                    "Table 8. Computational cost on an NVIDIA T4 GPU (training and inference: median over seeds; inference warmed-up, batches of 64, measured in the PBC runs). " + EXCL)
T["TABLE9"] = MAN["TABLE9"]

# ---------------- generated passages ----------------
def lower_higher(x): return "raised" if x > 0 else "lowered"
X = {}
P_ = "P"
X["P.final.acc"] = pct("P", "final"); X["K.final.acc"] = pct("K", "final"); X["P.final.kacc"] = pct("P", "final", "ext_acc"); X["K.final.eacc"] = pct("K", "final", "ext_acc")
X["CODE_URL"] = MAN["CODE_URL"]; X["WEIGHTS_URL"] = MAN["WEIGHTS_URL"]; X["K.faithful"] = MAN.get("K_FAITHFUL", ""); X["K.faithful6"] = MAN.get("K_FAITHFUL6", "")
X["P.final.accp"] = f"{m('P','final','acc')*100:.1f}%"; X["P.final.f1p"] = f"{m('P','final','f1'):.3f}"
X["P.final.aucp"] = f"{m('P','final','auc'):.3f}"; X["P.final.ecetsp"] = f"{m('P','final','ece_ts'):.3f}"
X["C.final.ms6"] = f"{(R['P']['cost'].get('M8') or R['K']['cost']['M8'])['lat']['6-fold']:.0f}"

_NM = {"M1->M2": "CLAHE on the frozen backbone (M1→M2)", "M2->M3": "fine-tuning (M2→M3)", "M3->M4": "the change of backbone (M3→M4)",
       "M4->M5": "focal loss (M4→M5)", "M5->M6": "four-fold TTA", "M6->M7": "six-fold TTA after four-fold", "M5->M7": "six-fold TTA in total",
       "M5->M8": "class-balanced sampling (M5→M8)", "M7->final": "class-balanced sampling under TTA (M7→final)", "M5->final": "the combined M5→final comparison",
       "X_noCLAHE->M5": "removing CLAHE from M5", "B_resnet50->M5": "EfficientNetB0 against ResNet50"}
def holm_all(ds):
    return [k for k, d in R[ds]["mcnemar"].items() if all(v.get("p_holm", 1) < 0.05 for v in d.values()) and len(d) == S[ds]]
_h = holm_all("P")
HOLM_TXT = ("After Holm correction within each seed, " + (", ".join(_NM.get(k, k) for k in _h[:-1]) + (" and " if len(_h) > 1 else "") + _NM.get(_h[-1], _h[-1]) if _h else "no transition") +
            (" remained significant in every seed." if len(_h) != 1 else " remained significant in every seed."))
if RB:
    _c = RB["counts"]
    X["RB.n"] = f"{RB['n']:,}"; X["RB.counts"] = f"{_c[3]:,} neutrophils, {_c[1]:,} lymphocytes, {_c[0]:,} eosinophils and {_c[2]:,} monocytes"
    X["RB.ratio"] = f"{max(_c)/min(_c):.1f}"; X["RB.neu"] = f"{_c[3]/sum(_c)*100:.0f}"
    fm = RB["mean"]["final"]; xm = RB["mean"]["X_noCLAHE"]; m1 = RB["mean"]["M1"]
    X["RB.final.acc"] = rbv("final"); X["RB.final.f1"] = rbv("final", "f1")
    X["RB.text"] = (
        f"On Raabin-WBC, which like PBC consists of single-cell crops but comes from another laboratory, the final pipeline reached {rbv('final')} accuracy, "
        f"but its macro-F1 was only {rbv('final','f1')}: neutrophils and lymphocytes were mostly recognised (recall {fm['rec_neu']*100:.0f}% and {fm['rec_lym']*100:.0f}%), "
        f"whereas eosinophils and monocytes were mostly missed (recall {fm['rec_eos']*100:.0f}% and {fm['rec_mon']*100:.0f}%). Because neutrophils make up {X['RB.neu']}% of these images, "
        f"accuracy alone overstates transfer. The loss on Kaggle images is therefore not only an effect of the field of view: even with similar single-cell crops, eosinophils and monocytes did not transfer, as on Kaggle. "
        f"{'CLAHE, which helped in-domain, did not help here: M5 without CLAHE reached ' + rbv('X_noCLAHE') + ' accuracy and a macro-F1 of ' + rbv('X_noCLAHE','f1') + ', against ' + rbv('M5') + ' and ' + rbv('M5','f1') + ' with CLAHE (no TTA; no significance test was run). ' if xm['f1'] > RB['mean']['M5']['f1'] else ''}"
        f"Few-shot adaptation, described next, was tested only on the Kaggle laboratory.")
else:
    for k in ["RB.n", "RB.counts", "RB.ratio", "RB.neu", "RB.text"]: X[k] = "[RAABIN PENDING]"
if "mcnemar_loo" in R["P"]:
    def lk(c): return sum(v["p_holm"] < 0.05 for v in R["P"]["mcnemar_loo"][f"{c}->final"].values())
    def ld(c): return signed(pp("P", "final", c))
    X["LOO.text"] = (
        f"The leave-one-out analysis (lower part of Table 3) confirms that fine-tuning dominates, whereas the backbone and CLAHE effects shrink to less than one point when all other components are present. Removing fine-tuning lowered accuracy by "
        f"{abs(pp('P','final','LOO_noFT')):.1f} points, significantly in {lk('LOO_noFT')} of {S['P']} seeds after Holm correction (per seed: " + ", ".join(f"{R['P']['per_seed']['LOO_noFT'][s]['acc']*100:.1f}%" for s in R['P']['per_seed']['LOO_noFT']) + "); this frozen EfficientNetB0 with CLAHE scored on average below M1, a frozen MobileNetV2 without CLAHE, so the drop includes the interaction between CLAHE and a frozen backbone seen in M1→M2. "
        f"Removing TTA ({ld('LOO_noTTA')} points), CLAHE ({ld('LOO_noCLAHE')}), the EfficientNetB0 backbone ({ld('LOO_mobilenet')}) or class-balanced sampling ({ld('LOO_nobal')}) "
        f"changed accuracy by less than one point each, significant after correction in at most {max(lk(c) for c in ['LOO_noTTA','LOO_noCLAHE','LOO_mobilenet','LOO_nobal'])} of {S['P']} seeds. "
        f"Replacing focal loss by cross-entropy left accuracy unchanged ({pct('P','LOO_ce')} against {pct('P','final')}) and lowered the ECE of the final pipeline from "
        f"{num('P','final','ece')} to {num('P','LOO_ce','ece')}, so in this pipeline focal loss affects calibration rather than accuracy.")
    X["LOO.ft"] = f"{abs(pp('P','final','LOO_noFT')):.1f}"
    X["LOO.focal"] = (f" The leave-one-out analysis supports this: with cross-entropy instead of focal loss, the final pipeline kept its accuracy and was better calibrated before scaling, so focal loss can be omitted from this pipeline.")
else:
    X["LOO.text"] = "[LOO PENDING]"; X["LOO.focal"] = ""
X["HOLM.M5final"] = f"{sum(v['p_holm'] < 0.05 for v in R['P']['mcnemar']['M5->final'].values())} of {S['P']} seeds"
X["N.pairs"] = str(len(R["P"]["mcnemar"]))
X["P.text.ablation"] = (
    f"The frozen MobileNetV2 reference (M1) reached {pct('P','M1')}. Luminance CLAHE on the frozen backbone (M2) changed accuracy by "
    f"{signed(pp('P','M1','M2'))} points ({sig('P','M1','M2')}). Fine-tuning the last 30 layers (M3) changed it by {signed(pp('P','M2','M3'))} points "
    f"({sig('P','M2','M3')}), and replacing MobileNetV2 by EfficientNetB0 (M4) by a further {signed(pp('P','M3','M4'))} points ({sig('P','M3','M4')}). "
    f"Focal loss (M5) changed accuracy by {signed(pp('P','M4','M5'))} points ({sig('P','M4','M5')}). Four-fold TTA added "
    f"{signed(pp('P','M5','M6'))} points ({sig('P','M5','M6')}) and six-fold TTA a further {signed(pp('P','M6','M7'))} ({sig('P','M6','M7')}); in total, six-fold TTA changed accuracy by {signed(pp('P','M5','M7'))} points relative to M5 ({sig('P','M5','M7')}). Class-balanced sampling (M8) "
    f"changed accuracy by {signed(pp('P','M5','M8'))} points relative to M5 ({sig('P','M5','M8')}), and the final pipeline (M8 with six-fold TTA) reached "
    f"{pct('P','final')}, macro-F1 {num('P','final','f1')} and macro AUC {num('P','final','auc')}. Although neither class-balanced sampling nor TTA was consistently significant on its own, their combination, tested post hoc, improved on M5 by {signed(pp('P','M5','final'))} points ({sig('P','M5','final')}). Removing CLAHE from M5 (control A) changed accuracy by "
    f"{signed(-pp('P','X_noCLAHE','M5'))} points ({sig('P','X_noCLAHE','M5')}), and the ResNet50 backbone (control B) reached {pct('P','B_resnet50')} "
    f"against {pct('P','M5')} for EfficientNetB0 ({sig('P','B_resnet50','M5')}). {HOLM_TXT} In the sensitivity analysis, eight-fold TTA gave {pct('P','M8_8')} on the test set, "
    f"but on the validation set it was not better than six-fold TTA ({m('P','M8_8','val_acc')*100:.1f}% against {m('P','final','val_acc')*100:.1f}%), so six-fold TTA was kept.")
X["K.text.ablation"] = (
    f"M1 reached {pct('K','M1')}; CLAHE added {signed(pp('K','M1','M2'))} points, fine-tuning {signed(pp('K','M2','M3'))} and EfficientNetB0 "
    f"{signed(pp('K','M3','M4'))}. Focal loss changed accuracy by {signed(pp('K','M4','M5'))} points ({sig('K','M4','M5')}), four-fold TTA by "
    f"{signed(pp('K','M5','M6'))} ({sig('K','M5','M6')}) and six-fold TTA by a further {signed(pp('K','M6','M7'))} ({sig('K','M6','M7')}), a total of {signed(pp('K','M5','M7'))} points relative to M5 ({sig('K','M5','M7')}). "
    f"The final pipeline reached {pct('K','final')} (eight-fold TTA: {pct('K','M8_8')}). Removing CLAHE from M5 lowered accuracy by {abs(pp('K','X_noCLAHE','M5')):.1f} points "
    f"({sig('K','X_noCLAHE','M5')}){', the largest effect of CLAHE in the study' if abs(pp('K','X_noCLAHE','M5')) >= max(abs(pp('P','M1','M2')), abs(pp('K','M1','M2')), abs(pp('P','X_noCLAHE','M5'))) else ''}. Unlike on PBC, ResNet50 reached a higher mean accuracy than EfficientNetB0 on this dataset "
    f"({pct('K','B_resnet50')} against {pct('K','M5')}; {sig('K','B_resnet50','M5')}). Validation accuracy of the fine-tuned stages ranged from "
    f"{min(m('K',c,'val_acc') for c in ['M3','M4','M5','M8','X_noCLAHE','B_resnet50'])*100:.1f}% to {max(m('K',c,'val_acc') for c in ['M3','M4','M5','M8','X_noCLAHE','B_resnet50'])*100:.1f}%, well above test accuracy, which is consistent with residual similarity between training and validation images (Section 3.1).")

HAVE_CM = "final" in R["P"]["confusion"] and "final_ext" in R["P"]["confusion"]
if HAVE_CM:
    cmP = np.array(list(R["P"]["confusion"]["final"].values())[0])
    cmX = np.array(list(R["P"]["confusion"]["final_ext"].values())[0])
    wrong_pred = [(cmX[:, j].sum() - cmX[j, j]) for j in range(4)]
    ext_err_cls = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"][int(np.argmax(wrong_pred))]
    ext_err_share = max(wrong_pred) / max(1, (cmX.sum() - np.trace(cmX)))
    off = [(cmP[i, j], i, j) for i in range(4) for j in range(4) if i != j]; off.sort(reverse=True)
CN = ["eosinophils", "lymphocytes", "monocytes", "neutrophils"]
X["P.text.perclass"] = (
    f"Table 5 reports per-class recall and Figure 4 the confusion matrices of the final pipeline. Monocytes, the second-smallest class, were the hardest: "
    f"their recall was {pct('P','M5','rec_mon')} with M5 and {pct('P','M8','rec_mon')} with class-balanced sampling (M8), while eosinophil recall changed from "
    f"{pct('P','M5','rec_eos')} to {pct('P','M8','rec_eos')}. With six-fold TTA, monocyte recall was {pct('P','M7','rec_mon')} for M7 and "
    f"{pct('P','final','rec_mon')} for the final pipeline. On the Kaggle test split, monocytes were the main failure: their recall was {pct('K','final','rec_mon')} for the final pipeline, "
    f"{pct('K','M5','rec_mon')} for M5 and {pct('K','X_noCLAHE','rec_mon')} without CLAHE." + (f" In the first seed, the most frequent errors of the final pipeline were "
    f"{CN[off[0][1]]} predicted as {CN[off[0][2]]} ({off[0][0]} images) and {CN[off[1][1]]} predicted as {CN[off[1][2]]} ({off[1][0]} images)." if HAVE_CM else ""))

X["P.text.calibration"] = (
    f"On PBC, the cross-entropy model M4 was well calibrated (ECE {num('P','M4','ece')}). Focal loss raised ECE to {num('P','M5','ece')}; the fitted temperature "
    f"of {m('P','M5','T') + 1e-9:.2f} (< 1) shows that the model became under-confident, as expected from a loss that down-weights confident predictions. Averaging over TTA views "
    f"softened the probabilities further (ECE {num('P','M7','ece')} for M7). After temperature scaling, ECE fell to {num('P','M5','ece_ts')} for M5, "
    f"{num('P','M7','ece_ts')} for M7 and {num('P','final','ece_ts')} for the final pipeline, whose Brier score fell from {num('P','final','brier')} to "
    f"{num('P','final','brier_ts')}. The Kaggle dataset behaved differently: focal loss lowered ECE from {num('K','M4','ece')} to {num('K','M5','ece')}, "
    f"but temperature scaling fitted on the Kaggle validation split raised test ECE to {num('K','M5','ece_ts')}, because that validation split is easier "
    f"than the test split and does not represent it.")

X["X.text.cross"] = (
    f"Table 7 reports transfer between the datasets. Models trained on PBC lost most of their accuracy on the Kaggle test split (final pipeline "
    f"{pct('P','final')} in-domain, {pct('P','final','ext_acc')} on Kaggle), and models trained on Kaggle reached {pct('K','final','ext_acc')} on PBC against "
    f"{pct('K','final')} in-domain. CLAHE mattered in one direction only: without it, Kaggle-to-PBC accuracy of M5 fell from {pct('K','M5','ext_acc')} to "
    f"{pct('K','X_noCLAHE','ext_acc')}, whereas PBC-to-Kaggle accuracy was {pct('P','M5','ext_acc')} with and {pct('P','X_noCLAHE','ext_acc')} without it. "
    f"ResNet50 transferred worse than EfficientNetB0 from Kaggle to PBC ({pct('K','B_resnet50','ext_acc')} against {pct('K','M5','ext_acc')}). "
    + (f"In the first seed, {ext_err_share*100:.0f}% of the errors of the PBC-trained final pipeline on Kaggle images were predictions of {ext_err_cls} (Figure 4), consistent with the different field of view of the two datasets." if HAVE_CM else ""))

CB = json.load(open("cpu_bench.json"))
_e, _r = CB["EfficientNetB0"], CB["ResNet50"]
CPU_TXT = (f"To assess use without a GPU, the networks were also timed on a two-core CPU ({CB['cpu'] if CB['cpu'] else 'Intel Xeon'}, 2.1 GHz, TensorFlow {CB['tf']}, batches of 16). "
    f"EfficientNetB0 needs {_e['gflops']:.2f} G multiply-accumulate operations per image and {_e['ms_bs16']:.1f} ms per image without TTA or {_e['ms_bs16_6view']:.1f} ms with six-fold TTA, "
    f"so a 100-cell differential count is classified in about {_e['ms_bs16']*100/1000:.1f} s, or {_e['ms_bs16_6view']*100/1000:.0f} s with TTA; ResNet50 needs {_r['gflops']:.2f} G operations and "
    f"{_r['ms_bs16']:.1f} or {_r['ms_bs16_6view']:.1f} ms. In 32-bit floating point the weights occupy {_e['params']*4/1e6:.1f} MB for EfficientNetB0 and {_r['params']*4/1e6:.1f} MB for ResNet50.")
ck = R["K"]["cost"]; cp = R["P"]["cost"]; ref = cp if "M5" in cp else ck
X["C.text.compute"] = (
    f"Table 8 reports the cost of each configuration. EfficientNetB0 has {ref['M5']['params']/1e6:.2f} M parameters, of which "
    f"{ref['M5']['trainable']/1e6:.2f} M are trained in phase 2, against {ref['B_resnet50']['params']/1e6:.1f} M for ResNet50. Inference with the final pipeline takes "
    f"{ref['M8']['lat']['none']:.1f} ms per image without TTA and {ref['M8']['lat']['6-fold']:.1f} ms with six-fold TTA; eight-fold TTA takes {ref['M8']['lat']['8-fold (D4)']:.1f} ms. On the GPU the cost grows "
    f"less than proportionally with the number of views because the views are processed in batches, whereas the accuracy gain is small (Section 4.1). Peak GPU memory is not reported because the values logged by TensorFlow were "
    f"inconsistent between identical models; the memory needed for the weights is reported instead. " + CPU_TXT)
X["G.text.gradcam"] = MAN["GRADCAM"]
X["CPU.ms"] = f"{CB['EfficientNetB0']['ms_bs16']:.0f}"; X["CPU.ms6"] = f"{CB['EfficientNetB0']['ms_bs16_6view']:.0f}"
_W = {1: "one", 2: "two", 3: "three"}
X["SEEDLIST"] = ", ".join(map(str, R["P"]["seeds"])) if S["P"] == S["K"] else "0 and 1 on PBC; 0, 1 and 2 on Kaggle"
X["NSEEDS"] = _W[S["P"]] if S["P"] == S["K"] else f"{_W[S['P']]} (PBC) and {_W[S['K']]} (Kaggle)"
_FT = ['M3','M4','M5','M8','X_noCLAHE','B_resnet50']
X["K.valrange"] = f"{min(m('K',c,'val_acc') for c in _FT)*100:.1f}–{max(m('K',c,'val_acc') for c in _FT)*100:.1f}%"
X["L.text.compare"] = MAN["COMPARE"]

X["D.text.ablation"] = (
    f"The ablation separates three groups of factors. Adapting the network to the task, by fine-tuning ({signed(pp('P','M2','M3'))} points on PBC, "
    f"{signed(pp('K','M2','M3'))} on Kaggle) and by changing the backbone from MobileNetV2 to EfficientNetB0 ({signed(pp('P','M3','M4'))} and {signed(pp('K','M3','M4'))}), produced the largest steps on both datasets; "
    f"because the stages are cumulative, these step sizes depend on the order of the path (Section 5.3), and the leave-one-out analysis confirms this: with all other components present, removing fine-tuning still cost {abs(pp('P','final','LOO_noFT')):.1f} points on PBC, whereas replacing EfficientNetB0 by MobileNetV2 cost less than one point. The choice between the two strong backbones was dataset-specific: EfficientNetB0 was better on PBC and ResNet50 on Kaggle. "
    f"The effect of luminance CLAHE depended on fine-tuning. With a frozen backbone it changed accuracy by {signed(pp('P','M1','M2'))} points on PBC and {signed(pp('K','M1','M2'))} on Kaggle, "
    f"presumably because enhanced images move further from the ImageNet statistics that the frozen features expect; once the backbone was fine-tuned, removing CLAHE lowered accuracy on both datasets "
    f"({signed(pp('P','M5','X_noCLAHE'))} and {signed(pp('K','M5','X_noCLAHE'))} points), most on the unevenly illuminated Kaggle fields. "
    f"Focal loss did not improve accuracy on either dataset ({signed(pp('P','M4','M5'))} and {signed(pp('K','M4','M5'))} points) and mainly changed calibration, which "
    f"supports describing it as hard-example reweighting rather than as an accuracy component.{{{{LOO.focal}}}} Class-balanced sampling changed monocyte recall on PBC by {signed(pp('P','M5','M8','rec_mon'))} points "
    f"and accuracy by {signed(pp('P','M5','M8'))} points, neither of which is a clear effect on its own; its combination with six-fold TTA, however, improved on M5 in every seed before correction (in {{{{HOLM.M5final}}}} after Holm correction). "
    f"TTA added small gains at a cost that grows with the number of views. For a practitioner, the order of priority is therefore: fine-tune (luminance contrast enhancement helped mainly on Kaggle and did not improve transfer to Raabin-WBC), "
    f"compare backbones on validation data from the target laboratory, recalibrate on held-out local data, and add class balancing and TTA when the latency budget allows.")
X["D.text.conclusion"] = (
    f"This study measured the contribution of five pipeline factors and class-balanced sampling to leukocyte classification under a single-factor protocol, on two datasets "
    f"and with {{{{NSEEDS}}}} seeds. On the PBC dataset the final pipeline reached {pct('P','final')}, with fine-tuning producing the largest effect in both the cumulative and the leave-one-out analyses. "
    f"Focal loss and TTA contributed little accuracy on their own, the effect of contrast enhancement depended on fine-tuning, and calibration required temperature scaling fitted on representative data. "
    f"Accuracy fell sharply across datasets ({pct('P','final','ext_acc')} and {pct('K','final','ext_acc')}), and on a third laboratory eosinophils and monocytes were largely missed (macro-F1 {rbv('final','f1')}), which shows that single-dataset accuracy, however high, "
    f"does not establish readiness for another laboratory. The released benchmark, with code, splits, weights and predictions, gives future work a documented reference point "
    f"against which new components can be measured on accuracy, calibration, transfer and cost.")

# ---------------- few-shot adaptation (fewshot.json, optional) ----------------
import os
FS = json.load(open("fewshot.json")) if os.path.exists("fewshot.json") else None
def fsv(k, init, key="KTEST_none"):
    d = FS[str(k)][init][key]; return f"{d['mean']*100:.1f}" + (f" ± {d['sd']*100:.1f}%" if d.get("sd") is not None else "%")
if FS:
    g10 = FS["10"]["pbc"]["KTEST_none"]["mean"]; g50 = FS["50"]["pbc"]["KTEST_none"]["mean"]
    i50 = FS["50"]["imagenet"]["KTEST_none"]["mean"]; k0 = m("P", "M8", "ext_acc"); kin = m("K", "final", "acc")
    share = (g50 - k0) / max(1e-9, kin - k0)
    X["F.text.fewshot"] = (
        f"Few-shot adaptation (Section 3.10) {'recovered most of the loss on the Kaggle test split' if share >= 0.6 else 'recovered part of the loss on the Kaggle test split'}. Starting from {fsv(0,'pbc')} without adaptation, "
        f"the PBC-trained network reached {fsv(10,'pbc')} on the Kaggle test split after fine-tuning on 10 labelled Kaggle images per class and "
        f"{fsv(50,'pbc')} after 50 images per class ({fsv(50,'pbc','KTEST_6-fold')} with six-fold TTA), against {fsv(10,'imagenet')} and {fsv(50,'imagenet')} "
        f"when the same images were used to train from ImageNet weights. "
        f"{'Training on PBC therefore gave a better starting point for a new laboratory than ImageNet alone. ' if g50 > i50 and g10 > FS['10']['imagenet']['KTEST_none']['mean'] else ''}"
        f"{'With 50 images per class the adapted network reached about the level of the same configuration (M8) trained on the full Kaggle training split (' + pct('K','M8') + ' without TTA); no significance test was run for this comparison. ' if g50 >= m('K','M8','acc') - 0.005 else ''}"
        f"After adaptation with 50 images per class, accuracy on the PBC test set was {fsv(50,'pbc','TEST_none')}, against {fsv(0,'pbc','TEST_none')} before adaptation"
        f"{', a case of catastrophic forgetting; the adapted network should therefore serve the new laboratory alongside, not instead of, the source model, or be adapted with rehearsal of source images.' if FS['50']['pbc']['TEST_none']['mean'] < FS['0']['pbc']['TEST_none']['mean'] - 0.05 else '.'}")
    X["F.text.deploy"] = (f" Our few-shot experiment suggests that this need not be costly: fine-tuning on 50 labelled images per class from the new laboratory raised accuracy "
                          f"from {fsv(0,'pbc')} to {fsv(50,'pbc')} (Section 4.5).")
    X["F.pbc0.short"] = f"{FS['0']['pbc']['KTEST_none']['mean']*100:.0f}%"; X["F.pbc50.short"] = f"{FS['50']['pbc']['KTEST_none']['mean']*100:.0f}%"
    X["F.letter.judge"] = (" This gives a practical, low-cost route to deploying the pipeline in a new laboratory." if share >= 0.6 else "")
    for k in ["0", "10", "50"]:
        for init in (["pbc"] if k == "0" else ["pbc", "imagenet"]):
            X[f"F.{init}{k}"] = fsv(k, init); X[f"F.{init}{k}.tta"] = fsv(k, init, "KTEST_6-fold"); X[f"F.{init}{k}.src"] = fsv(k, init, "TEST_none")
else:
    X["F.text.fewshot"] = "[FEW-SHOT RESULTS PENDING]"; X["F.text.deploy"] = " [FEW-SHOT PENDING]"; X["F.letter.judge"] = " [JUDGEMENT PENDING]"
    for k in ["0", "10", "50"]:
        for init in ["pbc", "imagenet"]:
            for suf in ["", ".tta", ".src"]: X[f"F.{init}{k}{suf}"] = "[PENDING]"
_e, _r = CB["EfficientNetB0"], CB["ResNet50"]
X["SIGK.M5final"] = (lambda d: f"{sum(v['p'] < 0.05 for v in d.values())} of {len(d)} seeds")(R["P"]["mcnemar"]["M5->final"])
X["TTA.ratio"] = f"{R['P']['cost']['M8']['lat']['6-fold'] / R['P']['cost']['M8']['lat']['none']:.1f}"
X["TTA.cpuratio"] = f"{CB['EfficientNetB0']['ms_bs16_6view'] / CB['EfficientNetB0']['ms_bs16']:.1f}"
X["CPU.memE"] = f"{_e['params']*4/1e6:.1f}"; X["CPU.memR"] = f"{_r['params']*4/1e6:.1f}"
X["CPU.gE"] = f"{_e['gflops']:.2f}"; X["CPU.gR"] = f"{_r['gflops']:.2f}"
X["CPU.cell100"] = f"{_e['ms_bs16']*100/1000:.1f}"; X["CPU.cell100tta"] = f"{_e['ms_bs16_6view']*100/1000:.0f}"
X["ABS.seeds"] = _W[S["P"]] if S["P"] == S["K"] else "two or three"
X["SEEDWORD"] = _W[S["P"]] if S["P"] == S["K"] else f"{_W[S['P']]} (PBC) and {_W[S['K']]} (Kaggle)"

# ---------------- generic tokens: {{v:DS:cfg:key}} value ± sd, {{m:DS:cfg:key}} mean only, {{d:DS:a:b}} difference in points,
# {{s:DS:a:b}} McNemar summary, {{n:DS:cfg:key}} 3-decimal value ± sd, {{nm:DS:cfg:key}} 3-decimal mean, {{T:DS:cfg}} temperature
def _gen(mo):
    kind, rest = mo.group(1), mo.group(2).split(":")
    if kind == "v": return pct(rest[0], rest[1], rest[2] if len(rest) > 2 else "acc")
    if kind == "m": return f"{m(rest[0], rest[1], rest[2] if len(rest) > 2 else 'acc')*100:.1f}%"
    if kind == "d": return signed(pp(rest[0], rest[1], rest[2], rest[3] if len(rest) > 3 else "acc"))
    if kind == "s": return sig(rest[0], rest[1], rest[2])
    if kind == "n": return num(rest[0], rest[1], rest[2])
    if kind == "nm": return f"{m(rest[0], rest[1], rest[2]):.3f}"
    if kind == "T": return f"{m(rest[0], rest[1], 'T') + 1e-9:.2f}"
    if kind == "c": c = R[rest[0]]["cost"][rest[1]]; return f"{c['lat'][rest[2]]:.1f}" if rest[2] in c["lat"] else f"{c[rest[2]]:.1f}"
    raise ValueError(mo.group(0))

txt = open(tpl).read()
for _ in range(2):
    for k, v in {**T, **X}.items():
        txt = txt.replace("{{" + k + "}}", v)
    txt = re.sub(r"\{\{(v|m|d|s|n|nm|T|c):([^}]+)\}\}", _gen, txt)
left = re.findall(r"\{\{[^}]+\}\}", txt)
assert not left, left
open(out, "w").write(txt)
print("rendered", out)
