"""Supplementary Tables S2-S5 (second revision), typeset from results/secondary_stats.json.
Usage: python make_supp.py STATS_JSON OUT_DIR FIRST_PAGE_NUMBER"""
import sys, json, os
statsf, od, first = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(od, exist_ok=True)
R = json.load(open(statsf)); S = ["0", "1", "2"]

def pv(p): return "$<$0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.01 else f"{p:.2f}")
def sg(x, nd=1):
    if round(abs(x), nd) == 0: return f"{0:.{nd}f}"
    return f"+{x:.{nd}f}" if x >= 0 else f"$-${abs(x):.{nd}f}"
NAME = {"M5": "M5 + focal loss", "M8": "M8 (M5 + balanced sampling)", "X_noCLAHE": "Control: M5 without CLAHE",
        "final": "Final: M8 + 6-fold TTA", "B_resnet50": "Control: M5 with ResNet50"}
ARROW = lambda s: s.replace("->", " $\\rightarrow$ ")

# ---------------- S2: transfer comparisons + monocyte recall
PAIRN = {"X_noCLAHE->M5": "CLAHE (M5 against M5 without CLAHE)", "B_resnet50->M5": "Backbone (EfficientNetB0 against ResNet50)",
         "M5->M8": "Class-balanced sampling (M8 against M5)", "M8->final": "Six-fold TTA (final against M8)"}
rows2 = []
for ts, d in R["transfer_mcnemar"].items():
    for i, (pair, ds) in enumerate(d.items()):
        cells = " & ".join(f"{sg(ds[s]['diff_points'])} & {pv(ds[s]['p'])} & {pv(ds[s]['p_holm'])}" for s in S)
        rows2.append((ARROW(ts) if i == 0 else "") + f" & {PAIRN[pair]} & {cells} \\\\")
    rows2.append("\\addlinespace")
rows2 = "\n".join(rows2[:-1])
rowsm = []
for pair, lab in [("M5->M8", "M8 against M5"), ("M5->final", "Final pipeline against M5")]:
    ds = R["monocyte_mcnemar"][pair]
    cells = " & ".join(f"{ds[s]['recall_a']*100:.1f} $\\rightarrow$ {ds[s]['recall_b']*100:.1f} & {ds[s]['a_only']} / {ds[s]['b_only']} & {pv(ds[s]['p'])}" for s in S)
    rowsm.append(f"{lab} & {cells} \\\\")
rowsm = "\n".join(rowsm); nm = R["monocyte_mcnemar"]["M5->M8"]["0"]["n"]

# ---------------- S3: bootstrap intervals
rows3 = []
for ts, d in R["ci"].items():
    for i, cfg in enumerate(["M5", "M8", "X_noCLAHE", "final", "B_resnet50"]):
        a = " & ".join(f"{d[cfg][s]['acc']*100:.1f} [{d[cfg][s]['acc_ci'][0]*100:.1f}, {d[cfg][s]['acc_ci'][1]*100:.1f}]" for s in S)
        f = " & ".join(f"{d[cfg][s]['f1']:.3f} [{d[cfg][s]['f1_ci'][0]:.3f}, {d[cfg][s]['f1_ci'][1]:.3f}]" for s in S)
        rows3.append((ARROW(ts) + f" (n = {R['n'][ts]})" if i == 0 else "") + f" & {NAME[cfg]} & {a} & {f} \\\\")
    rows3.append("\\addlinespace")
rows3 = "\n".join(rows3[:-1])
rowsg = "\n".join(ARROW(k).replace(" minus ", " minus ") + " & " + " & ".join(
    f"{v[s]['diff_points']:.1f} [{v[s]['ci_points'][0]:.1f}, {v[s]['ci_points'][1]:.1f}]" for s in S) + " \\\\" for k, v in R["domain_gap"].items())
CN = {"eos": "Eosinophil", "lym": "Lymphocyte", "mon": "Monocyte", "neu": "Neutrophil"}
rowsr = "\n".join(f"{CN[c]} (n = {v['0']['n']}) & " + " & ".join(
    f"{v[s]['recall']*100:.1f} [{v[s]['ci'][0]*100:.1f}, {v[s]['ci'][1]*100:.1f}]" for s in S) + " \\\\" for c, v in R["raabin_recall_ci"].items())
fd = R["raabin_f1_diff"]
rowf = "Macro-F1, M5 minus M5 without CLAHE & " + " & ".join(f"{sg(fd[s]['diff'],3)} [{sg(fd[s]['ci'][0],3)}, {sg(fd[s]['ci'][1],3)}]" for s in S) + " \\\\"

# ---------------- S4: ECE differences
rows4 = "\n".join(k.replace("->", "$\\rightarrow$") + " & " + " & ".join(
    f"{v[s]['a']:.3f} $\\rightarrow$ {v[s]['b']:.3f} & {sg(v[s]['diff'],3)} [{sg(v[s]['ci'][0],3)}, {sg(v[s]['ci'][1],3)}]" for s in S) + " \\\\"
    for k, v in R["ece_diff"].items())

# ---------------- S5: few-shot
rows5 = []
for k in ["10", "50"]:
    for i, s in enumerate(S):
        v = R["fewshot"][k][s]
        def dc(key): return f"{sg(v[key]['diff_points'])} [{sg(v[key]['ci_points'][0])}, {sg(v[key]['ci_points'][1])}]"
        rows5.append((k if i == 0 else "") + f" & {int(s)+1} & {v['no_adaptation']*100:.1f} & {v['pbc_start']*100:.1f} & {v['imagenet_start']*100:.1f} & "
                     f"{v['full_training_M8']*100:.1f} & {dc('pbc_minus_imagenet')} & {dc('pbc_minus_full_training')} \\\\")
    rows5.append("\\addlinespace")
rows5 = "\n".join(rows5[:-1])

B = f"{R['bootstrap_resamples']:,}"
doc = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{fontspec}
\setmainfont{Latin Modern Roman}
\usepackage{graphicx,booktabs,tabularx,array,pdflscape,caption,makecell}
\captionsetup{font=small,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false}
\renewcommand{\thetable}{S\arabic{table}}
\setcounter{table}{1}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\setlength{\parindent}{0pt}
\pagestyle{plain}
\begin{document}
\setcounter{page}{FIRSTPAGE}
\begin{landscape}

\begin{table}[p]
\caption{Secondary comparisons, per seed. A: transfer comparisons on the external test sets, exact McNemar test; difference in accuracy (percentage points; positive when the first-named configuration is better), raw p and p after Holm correction within each seed and test set. B: monocyte recall on the PBC test set (NMONO monocyte images), exact McNemar test restricted to monocyte images, uncorrected. Test images of the Kaggle dataset include augmented copies of the same original image, so p-values on that test set are descriptive.}
\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}
\textbf{A. Transfer comparisons}\par\smallskip
\begin{tabularx}{\linewidth}{@{}l L rrr rrr rrr@{}}
\toprule
 & & \multicolumn{3}{c}{Seed 0} & \multicolumn{3}{c}{Seed 1} & \multicolumn{3}{c}{Seed 2} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}
Trained $\rightarrow$ tested & Comparison & Diff. & p & p (Holm) & Diff. & p & p (Holm) & Diff. & p & p (Holm) \\
\midrule
ROWS2
\bottomrule
\end{tabularx}

\bigskip
\textbf{B. Monocyte recall on PBC}\par\smallskip
\begin{tabularx}{\linewidth}{@{}L ccc ccc ccc@{}}
\toprule
 & \multicolumn{3}{c}{Seed 0} & \multicolumn{3}{c}{Seed 1} & \multicolumn{3}{c}{Seed 2} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}
Comparison & Recall (\%) & Lost / gained & p & Recall (\%) & Lost / gained & p & Recall (\%) & Lost / gained & p \\
\midrule
ROWSM
\bottomrule
\end{tabularx}

\smallskip
{\scriptsize Lost / gained: monocyte images classified correctly only by M5 / only by the compared configuration.}
\end{table}

\begin{table}[p]
\caption{Accuracy and macro-F1 on every test set with 95\% percentile bootstrap intervals (NBOOT resamples of the test images), per seed. The intervals reflect the sampling of test images, not training variability.}
\scriptsize\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.08}
{\footnotesize\textbf{A. Accuracy and macro-F1}}\par\smallskip
\begin{tabularx}{\linewidth}{@{}l L ccc ccc@{}}
\toprule
 & & \multicolumn{3}{c}{Accuracy (\%)} & \multicolumn{3}{c}{Macro-F1} \\
\cmidrule(lr){3-5}\cmidrule(l){6-8}
Trained $\rightarrow$ tested & Configuration & Seed 0 & Seed 1 & Seed 2 & Seed 0 & Seed 1 & Seed 2 \\
\midrule
ROWS3
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[p]
\ContinuedFloat
\caption{(continued) Further intervals for the final pipeline and for the effect of CLAHE on Raabin-WBC, per seed.}
\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}
\textbf{B. Difference between in-domain and cross-dataset accuracy of the final pipeline (percentage points; unpaired bootstrap)}\par\smallskip
\begin{tabularx}{\linewidth}{@{}L ccc@{}}
\toprule
Difference & Seed 0 & Seed 1 & Seed 2 \\
\midrule
ROWSG
\bottomrule
\end{tabularx}

\bigskip
\textbf{C. Per-class recall of the final pipeline on Raabin-WBC (\%)}\par\smallskip
\begin{tabularx}{\linewidth}{@{}L ccc@{}}
\toprule
Class & Seed 0 & Seed 1 & Seed 2 \\
\midrule
ROWSR
\bottomrule
\end{tabularx}

\bigskip
\textbf{D. Effect of CLAHE on macro-F1 on Raabin-WBC (paired bootstrap)}\par\smallskip
\begin{tabularx}{\linewidth}{@{}L ccc@{}}
\toprule
Difference & Seed 0 & Seed 1 & Seed 2 \\
\midrule
ROWF
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[p]
\caption{Differences in expected calibration error (ECE) between two configurations on the same test images, with 95\% paired bootstrap intervals (NBOOT resamples), per seed. The temperature is kept as fitted on the validation set. A negative difference means a lower ECE for the second-named configuration. Differences are computed from unrounded values.}
\footnotesize\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}L cc cc cc@{}}
\toprule
 & \multicolumn{2}{c}{Seed 0} & \multicolumn{2}{c}{Seed 1} & \multicolumn{2}{c}{Seed 2} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}
Comparison & ECE & Difference [95\% interval] & ECE & Difference [95\% interval] & ECE & Difference [95\% interval] \\
\midrule
ROWS4
\bottomrule
\end{tabularx}
\end{table}

\begin{table}[p]
\caption{Few-shot adaptation to the Kaggle laboratory (exploratory), per repetition: accuracy (\%) on the Kaggle test split (n = NKAG) without TTA. The per-image predictions of the adapted networks were not stored, so the differences are given with approximate 95\% intervals for independent proportions (Newcombe hybrid score); these intervals are descriptive, because the two networks are tested on the same images and Kaggle test images are not independent.}
\footnotesize\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\linewidth}{@{}cc cccc LL@{}}
\toprule
\makecell{Images\\per class} & \makecell{Repe-\\tition} & \makecell{No\\adaptation} & \makecell{PBC\\start} & \makecell{ImageNet\\start} & \makecell{M8 trained on the full\\Kaggle training split} & \makecell[l]{PBC start minus ImageNet start\\{[}95\% interval{]}} & \makecell[l]{PBC start minus full training\\{[}95\% interval{]}} \\
\midrule
ROWS5
\bottomrule
\end{tabularx}
\end{table}

\end{landscape}
\end{document}
"""
doc = (doc.replace("ROWS2", rows2).replace("ROWSM", rowsm).replace("ROWS3", rows3).replace("ROWSG", rowsg).replace("ROWSR", rowsr)
          .replace("ROWF", rowf).replace("ROWS4", rows4).replace("ROWS5", rows5).replace("NBOOT", B).replace("NMONO", str(nm))
          .replace("NKAG", f"{R['n']['Kaggle->Kaggle']:,}").replace("FIRSTPAGE", str(first)))
open(f"{od}/supp_tables.tex", "w").write(doc)
print("ok")
