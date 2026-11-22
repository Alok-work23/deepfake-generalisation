"""Local figures made only from the CSVs in results/ (no GPU, no dataset needed).

Run from the repo root:  python report/make_figures.py
Makes (in figures/):
  roc_curves.png      ROC curves of the FaceSwap- and DeepFakes-trained baselines
  seen_vs_unseen.png  bar chart: same-method vs unseen AUC, baseline vs augmented
  eer_table.csv       EER matrices (baseline, augmented)
  pipeline.png        preprocessing pipeline diagram
  split.png           split diagram
  experiment.png      experiment design diagram
  seed_summary.csv + seed_variance.png   only if seeds 43/44 are present
"""
# ======================================================================
# READ THIS FIRST (plain-language overview)
# ----------------------------------------------------------------------
# This file makes extra figures for the report. It needs NO GPU and NO images:
# it only reads the result tables (CSV files) that training already saved
# in results/baseline and results/augmented.
#
# Words used below:
#   "cell"      = one (trained-on method, tested-on method) pair, e.g. trained on
#                 FaceSwap and tested on DeepFakes. We have a 4 x 5 grid of cells.
#   "diagonal"  = cells where train method == test method (the model has seen it).
#   "unseen"    = all other cells (the model never saw that method in training).
#   AUC         = ranking score, 0.5 = random guessing, 1.0 = perfect.
#   EER         = equal error rate = error when false alarms equal missed fakes
#                 (lower is better).
#   ROC curve   = plot of true-positive rate against false-positive rate.
#
# What it produces (saved in figures/ or results/):
#   1. eer_table.csv      - EER numbers as a table
#   2. seen_vs_unseen.png - average AUC on seen vs unseen methods
#   3. roc_curves.png     - ROC curves for two baseline models
#   4. pipeline.png, split.png, experiment.png - simple box-and-arrow diagrams
#   5. seed_summary.csv / seed_variance.png - ONLY if results of more than one
#      seed exist (we currently have only seed 42, so this part is skipped)
# ======================================================================
import glob, os, re  # standard tools for file search, paths and regex
import numpy as np  # numpy does fast maths on arrays
import pandas as pd  # pandas handles tables (like Excel in code)
import matplotlib  # the main plotting library
matplotlib.use("Agg")  # draw to image files without opening a window
import matplotlib.pyplot as plt  # simple plotting interface
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # rounded boxes and arrows for diagrams
from sklearn.metrics import roc_curve  # computes points of an ROC curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root folder (two levels above this file)
FIG = os.path.join(ROOT, "figures")  # folder where figures are saved
M = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]  # fake methods used for training
T = M + ["FaceShifter"]  # test methods: the training ones plus one unseen method


def load(tag):  # read all result tables for one experiment tag
    fs = glob.glob(os.path.join(ROOT, "results", tag, "runs", "*", "results.csv"))  # find every run's results.csv
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)  # read each file and stack them into one table


base, aug = load("baseline"), load("augmented")  # load baseline and augmented results


def matrix(d, col, seed=None):  # build a train-vs-test table of one metric
    d = d if seed is None else d[d.seed == seed]  # keep only one seed if requested
    return d.groupby(["train_method", "test_method"])[col].mean().unstack().reindex(index=M, columns=T)  # average per pair, make a grid, fix row/column order


def seen_unseen(mat):  # summarise a matrix as same-method vs unseen-method averages
    seen = np.mean([mat.loc[m, m] for m in M])  # average of the diagonal (same method)
    unseen = np.nanmean([mat.loc[r, c] for r in M for c in T if r != c])  # average of off-diagonal cells, ignoring missing values
    return seen, unseen  # give back both averages


# --- EER table ------------------------------------------------------------
eb, ea = matrix(base, "eer"), matrix(aug, "eer")  # EER matrices for baseline and augmented
pd.concat({"baseline_eer": eb, "augmented_eer": ea}).round(3).to_csv(os.path.join(ROOT, "results", "eer_table.csv"))  # stack both, round, save as CSV

# --- seen vs unseen bars --------------------------------------------------
ab, aa = matrix(base, "auc"), matrix(aug, "auc")  # AUC matrices for baseline and augmented
sb, ub = seen_unseen(ab); sa, ua = seen_unseen(aa)  # seen/unseen averages for each model
fig, ax = plt.subplots(figsize=(5, 3.6))  # create a small figure with one plot
x = np.arange(2); w = 0.35  # two bar groups and the bar width
r1 = ax.bar(x - w / 2, [sb, ub], w, label="Baseline")  # baseline bars on the left of each group
r2 = ax.bar(x + w / 2, [sa, ua], w, label="Augmented")  # augmented bars on the right of each group
ax.bar_label(r1, fmt="%.3f", fontsize=8); ax.bar_label(r2, fmt="%.3f", fontsize=8)  # write the values on top of the bars
ax.axhline(0.5, color="gray", ls="--", lw=0.8); ax.text(1.45, 0.51, "chance", fontsize=7, color="gray", ha="right")  # dashed chance line with a label
ax.set_xticks(x, ["Same method\n(diagonal)", "Unseen methods\n(off-diagonal)"])  # name the two bar groups
ax.set_ylabel("mean AUC"); ax.set_ylim(0, 1.05); ax.legend(loc="upper right")  # y label, y range and legend
plt.tight_layout(); plt.savefig(os.path.join(FIG, "seen_vs_unseen.png"), dpi=200); plt.close()  # tidy layout, save image, close figure

# --- ROC curves -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)  # two side-by-side plots sharing the y axis
for ax, tm in zip(axes, ["FaceSwap", "Deepfakes"]):  # one plot per training method
    p = pd.read_csv(os.path.join(ROOT, "results", "baseline", "runs", f"{tm}_seed42", "predictions.csv"))  # load that model's predictions
    for t in T:  # draw one curve per test method
        s = p[p.test_method == t]  # rows for this test method only
        fpr, tpr, _ = roc_curve(s.label, s.score)  # compute false/true positive rates
        ax.plot(fpr, tpr, label=f"{t} ({ab.loc[tm, t]:.2f})", lw=2 if t == tm else 1.2)  # plot curve with AUC in label; thicker for same method
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)  # diagonal line = random guessing
    ax.set_title(f"Baseline trained on {tm}"); ax.set_xlabel("false positive rate"); ax.legend(fontsize=7)  # title, x label and legend
axes[0].set_ylabel("true positive rate")  # y label only on the left plot
plt.tight_layout(); plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=200); plt.close()  # tidy layout, save image, close figure


# --- diagrams -------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#dbe9f7"):  # draw a rounded box with centred text
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="black", lw=1))  # the rounded rectangle itself
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)  # text placed in the middle of the box


def arrow(ax, x1, y1, x2, y2):  # draw an arrow between two points
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.2))  # arrow from start to end


def canvas(w, h):  # make an empty drawing area of a given size
    f, a = plt.subplots(figsize=(w, h)); a.set_xlim(0, w); a.set_ylim(0, h); a.axis("off"); return f, a  # sized axes, no frame, return both


f, a = canvas(10, 1.8)  # canvas for the pipeline diagram
labs = ["FF++ c23\nvideo", "sample\n5 frames", "MTCNN\nface detect", "crop with\n1.3x margin", "resize\n224 x 224"]  # text of each pipeline step
for i, t in enumerate(labs):  # go through the steps in order
    box(a, 0.2 + i * 1.95, 0.4, 1.5, 1.0, t)  # draw this step's box, shifted right each time
    if i:  # skip the arrow before the first box
        arrow(a, 0.2 + i * 1.95 - 0.43, 0.9, 0.2 + i * 1.95, 0.9)  # arrow from the previous box into this one
plt.savefig(os.path.join(FIG, "pipeline.png"), dpi=200, bbox_inches="tight"); plt.close()  # save the diagram and close it

f, a = canvas(9, 2.6)  # canvas for the split diagram
box(a, 0.2, 0.5, 2.0, 1.6, "1000 videos\n(FF++ c23)")  # box for the whole dataset
for i, (t, c) in enumerate([("train\n720 videos", "#d6efd6"), ("val\n140 videos", "#fdf0c8"), ("test\n140 videos", "#f7d4d4")]):  # each split with its colour
    box(a, 3.2, 1.65 - i * 0.7, 2.2, 0.55, t.replace("\n", ": "), c)  # box for the split, stacked downward
    arrow(a, 2.3, 1.3, 3.15, 1.92 - i * 0.7)  # arrow from the dataset to the split
a.text(5.7, 1.3, "A fake video and its\nsource video always stay\nin the SAME split\n(video-disjoint)", fontsize=9, va="center")  # note about the split rule
plt.savefig(os.path.join(FIG, "split.png"), dpi=200, bbox_inches="tight"); plt.close()  # save the diagram and close it

f, a = canvas(10, 3.6)  # canvas for the experiment diagram
box(a, 0.2, 1.4, 2.2, 0.8, "real + ONE fake\nmethod (train)")  # training data box
box(a, 3.2, 1.4, 2.2, 0.8, "EfficientNet-B0\n(ImageNet init)")  # model box
arrow(a, 2.45, 1.8, 3.15, 1.8)  # data flows into the model
box(a, 3.2, 2.8, 2.2, 0.6, "+ blend / JPEG / blur\n(augmented only)", "#fdf0c8")  # augmentation box
arrow(a, 4.3, 2.75, 4.3, 2.25)  # augmentation feeds the model
box(a, 6.4, 2.7, 3.3, 0.7, "test: DeepFakes, Face2Face,\nFaceSwap, NeuralTextures", "#d6efd6")  # seen test methods box
box(a, 6.4, 1.75, 3.3, 0.7, "test: FaceShifter\n(never seen in training)", "#f7d4d4")  # unseen test method box
box(a, 6.4, 0.5, 3.3, 0.8, "AUC and EER per cell\n4 x 5 matrix", "#e6e0f5")  # results box
arrow(a, 5.45, 1.9, 6.35, 3.0); arrow(a, 5.45, 1.8, 6.35, 2.1); arrow(a, 5.45, 1.6, 6.35, 1.0)  # arrows from the model to the three boxes
plt.savefig(os.path.join(FIG, "experiment.png"), dpi=200, bbox_inches="tight"); plt.close()  # save the diagram and close it

# --- seed variance (only when extra seeds exist) --------------------------
seeds = sorted(set(base.seed) & set(aug.seed))  # seeds present in both experiments
if len(seeds) > 1:  # only continue if there are several seeds
    rows = []  # will collect one record per seed and model
    for s in seeds:  # loop over seeds
        for tag, d in (("baseline", base), ("augmented", aug)):  # loop over both models
            sn, un = seen_unseen(matrix(d, "auc", s))  # seen/unseen AUC for this seed
            rows.append(dict(seed=s, model=tag, same_method=sn, unseen=un))  # store the numbers as a record
    sd = pd.DataFrame(rows)  # turn the records into a table
    sd.to_csv(os.path.join(ROOT, "results", "seed_summary.csv"), index=False)  # save the table as CSV
    g = sd.groupby("model")[["same_method", "unseen"]].agg(["mean", "std"])  # mean and spread across seeds per model
    print(g.round(3))  # show the summary in the console
    fig, ax = plt.subplots(figsize=(5, 3.6))  # new small figure
    for i, tag in enumerate(["baseline", "augmented"]):  # one set of bars per model
        ax.bar(np.arange(2) + (i - 0.5) * 0.35, [g.loc[tag, ("same_method", "mean")], g.loc[tag, ("unseen", "mean")]],  # bar positions and mean heights
               0.35, yerr=[g.loc[tag, ("same_method", "std")], g.loc[tag, ("unseen", "std")]], capsize=4, label=tag)  # bar width, error bars from std, legend label
    ax.set_xticks([0, 1], ["Same method", "Unseen methods"]); ax.set_ylabel(f"mean AUC (+/- std, {len(seeds)} seeds)")  # x tick names and y label
    ax.set_ylim(0, 1.05); ax.legend()  # y range and legend
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "seed_variance.png"), dpi=200); plt.close()  # tidy layout, save image, close figure
print("figures written to", FIG)  # tell the user where figures went
