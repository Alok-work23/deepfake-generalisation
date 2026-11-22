"""Statistical analysis from the saved predictions (no GPU needed).

Run from the repo root:  python report/analysis.py
Writes:
  results/auc_ci.csv            95% video-level bootstrap CI for every AUC cell
  results/diff_ci.csv           paired bootstrap CI of (augmented - baseline) per cell
  results/stats_summary.json    Wilcoxon tests and error-analysis numbers
  figures/diff_ci.png           change in AUC per cell with 95% CI
  results/top_errors.csv        most confident wrong frames per model/test method
"""
import glob, json, os  # standard library: file patterns, JSON writing, paths
import numpy as np  # numpy for fast array maths
import pandas as pd  # pandas for tables (DataFrames)
from scipy.stats import wilcoxon  # paired significance test
from sklearn.metrics import roc_auc_score  # computes the AUC metric
import matplotlib  # plotting library
matplotlib.use("Agg")  # use a non-window backend so figures save to files
import matplotlib.pyplot as plt  # pyplot is the main plotting interface

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root = two folders up from this file
M = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]  # the four training methods
T = M + ["FaceShifter"]  # test methods = training ones plus one extra unseen method
B = 1000  # number of bootstrap resamples
rng = np.random.default_rng(42)  # seeded random generator so results are repeatable


def preds(tag, m):  # load the saved predictions for one model
    p = pd.read_csv(os.path.join(ROOT, "results", tag, "runs", f"{m}_seed42", "predictions.csv"))  # read the CSV file
    p["video"] = p.path.str.split("/").str[-2]  # video id is the folder name in the frame path
    return p  # give back the table


def fast_auc(y, s):  # small wrapper to compute AUC
    return roc_auc_score(y, s)  # AUC from true labels y and scores s


def boot_idx(vids_pos, vids_neg):  # pick random videos for one bootstrap round
    """Resample videos (not frames) with replacement, separately for positives and negatives."""
    ps = rng.choice(len(vids_pos), len(vids_pos))  # random positive-video indices (with replacement)
    ns = rng.choice(len(vids_neg), len(vids_neg))  # random negative-video indices (with replacement)
    return ps, ns  # return both index lists


def cell_boot(pa, pb):  # bootstrap one (train, test) cell
    """Return point AUCs and bootstrap arrays for baseline, augmented, and difference."""
    assert (pa.path.values == pb.path.values).all()  # both tables must list the same frames in the same order
    y = pa.label.values  # true labels (1 = fake, 0 = real)
    vid = pa.video.values  # video id of every frame
    pos_v = np.unique(vid[y == 1]); neg_v = np.unique(vid[y == 0])  # unique fake videos and unique real videos
    pos_rows = {v: np.where((vid == v) & (y == 1))[0] for v in pos_v}  # row numbers of frames for each fake video
    neg_rows = {v: np.where((vid == v) & (y == 0))[0] for v in neg_v}  # row numbers of frames for each real video
    sa, sb = pa.score.values, pb.score.values  # model scores for the two tables
    a0, b0 = fast_auc(y, sa), fast_auc(y, sb)  # the plain (non-resampled) AUC of each model
    A, Bv = [], []  # lists to collect bootstrap AUCs
    for _ in range(B):  # repeat the resampling B times
        ps, ns = boot_idx(pos_v, neg_v)  # choose random videos
        rows = np.concatenate([pos_rows[pos_v[i]] for i in ps] + [neg_rows[neg_v[i]] for i in ns])  # gather all frames of the chosen videos
        A.append(fast_auc(y[rows], sa[rows])); Bv.append(fast_auc(y[rows], sb[rows]))  # store AUC of each model on this resample
    A, Bv = np.array(A), np.array(Bv)  # turn the lists into arrays
    return a0, b0, A, Bv  # return point AUCs and bootstrap arrays


rows_ci, rows_d = [], []  # collect rows for the CI table and the difference table
base_auc, aug_auc = {}, {}  # dictionaries holding the point AUC per cell
for m in M:  # loop over each training method
    pb_all, pa_all = preds("baseline", m), preds("augmented", m)  # load both models' predictions
    for t in T:  # loop over each test method
        pb = pb_all[pb_all.test_method == t].reset_index(drop=True)  # keep only baseline rows for this test method
        pa = pa_all[pa_all.test_method == t].reset_index(drop=True)  # keep only augmented rows for this test method
        a0, b0, A, Bv = cell_boot(pb, pa)  # run the bootstrap for this cell
        base_auc[(m, t)], aug_auc[(m, t)] = a0, b0  # remember the point AUCs
        lo = lambda x: np.percentile(x, 2.5); hi = lambda x: np.percentile(x, 97.5)  # helpers for the 95% interval ends
        rows_ci.append(dict(train=m, test=t, model="baseline", auc=a0, lo=lo(A), hi=hi(A)))  # add baseline CI row
        rows_ci.append(dict(train=m, test=t, model="augmented", auc=b0, lo=lo(Bv), hi=hi(Bv)))  # add augmented CI row
        d = Bv - A  # paired difference of bootstrap AUCs
        rows_d.append(dict(train=m, test=t, diff=b0 - a0, lo=lo(d), hi=hi(d),  # add the difference row: point diff and CI
                           significant=bool(lo(d) > 0 or hi(d) < 0), seen=(m == t)))  # significant if CI excludes 0; seen if same method
        print(m, t, round(a0, 3), round(b0, 3), flush=True)  # show progress

ci = pd.DataFrame(rows_ci).round(4); ci.to_csv(os.path.join(ROOT, "results", "auc_ci.csv"), index=False)  # build, round and save CI table
dd = pd.DataFrame(rows_d).round(4); dd.to_csv(os.path.join(ROOT, "results", "diff_ci.csv"), index=False)  # build, round and save difference table

# --- Wilcoxon signed-rank on the paired cells --------------------------------
off = dd[~dd.seen]  # cells where the test method was unseen in training
w_off = wilcoxon(off["diff"])  # Wilcoxon test on unseen-cell differences
w_diag = wilcoxon(dd[dd.seen]["diff"])  # Wilcoxon test on seen (diagonal) differences
summary = {  # dictionary of summary numbers
    "n_unseen_cells": int(len(off)),  # how many unseen cells
    "unseen_mean_diff": float(off["diff"].mean()),  # average AUC change on unseen cells
    "unseen_wilcoxon_p": float(w_off.pvalue),  # p-value for unseen cells
    "unseen_cells_improved": int((off["diff"] > 0).sum()),  # unseen cells that got better
    "unseen_cells_significant_up": int(((off.lo > 0)).sum()),  # unseen cells significantly better
    "unseen_cells_significant_down": int(((off.hi < 0)).sum()),  # unseen cells significantly worse
    "seen_mean_diff": float(dd[dd.seen]["diff"].mean()),  # average AUC change on seen cells
    "seen_wilcoxon_p": float(w_diag.pvalue),  # p-value for seen cells
    "seen_cells_significant_down": int(((dd.seen) & (dd.hi < 0)).sum()),  # seen cells significantly worse
    "all_significant_cells": int(dd.significant.sum()),  # all cells with a significant change
    "total_cells": int(len(dd)),  # total number of cells
}  # end of the summary dictionary

# --- error analysis ----------------------------------------------------------
errs = []  # list of the worst mistakes
for tag in ("baseline", "augmented"):  # loop over both model versions
    for m in M:  # loop over training methods
        p = preds(tag, m)  # load this model's predictions
        p["wrong_conf"] = np.where(p.label == 1, 1 - p.score, p.score)  # confidence in the wrong class
        p["wrong"] = (p.score >= 0.5).astype(int) != p.label  # True when the 0.5-threshold prediction is wrong
        for t in T:  # loop over test methods
            q = p[p.test_method == t]  # rows for this test method
            summary[f"{tag}|{m}|{t}|acc"] = float(1 - q.wrong.mean())  # accuracy = 1 - error rate
            summary[f"{tag}|{m}|{t}|fake_missed"] = float(((q.label == 1) & (q.score < 0.5)).sum() / (q.label == 1).sum())  # share of fakes called real
            summary[f"{tag}|{m}|{t}|real_flagged"] = float(((q.label == 0) & (q.score >= 0.5)).sum() / (q.label == 0).sum())  # share of reals called fake
            top = q[q.wrong].sort_values("wrong_conf", ascending=False).head(3)  # three most confident mistakes
            for _, r in top.iterrows():  # go through each of those rows
                errs.append(dict(model=tag, train=m, test=t, label=int(r.label), score=round(r.score, 4),  # record model, methods, label and score
                                 path=r.path.split("/faces/")[-1]))  # record the short frame path
pd.DataFrame(errs).to_csv(os.path.join(ROOT, "results", "top_errors.csv"), index=False)  # save the error list as CSV
json.dump(summary, open(os.path.join(ROOT, "results", "stats_summary.json"), "w"), indent=1)  # save the summary as JSON

# --- figure: AUC change with 95% CI --------------------------------------------
dd["lab"] = dd.train.str[:6] + " -> " + dd.test.str[:6]  # short label like "Deepfa -> FaceSw"
dd = dd.sort_values("diff").reset_index(drop=True)  # sort cells by AUC change
fig, ax = plt.subplots(figsize=(6.5, 7))  # create the figure and axes
col = np.where(dd.seen, "tab:red", np.where(dd.significant, "tab:green", "tab:gray"))  # colour by seen / significant / not
ax.errorbar(dd["diff"], range(len(dd)), xerr=[dd["diff"] - dd.lo, dd.hi - dd["diff"]], fmt="none", ecolor="k", lw=0.8)  # draw CI bars
ax.scatter(dd["diff"], range(len(dd)), c=col, zorder=3, s=22)  # draw a dot at each difference
ax.axvline(0, color="k", lw=0.8)  # vertical line at zero change
ax.set_yticks(range(len(dd)), dd.lab, fontsize=7)  # label each row
ax.set_xlabel("AUC change (augmented - baseline), 95% video-bootstrap CI")  # x-axis title
ax.set_title("red = same method, green = unseen & significant, gray = not significant", fontsize=8)  # legend-like title
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures", "diff_ci.png"), dpi=200); plt.close()  # fit layout, save image, close figure
print(json.dumps({k: v for k, v in summary.items() if "|" not in k}, indent=1))  # print only the top-level summary numbers
