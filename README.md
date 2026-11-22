# Cross-Manipulation Generalisation in Deepfake Face Detection

**Team:** Arun Kumar, Amit Kumar, Alok Raj, Arpita Tripathi

Deepfake detectors often score very well on the manipulation method they were trained on, but fail on methods they have never seen. This project measures that gap on FaceForensics++ (c23) by training one detector per manipulation method and testing it on all the others. It then tests whether blending-based augmentation (with JPEG and blur) reduces the gap.

The full write-up, in the ELCVIA journal format, is `report/ELCVIA_report.pdf`.

## Key results

Model: EfficientNet-B0, frame-level AUC, seed 42. Details are in the report.

| Measure | Baseline | Augmented | Change |
|---|---|---|---|
| Mean AUC, same method (diagonal) | 0.906 | 0.844 | −0.062 |
| Mean AUC, unseen methods (off-diagonal, including FaceShifter) | 0.571 | 0.603 | +0.032 |

- Cross-method transfer is far weaker than same-method performance, and some models score below chance (AUC < 0.5) on unseen methods.
- Augmentation helps on unseen methods on average, but lowers same-method accuracy.
- Bootstrap intervals (video-level, 95%) show that all four same-method drops and 8 of the 16 unseen-method gains are larger than test-sample noise. The average unseen-method gain is not significant (Wilcoxon p = 0.079).
- Augmentation also makes the detector flag more real frames as fake (11.6-23.0% before, 28.7-43.6% after, at threshold 0.5).
- One seed and a training subset were used, so the results are indicative. See the Limitations section of the report.

## Repository layout

```
data/splits.csv        official FF++ split (frozen, do not modify)
preprocessing/         preprocess.py: video -> 224x224 face crops
models/train.py        trains one model (use --augment for the augmented run)
models/augment.py      blending, JPEG and blur augmentation
models/run_augmented.py  runs all four augmented jobs (seed 42)
evaluation/metrics.py  AUC / EER computation, checks and heat maps
results/               baseline/, augmented/ (per-run CSVs), comparison.csv, eer_table.csv,
                       auc_ci.csv, diff_ci.csv, stats_summary.json, top_errors.csv
figures/               heat maps, ROC curves, diagrams
report/                ELCVIA_report.pdf, build_report.py, make_figures.py, analysis.py
```

## Data

- **Raw:** Kaggle `xdxd003/ff-c23` (FaceForensics++ c23 videos).
- **Splits:** `data/splits.csv`, the official FF++ split (720/140/140 videos), frozen. Both IDs of a fake video stay in the same split (identity-disjoint). Columns: `video_id, method, label, role, split, rel_path`.
- **Processed faces (224x224):** private Kaggle dataset `arpitatripathi2006/ff-faces-224`, shared with the team. It is not public because of the FF++ terms of use.
- **Settings:** 5 frames/video, 200 train videos per method, all val/test videos, MTCNN crop with 1.3x margin.
- **Methods:** original (real), Deepfakes, Face2Face, FaceSwap, NeuralTextures. FaceShifter is test-only.
- **Known gap:** DeepFakes validation has 698 frames instead of 700 (two frames were dropped in face detection).

### Using the faces

Attach the Kaggle dataset, then read `frames.csv` (columns: `split, method, label, video_id, path`; label 0 = real, 1 = fake). Always use the `split` column. Do not re-split.

### Regenerate faces

```
python preprocessing/preprocess.py --videos <videos_dir> --splits data/splits.csv --max-train 200 --frames 5
```

## How to run

Training needs a GPU and the Kaggle dataset, so it runs on Kaggle (`/kaggle/input` and `/kaggle/working` are hard-coded).

```
pip install -r requirements.txt            # facenet-pytorch: pip install facenet-pytorch --no-deps

# one model (baseline)
python models/train.py --train_method Deepfakes --seed 42
# one model (augmented)
python models/train.py --train_method Deepfakes --seed 42 --augment
# quick code check
python models/train.py --train_method Deepfakes --dry_run
```

`--train_method` is one of Deepfakes, Face2Face, FaceSwap, NeuralTextures. Each run saves `best.pt` (the epoch with the best validation AUC), `predictions.csv` and `results.csv`.

### Figures and report (local)

```
python report/make_figures.py     # ROC curves, bar charts, diagrams, EER table, seed variance
python report/analysis.py         # bootstrap CIs, Wilcoxon test, error analysis, diff_ci.png (takes a few minutes)
python report/build_report.py     # builds report/ELCVIA_report.pdf
```

`make_figures.py` needs pandas, scikit-learn and matplotlib; `analysis.py` also needs scipy. `build_report.py` needs reportlab and Times New Roman (Windows). The report adds optional figures and sections only when their files exist.

## Rules

- Do not modify `splits.csv`.
- Do not make the data public.
- Fix and report a random seed.

## References

- Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019. https://arxiv.org/abs/1901.08971
- Li et al., *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020. https://arxiv.org/abs/1909.12962
- Tan, Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019.
- Wightman, *PyTorch Image Models (timm)*. https://github.com/huggingface/pytorch-image-models
