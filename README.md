# Cross-Manipulation Generalisation in Deepfake Face Detection

## Data

- **Raw:** Kaggle `xdxd003/ff-c23` (FaceForensics++ c23 videos).
- **Splits:** `data/splits.csv`, the official FF++ split (720/140/140 videos), frozen. Both IDs of a fake video stay in the same split (identity-disjoint). Columns: `video_id, method, label, role, split, rel_path`.
- **Processed faces (224x224):** private Kaggle dataset `arpitatripathi2006/ff-faces-224`, shared with the team. It is not public because of the FF++ terms of use.
- **Settings:** 5 frames/video, 200 train videos per method, all val/test videos, MTCNN crop with 1.3x margin.
- **Methods:** original (real), Deepfakes, Face2Face, FaceSwap, NeuralTextures. FaceShifter is test-only.

## Regenerate faces

```
python preprocessing/preprocess.py --videos <videos_dir> --splits data/splits.csv --max-train 200 --frames 5
```

## Using the faces

Attach the Kaggle dataset, then read `frames.csv` (columns: `split, method, label, video_id, path`; label 0 = real, 1 = fake). Always use the `split` column. Do not re-split.

## Rules

- Do not modify `splits.csv`.
- Do not make the data public.
- Fix and report a random seed.

## Citation

Rossler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019.
