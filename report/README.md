# Cross-Manipulation Generalisation in Deepfake Face Detection

Team: Arun Kumar, Amit Kumar, Alok Raj, Arpita Tripathi
Date: [date]

## 1. Introduction

Deepfake detectors often perform very well on the manipulation method they were trained on, but fail on methods they have never seen. This project measures that gap using leave-one-manipulation-out evaluation on FaceForensics++ (c23), and tests whether blending-based augmentation reduces it.

## 2. Background

FaceForensics++ (Rössler et al., ICCV 2019, arXiv:1901.08971) contains 1,000 original videos and four manipulation methods: DeepFakes, Face2Face, FaceSwap and NeuralTextures. The paper trains an XceptionNet detector on all four methods together and reports results on compressed (c23 and c40) videos.

Celeb-DF (Li et al., CVPR 2020, arXiv:1909.12962) is a second-generation deepfake dataset with higher visual quality. Its evaluation shows that detectors trained on earlier datasets lose a large amount of accuracy when tested on new data. Its Table 4 reports frame-level AUC for XceptionNet (c23) trained on FaceForensics++: 99.7% on FF-DF, 85.9% on DFD, 72.2% on DFDC and 65.3% on Celeb-DF.

Those experiments test cross-dataset transfer with a model trained on all four methods. This project tests cross-method transfer, where each model sees only one method during training.

## 3. Data

- **Source:** FaceForensics++ c23, obtained as a public Kaggle copy (ff-faces-224). Used under the FaceForensics++ Terms of Use for research only; not redistributed.
- **Split:** the official FaceForensics++ split: 720 training, 140 validation and 140 test videos. Each fake video is in the same split as its source video. No video appears in more than one split.
- **Frames:** 5 frames per video, face crops from MTCNN with a 1.3× margin, resized to 224 × 224.
- **Training subset:** 200 training videos per method, plus the original videos used for each training run. Validation and test use all 140 videos.
- **Test methods:** DeepFakes, Face2Face, FaceSwap, NeuralTextures, and FaceShifter. FaceShifter is used only for testing, as an extra unseen method.
- **Known data gap:** DeepFakes validation has 698 frames instead of 700. Two frames were dropped during face detection.

## 4. Method

**Model:** EfficientNet-B0 (timm), ImageNet pre-trained, with one output for a real/fake score.

**Training:** AdamW, learning rate 1e-4, weight decay 1e-4, batch size 32, 5 epochs, mixed precision, random horizontal flip. Positive weight set from the class balance. The epoch with the highest validation AUC is kept.

**Baseline:** four models, one per training method (DeepFakes, Face2Face, FaceSwap, NeuralTextures).

**Augmented:** the same four models and settings, with blending-based augmentation, JPEG and blur augmentation added during training (see `models/augment.py`). The only difference from the baseline is the augmentation flag.

**Test rule:** for each test method, positives are that method's fake frames and negatives are the original frames from the same test videos. Metrics are frame-level AUC and equal error rate (EER).

**Seed:** 42 for all runs.

## 5. Results

### 5.1 Baseline matrix (AUC)

| Trained on | DeepFakes | Face2Face | FaceSwap | NeuralTextures | FaceShifter |
|---|---|---|---|---|---|
| DeepFakes | **0.943** | 0.607 | 0.398 | 0.728 | 0.570 |
| Face2Face | 0.728 | **0.901** | 0.562 | 0.671 | 0.436 |
| FaceSwap | 0.516 | 0.576 | **0.926** | 0.468 | 0.475 |
| NeuralTextures | 0.738 | 0.635 | 0.445 | **0.854** | 0.575 |

Bold cells are the same method for training and testing.

### 5.2 Augmented matrix (AUC)

| Trained on | DeepFakes | Face2Face | FaceSwap | NeuralTextures | FaceShifter |
|---|---|---|---|---|---|
| DeepFakes | **0.878** | 0.631 | 0.549 | 0.661 | 0.568 |
| Face2Face | 0.703 | **0.838** | 0.608 | 0.630 | 0.493 |
| FaceSwap | 0.605 | 0.600 | **0.866** | 0.555 | 0.570 |
| NeuralTextures | 0.716 | 0.674 | 0.553 | **0.794** | 0.536 |

### 5.3 Summary

| Measure | Baseline | Augmented | Change |
|---|---|---|---|
| Mean AUC, same method (diagonal) | 0.906 | 0.844 | −0.062 |
| Mean AUC, unseen methods (off-diagonal, including FaceShifter) | 0.571 | 0.603 | +0.032 |

The figures are `figures/baseline_auc.png`, `figures/augmented_auc.png` and `figures/auc_difference.png`. The per-cell comparison is in `results/comparison.csv`.

### 5.4 Largest changes

- **Gains:** FaceSwap-trained model on DeepFakes (0.516 → 0.605) and on FaceShifter (0.475 → 0.570); Deepfakes-trained model on FaceSwap (0.398 → 0.549).
- **Losses:** Deepfakes-trained model on NeuralTextures (0.728 → 0.661); NeuralTextures-trained model on FaceShifter (0.575 → 0.536).

## 6. Discussion

**Cross-method transfer is much weaker than same-method performance.** The average same-method AUC is 0.906, but the average on unseen methods is 0.571, close to chance. A detector that looks strong on its own training method can rank unseen manipulations poorly.

**Some models score below chance on unseen methods.** The FaceSwap-trained baseline scores 0.516 on DeepFakes, 0.468 on NeuralTextures and 0.475 on FaceShifter. An AUC below 0.5 means the model ranks fake frames as more real than real frames. This suggests the model learned artefacts specific to FaceSwap, which are absent or reversed in other methods. We report these values as measured.

**Augmentation improves transfer on average but reduces same-method accuracy.** Mean unseen-method AUC rises by 0.032, while mean same-method AUC falls by 0.062. The largest gains are on the FaceSwap-trained and Deepfakes-trained rows, where the baseline was weakest. The gains are not uniform: several cells dropped, including NeuralTextures-trained on FaceShifter.

**Comparison with the literature.** The Celeb-DF paper reports that XceptionNet trained on FaceForensics++ loses accuracy across datasets (99.7% on FF-DF down to 65.3% on Celeb-DF). Our results show a similar pattern across manipulation methods, but they are not directly comparable: our models are smaller, trained on a subset, use one method each, and are evaluated with a different test rule.

## 7. Limitations

1. **Single seed.** All results use seed 42. The differences between baseline and augmented cells are often 0.01 to 0.03, which may be within run-to-run variation. Only the large changes (such as the FaceSwap rows) should be treated as likely real. Seeds 43 and 44 would be needed to estimate variance.
2. **Training subset.** Each model used 200 training videos per method, not the full training set. Absolute scores are therefore likely lower than with full data.
3. **Five frames per video.** Frame sampling is coarse, and video-level results may differ.
4. **Split independence.** The split is video-disjoint, but the same person may appear in more than one video, so identity overlap across splits is possible.
5. **Data gap.** DeepFakes validation has 698 frames instead of 700. This affects model selection only slightly.
6. **Third-party data copy.** The dataset was obtained from a public Kaggle upload, not directly from the FaceForensics++ authors.
7. **Paper comparison.** Table 4 of the Celeb-DF paper tests cross-dataset transfer with models trained on all four methods, so it cannot be used to check our numbers directly.

## 8. Conclusion

Detectors trained on a single FaceForensics++ manipulation method generalise poorly to other methods. The average AUC drops from 0.906 on the training method to 0.571 on unseen methods, and some models fall below chance. Blending-based augmentation with JPEG and blur augmentation increases average unseen-method AUC to 0.603, at the cost of lower same-method accuracy. With one seed and a training subset, these results are indicative. Multiple seeds and the full training set are the next steps.

## References

- Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., Nießner, M. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019. https://arxiv.org/abs/1901.08971
- Li, Y., Yang, X., Sun, P., Qi, H., Lyu, S. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.* CVPR 2020. https://arxiv.org/abs/1909.12962
- Tan, M., Le, Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- Wightman, R. *PyTorch Image Models (timm).* https://github.com/huggingface/pytorch-image-models