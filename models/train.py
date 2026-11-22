
# Walk Kaggle's input folders and create output directories.
import os
# Define and parse the configuration supplied on the command line.
import argparse
# Seed Python's random generator for reproducible augmentation choices.
import random

# Use NumPy for numerical operations when calculating metrics.
import numpy as np
# Use pandas to load, filter, and write tabular CSV data.
import pandas as pd

# PyTorch handles tensors, GPU computation, training, and saved weights.
import torch
# Use PyTorch's neural-network module for the binary classification loss.
import torch.nn as nn

# Load a crop from disk and convert it to RGB.
from PIL import Image
# Dataset describes one example at a time. DataLoader groups examples into batches.
from torch.utils.data import Dataset, DataLoader
# Compose image operations such as flipping, tensor conversion, and normalization.
from torchvision import transforms

# Load the pretrained EfficientNet architecture.
import timm

# Calculate the ROC curve and the area under it.
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# 1. COMMAND-LINE ARGUMENTS
# ============================================================
# These arguments allow us to change training settings
# without editing the Python code every time.
#
# Example:
# python train.py --train_method Deepfakes --epochs 5 --batch 32

p = argparse.ArgumentParser()

# Name of the fake video generation method used for training.
# Example: Deepfakes, Face2Face, FaceSwap, etc.
p.add_argument("--train_method", required=True)

# Number of times the model will go through the entire
# training dataset.
p.add_argument("--epochs", type=int, default=5)

# Number of images processed together in one training step.
p.add_argument("--batch", type=int, default=32)

# Learning rate controls how much the model updates
# its weights during training.
p.add_argument("--lr", type=float, default=1e-4)

# Random seed makes experiments more reproducible.
p.add_argument("--seed", type=int, default=42)

# If enabled, trains on a very small subset of data
# for quickly checking whether the code works.
p.add_argument("--dry_run", action="store_true")

# Enable augmentation only when this flag appears on the command line.
p.add_argument("--augment", action="store_true")

args = p.parse_args()

if args.augment:
    from augment import augment_training_image


# ============================================================
# 2. SET RANDOM SEEDS
# ============================================================
# Randomness is used in data shuffling, model initialization,
# and other operations.
#
# Setting the same seed helps us get similar results
# when we run the experiment multiple times.

random.seed(args.seed)
np.random.seed(args.seed)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Use GPU for faster training.
# This code assumes that a CUDA-enabled GPU is available.
dev = "cuda"


# ============================================================
# 3. FIND THE DATASET
# ============================================================
# Kaggle stores input datasets inside /kaggle/input.
#
# We search all folders until we find frames.csv.
# This CSV file contains information about the dataset,
# including image paths, labels, methods, and data splits.

ROOT = None

# os.walk yields the current directory path, subdirectory names, and filenames. The last matching directory becomes ROOT.
for dp, dn, fn in os.walk("/kaggle/input"):

    # Look for the manifest name among the files in this directory.
    if "frames.csv" in fn:
        ROOT = dp

# Read the dataset information from the CSV file.
df = pd.read_csv(f"{ROOT}/frames.csv")

# Convert relative image paths into complete file paths.
# This allows Python to locate and load the actual images.
df["path"] = ROOT + "/" + df["path"]


# ============================================================
# 4. FAKE METHODS USED FOR FINAL TESTING
# ============================================================
# These are the five fake video generation methods
# on which we will evaluate our trained model.
#
# The model is trained using only ONE fake method at a time,
# but it is tested on ALL five methods.
#
# This helps us understand whether the model can detect
# fake images from methods it may not have seen during training.

TEST_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    # FaceShifter appears in evaluation only in the four-run experiment.
    "FaceShifter"
]


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================
# Neural networks work with numerical image data.
# Therefore, we convert images into tensors and normalize them.
#
# Normalization uses the mean and standard deviation
# of the ImageNet dataset because our model is pretrained
# on ImageNet.

norm = transforms.Normalize(
    [0.485, 0.456, 0.406],  # RGB mean values
    [0.229, 0.224, 0.225]   # RGB standard deviation values
)


# Transformations applied to training images.
train_tf = transforms.Compose([

    # Randomly flip an image horizontally.
    # This increases data variety and helps prevent overfitting.
    transforms.RandomHorizontalFlip(),

    # Convert the image into a PyTorch tensor.
    transforms.ToTensor(),

    # Normalize the image using ImageNet statistics.
    norm
])


# Transformations applied during validation and testing.
# We do not use random augmentation here because we want
# consistent predictions for the same image.

eval_tf = transforms.Compose([
    transforms.ToTensor(),
    norm
])


# ============================================================
# 6. CUSTOM DATASET CLASS
# ============================================================
# PyTorch uses Dataset classes to load data one sample at a time.
#
# This class:
# 1. Receives image information from the CSV.
# 2. Opens the image.
# 3. Converts it to RGB.
# 4. Applies the required transformations.
# 5. Returns the image and its label.
#
# Label:
# 0 = Real image
# 1 = Fake image

class FrameDS(Dataset):

    def __init__(self, d, tf, augment=False):
        self.augment = augment

        # Store the dataframe and reset its row numbering.
        # Reset the filtered table's row index so its rows are easy to address in order.
        self.d = d.reset_index(drop=True)

        # Store the image transformations.
        self.tf = tf

    def __len__(self):

        # Return the total number of images in the dataset.
        return len(self.d)

    def __getitem__(self, i):

        # Get information about the image at index i.
        r = self.d.iloc[i]

        # Open the image and convert it to RGB format.
        img = Image.open(r.path).convert("RGB")

        # Apply preprocessing and convert the label to float.
        # Convert the CSV target to a floating-point value for the binary loss.
        label = float(r.label)
        if self.augment:
            img, label = augment_training_image(img, label)
        return self.tf(img), label


# ============================================================
# 7. CREATE DATA LOADERS
# ============================================================
# DataLoader loads images in batches instead of loading
# the entire dataset into memory.
#
# This makes training more memory-efficient.

def loader(d, tf, shuffle):

    return DataLoader(
        # Only the training loader passes shuffle=True, so evaluation never calls augmentation.
        FrameDS(d, tf, augment=args.augment and shuffle),

        # Number of images loaded in one batch.
        batch_size=args.batch,

        # Shuffle training data to avoid learning
        # from a fixed image order.
        shuffle=shuffle,

        # Use two worker processes to load images faster.
        num_workers=2,

        # Speeds up transfer of data to the GPU.
        pin_memory=True
    )


# ============================================================
# 8. PREPARE TRAINING AND VALIDATION DATA
# ============================================================
# We train a separate model for each fake generation method.
#
# For example, if:
# --train_method Deepfakes
#
# Then the model will only see:
#   Real images (original)
#   Fake images (Deepfakes)
#
# We do not use other fake generation methods for training.

train_df = df[
    # Take training frames from the existing split column.
    (df.split == "train") &
    (df.method.isin(["original", args.train_method]))
]

# Validation data is selected using the same two methods.
# Validation helps us check model performance after each epoch.
# It is not used to update model weights.

val_df = df[
    # Use the existing validation split to decide which epoch's weights to save.
    (df.split == "val") &
    (df.method.isin(["original", args.train_method]))
]


# ============================================================
# 9. DRY RUN MODE
# ============================================================
# Dry run is used to test whether the complete pipeline works.
#
# Instead of training on the entire dataset, we randomly
# select at most 64 training images and 64 validation images.
#
# We also reduce training to just one epoch.

if args.dry_run:

    train_df = train_df.sample(
        min(64, len(train_df)),
        random_state=args.seed
    )

    val_df = val_df.sample(
        min(64, len(val_df)),
        random_state=args.seed
    )

    args.epochs = 1


# ============================================================
# 10. LOAD THE PRETRAINED MODEL
# ============================================================
# We use EfficientNet-B0 as our image classification model.
#
# pretrained=True:
# Loads weights that were already learned from ImageNet.
#
# num_classes=1:
# We need only one output value because this is a binary
# classification problem (Real vs Fake).
#
# The output is a raw score called a LOGIT.
# We later convert it into a probability using sigmoid.

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=1
).to(dev)


# ============================================================
# 11. HANDLE CLASS IMBALANCE
# ============================================================
# Sometimes the dataset contains more real images than fake
# images, or vice versa.
#
# If one class has significantly more images, the model
# might become biased toward that class.
#
# To reduce this problem, we calculate a positive class weight.

# Count the fake labels in the manifest before on-the-fly blending.
n_pos = (train_df.label == 1).sum()
# Count the real labels from the same filtered training table.
n_neg = (train_df.label == 0).sum()

# Give more importance to the fake class if there are
# fewer fake samples than real samples.
#
# pos_weight = Number of real samples / Number of fake samples

loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(n_neg / n_pos, device=dev)
)


# ============================================================
# 12. OPTIMIZER AND MIXED PRECISION
# ============================================================
# AdamW is the optimizer responsible for updating
# the model's learnable weights.
#
# Learning rate determines the size of each update.
# Weight decay helps reduce overfitting.

opt = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-4
)


# GradScaler is used for mixed-precision training.
# It helps maintain numerical stability when using FP16.
#
# This can reduce GPU memory usage and improve speed.

scaler = torch.cuda.amp.GradScaler()


# ============================================================
# 13. PREDICTION FUNCTION
# ============================================================
# This function takes a dataframe of images and generates
# fake probabilities for every image.
#
# torch.no_grad():
# Disables gradient calculations because we are only
# making predictions, not training the model.
#
# model.eval():
# Switches the model into evaluation mode.
#
# sigmoid():
# Converts raw model outputs (logits) into probabilities
# between 0 and 1.
#
# A score closer to 1 means the model thinks the image
# is fake, while a score closer to 0 means it thinks
# the image is real.

# Disable gradient tracking throughout prediction to save memory.
@torch.no_grad()
def predict(d):

    model.eval()

    out = []

    # Keep batches in row order so scores line up with the original table. Ignore labels here.
    for x, _ in loader(d, eval_tf, False):

        # Use FP16 autocasting for faster inference on GPU.
        with torch.autocast("cuda", dtype=torch.float16):

            # Move images to GPU and get model predictions.
            scores = model(x.to(dev)).squeeze(1)

            # Convert logits into probabilities.
            probs = torch.sigmoid(scores)

            # Move predictions back to CPU.
            out.append(probs.float().cpu())

    # Combine predictions from all batches into one array.
    return torch.cat(out).numpy()


# ============================================================
# 14. EQUAL ERROR RATE (EER) FUNCTION
# ============================================================
# EER stands for Equal Error Rate.
#
# It is a common metric in face manipulation detection
# and biometric verification tasks.
#
# FPR = False Positive Rate
#     Real images incorrectly classified as fake.
#
# FNR = False Negative Rate
#     Fake images incorrectly classified as real.
#
# EER is the error rate at the point where FPR and FNR
# are approximately equal.
#
# Lower EER generally indicates better performance.

def eer(y, s):

    # Calculate False Positive Rate and True Positive Rate
    # at different classification thresholds.
    fpr, tpr, _ = roc_curve(y, s)

    # False Negative Rate = 1 - True Positive Rate.
    fnr = 1 - tpr

    # Find the threshold where FPR and FNR are closest.
    # Pick the index where the false-positive and false-negative rates are closest, ignoring NaNs.
    i = np.nanargmin(np.abs(fnr - fpr))

    # Return the average of the two error rates.
    return (fpr[i] + fnr[i]) / 2


# ============================================================
# 15. CREATE OUTPUT DIRECTORY
# ============================================================
# Every experiment gets its own output folder.
#
# Example:
# /kaggle/working/runs/Deepfakes_seed42/
#
# This prevents results from different experiments
# from being mixed together.

# Use the baseline output location unless augmentation is enabled.
run_dir = f"/kaggle/working/runs/{args.train_method}_seed{args.seed}"

# Import the helper only for augmented runs so baseline runs can start without it.
if args.augment:
    mode = "augmented-dry-run" if args.dry_run else "augmented"
    run_dir = f"/kaggle/working/results/{mode}/runs/{args.train_method}_seed{args.seed}"

os.makedirs(run_dir, exist_ok=True)


# ============================================================
# 16. MODEL TRAINING
# ============================================================
# This is the main training loop.
#
# The model will:
# 1. Read a batch of images.
# 2. Predict whether each image is real or fake.
# 3. Calculate prediction error (loss).
# 4. Calculate gradients using backpropagation.
# 5. Update its weights using AdamW.
# 6. Evaluate performance on validation data.
#
# This process repeats for the specified number of epochs.

# Start below all valid AUC values so the first valid epoch can become the best.
best_auc = -1
# This variable will hold a snapshot of the best model weights.
best_state = None

for ep in range(args.epochs):

    # Put the model into training mode.
    model.train()

    for x, y in loader(train_df, train_tf, True):

        # Move images and labels to GPU.
        # Move input images to the same GPU as the model.
        x = x.to(dev)
        # Move their target labels to the GPU as well.
        y = y.to(dev)

        # Mixed-precision forward pass.
        with torch.autocast("cuda", dtype=torch.float16):

            # Get raw predictions from the model.
            logits = model(x).squeeze(1)

            # Calculate loss by comparing predictions
            # with the actual labels.
            loss = loss_fn(logits, y)

        # Clear gradients from the previous batch.
        opt.zero_grad()

        # Calculate gradients using backpropagation.
        scaler.scale(loss).backward()

        # Update model weights using the optimizer.
        scaler.step(opt)

        # Update the gradient scaler.
        scaler.update()


    # --------------------------------------------------------
    # VALIDATION AFTER EACH EPOCH
    # --------------------------------------------------------
    # After completing one epoch, evaluate the model
    # on validation images.
    #
    # AUC measures how well the model separates real and
    # fake images across different classification thresholds.
    #
    # Higher AUC is better.
    # AUC = 1.0 means perfect separation.
    # AUC = 0.5 means random guessing.

    v_auc = roc_auc_score(
        val_df.label,
        predict(val_df)
    )

    print(
        f"epoch {ep+1} "
        # Report the last batch's loss for this epoch.
        f"loss {loss.item():.4f} "
        f"val_auc {v_auc:.4f}"
    )


    # --------------------------------------------------------
    # SAVE THE BEST MODEL
    # --------------------------------------------------------
    # We compare the current validation AUC with the
    # best validation AUC achieved so far.
    #
    # If the current model performs better, we save its weights.
    #
    # This ensures that we do not necessarily keep the weights
    # from the last epoch, but instead keep the best-performing
    # model according to validation AUC.

    if v_auc > best_auc:

        best_auc = v_auc

        best_state = {
            # Copy each parameter so later optimizer steps cannot change the saved snapshot.
            k: t.detach().clone()
            for k, t in model.state_dict().items()
        }


# ============================================================
# 17. LOAD AND SAVE THE BEST MODEL
# ============================================================
# After training, load the weights that achieved
# the highest validation AUC.
#
# Save these weights as best.pt so that the trained model
# can be reused later without training again.

model.load_state_dict(best_state)

torch.save(
    best_state,
    f"{run_dir}/best.pt"
)


# ============================================================
# 18. TEST THE MODEL ON ALL FAKE GENERATION METHODS
# ============================================================
# Now we evaluate the trained model on the test dataset.
#
# IMPORTANT:
# The model was trained using only:
#   Real images + One selected fake method.
#
# But here we test it on all five fake methods.
#
# This experiment helps measure GENERALIZATION.
#
# Generalization means checking whether the model can
# recognize fake images generated by methods that it
# may not have encountered during training.

results = []
preds = []

for tm in TEST_METHODS:

    # Select test images containing:
    # 1. Original (real) images.
    # 2. Fake images from the current test method.
    #
    # For example, when tm = "FaceSwap":
    # We select real images + FaceSwap images.

    t = df[
        (df.split == "test") &
        (df.method.isin(["original", tm]))
    ].copy()

    # Generate fake probabilities for all selected images.
    # Score the test crops using the selected validation checkpoint.
    t["score"] = predict(t)

    # Store the name of the fake method being tested.
    # Tag the rows with this comparison, including the real crops reused across comparisons.
    t["test_method"] = tm

    # Save image paths, labels, predictions, and test method.
    preds.append(
        t[
            ["path", "method", "label", "score", "test_method"]
        ]
    )


    # --------------------------------------------------------
    # CALCULATE EVALUATION METRICS
    # --------------------------------------------------------
    # AUC:
    # Measures how well the model distinguishes real and fake.
    # Higher is better.
    #
    # EER:
    # Measures the error rate where false positives and
    # false negatives are approximately equal.
    # Lower is better.
    #
    # n_pos:
    # Number of fake images.
    #
    # n_neg:
    # Number of real images.

    results.append(
        dict(

            # Fake method used to train this model.
            train_method=args.train_method,

            # Fake method used during the current test.
            test_method=tm,

            # Random seed used in this experiment.
            seed=args.seed,

            # Number of training epochs.
            epochs=args.epochs,

            # Area Under the ROC Curve.
            auc=round(
                roc_auc_score(t.label, t.score),
                4
            ),

            # Equal Error Rate.
            eer=round(
                eer(t.label, t.score),
                4
            ),

            # Total number of fake images.
            n_pos=int((t.label == 1).sum()),

            # Total number of real images.
            n_neg=int((t.label == 0).sum())
        )
    )

    # Print results for the current test method.
    print(results[-1])


# ============================================================
# 19. SAVE ALL PREDICTIONS
# ============================================================
# Save the prediction score for every test image.
#
# This file can be used later for:
# - Further analysis.
# - Plotting ROC curves.
# - Comparing different training methods.
# - Studying individual image predictions.
#
# score is the predicted probability that an image is fake.

pd.concat(preds).to_csv(
    f"{run_dir}/predictions.csv",
    # Write the CSV without the pandas row index.
    index=False
)


# ============================================================
# 20. SAVE FINAL EVALUATION RESULTS
# ============================================================
# Save the evaluation metrics (AUC, EER, etc.)
# for every tested fake generation method.
#
# Each row represents one test method.

pd.DataFrame(results).to_csv(
    f"{run_dir}/results.csv",
    # Write the CSV without the pandas row index.
    index=False
)


# ============================================================
# 21. TRAINING COMPLETED
# ============================================================
# All model weights and evaluation results have now
# been saved inside the experiment folder.

# Display the run folder containing the checkpoint and both CSVs.
print("saved", run_dir)
