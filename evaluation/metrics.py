# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# glob:
# Used to find multiple folders or files matching a pattern.
# Example: Find all experiment folders inside results/runs/.

import glob

# os:
# Used to work with folders, file paths, and directories.
import os

# matplotlib:
# Used to create graphs and heatmaps.
import matplotlib

# Agg is a non-interactive backend.
# This means graphs are saved directly as image files
# without opening a separate window.
#
# This is useful when running Python on Kaggle, servers,
# or machines without a graphical display.

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# numpy:
# Used for mathematical operations such as calculating averages.
import numpy as np

# pandas:
# Used to read CSV files, organize data, and create tables.
import pandas as pd

# These functions are used to calculate classification metrics.
from sklearn.metrics import roc_auc_score, roc_curve


# ============================================================
# 2. DEFINE PROJECT FOLDER PATHS
# ============================================================

# Folder containing baseline experiment results.
#
# Expected structure:
# results/
#     baseline/
#         baseline.csv
#         runs/
#             Deepfakes_seed42/
#             Face2Face_seed42/
#             ...

BASE_DIR = "results/baseline"


# Folder containing augmented experiment results.
#
# Expected structure:
# results/
#     augmented/
#         augmented.csv
#         runs/
#             Deepfakes_seed42/
#             Face2Face_seed42/
#             ...

AUG_DIR = "results/augmented"


# Folder where all generated heatmaps will be saved.
FIG_DIR = "figures"


# Main results folder.
# The final comparison.csv file will be saved here.

RES_DIR = "results"


# ============================================================
# 3. CALCULATE EQUAL ERROR RATE (EER)
# ============================================================

def eer(y, s):

    """
    Calculate Equal Error Rate (EER).

    PARAMETERS:

    y = Actual labels of the images.
        0 = Real image
        1 = Fake image

    s = Predicted fake probabilities from the model.

    WHAT IS EER?

    EER stands for Equal Error Rate.

    It is a metric used to evaluate how well a model
    distinguishes between two classes.

    There are two types of classification errors:

    1. FPR (False Positive Rate):
       Real images incorrectly predicted as fake.

    2. FNR (False Negative Rate):
       Fake images incorrectly predicted as real.

    EER is the error rate at the point where FPR
    and FNR are approximately equal.

    IMPORTANT:

    Lower EER means better model performance.
    """

    # Calculate ROC curve values.
    #
    # FPR = False Positive Rate.
    #
    # TPR = True Positive Rate.
    #
    # Threshold = Different probability cutoffs used
    # to classify images as real or fake.

    fpr, tpr, _ = roc_curve(y, s)

    # Calculate False Negative Rate.
    #
    # TPR tells us how many fake images were correctly detected.
    #
    # FNR tells us how many fake images were missed.
    #
    # Formula:
    # FNR = 1 - TPR

    fnr = 1 - tpr

    # Find the position where FPR and FNR are closest.
    #
    # np.abs() calculates the absolute difference.
    #
    # np.nanargmin() finds the index with the smallest
    # difference while ignoring NaN values.

    i = np.nanargmin(np.abs(fnr - fpr))

    # EER is calculated as the average of FPR and FNR
    # at the selected position.

    return (fpr[i] + fnr[i]) / 2


# ============================================================
# 4. RECOMPUTE METRICS FOR ONE EXPERIMENT
# ============================================================

def recompute_run(run_dir):

    """
    Recalculate AUC and EER using predictions.csv.

    WHY DO WE NEED THIS?

    During training, we already calculated AUC and EER
    and saved them inside results.csv.

    Here, we calculate those metrics AGAIN using the
    individual image predictions.

    This helps us verify that the saved evaluation
    results are correct.

    PARAMETER:

    run_dir = Path of one experiment folder.

    Example:
    results/baseline/runs/Deepfakes_seed42/

    INPUT FILE:

    predictions.csv

    OUTPUT:

    A pandas DataFrame containing:

    - Training method
    - Testing method
    - AUC
    - EER
    - Number of fake images
    - Number of real images
    """

    # --------------------------------------------------------
    # STEP 1: EXTRACT TRAINING METHOD FROM FOLDER NAME
    # --------------------------------------------------------

    # Example folder name:
    # Deepfakes_seed42
    #
    # We remove the "_seed42" part.
    #
    # Final result:
    # Deepfakes

    train_method = os.path.basename(
        run_dir.rstrip("/\\")
    ).split("_seed")[0]


    # --------------------------------------------------------
    # STEP 2: READ PREDICTIONS FILE
    # --------------------------------------------------------

    # Load the predictions generated by train.py.
    #
    # Each row represents one test image.
    #
    # Important columns:
    #
    # path        = Image file location.
    # method      = Actual image generation method.
    # label       = Actual label (0 = Real, 1 = Fake).
    # score       = Predicted probability of being fake.
    # test_method = Fake method currently being evaluated.

    p = pd.read_csv(
        os.path.join(run_dir, "predictions.csv")
    )


    # Create an empty list to store calculated metrics.
    rows = []


    # --------------------------------------------------------
    # STEP 3: GROUP PREDICTIONS BY TEST METHOD
    # --------------------------------------------------------

    # Suppose predictions.csv contains images from:
    #
    # Deepfakes
    # Face2Face
    # FaceSwap
    # NeuralTextures
    # FaceShifter
    #
    # groupby() separates the predictions into groups
    # based on test_method.
    #
    # We calculate AUC and EER separately for each group.

    for test_method, g in p.groupby("test_method"):


        # ----------------------------------------------------
        # STEP 4: CALCULATE METRICS FOR CURRENT TEST METHOD
        # ----------------------------------------------------

        rows.append(dict(

            # Name of the fake method used during training.
            train_method=train_method,

            # Name of the fake method used during testing.
            test_method=test_method,

            # AUC (Area Under the ROC Curve).
            #
            # Measures how well the model separates
            # real images from fake images.
            #
            # Higher AUC = Better performance.
            #
            # 1.0 = Perfect classification.
            # 0.5 = Random guessing.

            auc=round(
                roc_auc_score(g.label, g.score),
                4
            ),

            # Equal Error Rate.
            #
            # Lower EER = Better performance.

            eer=round(
                eer(g.label, g.score),
                4
            ),

            # Count the number of fake images.
            #
            # label == 1 means fake.

            n_pos=int(
                (g.label == 1).sum()
            ),

            # Count the number of real images.
            #
            # label == 0 means real.

            n_neg=int(
                (g.label == 0).sum()
            ),

        ))


    # Convert the list of dictionaries into a DataFrame.
    #
    # This gives us a clean table containing
    # the recalculated evaluation metrics.

    return pd.DataFrame(rows)


# ============================================================
# 5. VERIFY SAVED RESULTS
# ============================================================

def verify(results_dir):

    """
    Check whether saved AUC and EER values are correct.

    We compare two things:

    1. Recalculated AUC and EER from predictions.csv.
    2. Previously saved AUC and EER from results.csv.

    WHY?

    This is a verification step.

    It helps us detect possible mistakes in metric
    calculations or result saving.

    PARAMETER:

    results_dir = Baseline or augmented results directory.

    Example:
    results/baseline
    results/augmented
    """

    print(f"\nChecking {results_dir}")


    # --------------------------------------------------------
    # STEP 1: FIND ALL EXPERIMENT FOLDERS
    # --------------------------------------------------------

    # glob searches for every folder inside:
    #
    # results/baseline/runs/
    #
    # Example:
    # Deepfakes_seed42
    # Face2Face_seed42
    # FaceSwap_seed42
    #
    # sorted() keeps the folders in alphabetical order.

    for run_dir in sorted(
        glob.glob(
            os.path.join(results_dir, "runs", "*")
        )
    ):


        # ----------------------------------------------------
        # STEP 2: RECALCULATE METRICS
        # ----------------------------------------------------

        # Calculate AUC and EER again from predictions.csv.

        re_df = recompute_run(run_dir)


        # ----------------------------------------------------
        # STEP 3: LOAD PREVIOUSLY SAVED METRICS
        # ----------------------------------------------------

        # Read the original results.csv generated during training.

        stored = pd.read_csv(
            os.path.join(run_dir, "results.csv")
        )


        # ----------------------------------------------------
        # STEP 4: MATCH BOTH TABLES
        # ----------------------------------------------------

        # Merge the recalculated results and saved results.
        #
        # train_method and test_method are used as matching
        # columns.
        #
        # suffixes:
        #
        # _re    = Recalculated metric.
        # _saved = Previously saved metric.
        #
        # Example:
        # auc_re
        # auc_saved
        #
        # This allows us to compare both values.

        m = re_df.merge(
            stored,
            on=["train_method", "test_method"],
            suffixes=("_re", "_saved")
        )


        # ----------------------------------------------------
        # STEP 5: CALCULATE DIFFERENCES
        # ----------------------------------------------------

        # Calculate the absolute difference between
        # recalculated AUC and saved AUC.
        #
        # max() finds the largest difference across
        # all test methods in this experiment.

        auc_gap = (
            m.auc_re - m.auc_saved
        ).abs().max()


        # Do the same for EER.

        eer_gap = (
            m.eer_re - m.eer_saved
        ).abs().max()


        # ----------------------------------------------------
        # STEP 6: CHECK WHETHER RESULTS MATCH
        # ----------------------------------------------------

        # If both differences are smaller than 0.001,
        # we consider the results correct.
        #
        # Otherwise, there is a mismatch.
        #
        # 0.001 is used as a tolerance because metrics
        # are rounded to four decimal places.

        status = (
            "OK"
            if auc_gap < 0.001 and eer_gap < 0.001
            else "MISMATCH"
        )


        # Print the verification report for this experiment.

        print(
            f"  {os.path.basename(run_dir):25s} "
            f"rows={len(m)}  "
            f"max AUC gap={auc_gap:.4f}  "
            f"max EER gap={eer_gap:.4f}  "
            f"{status}"
        )


# ============================================================
# 6. CHECK SUMMARY CSV FILES
# ============================================================

def check_summary(path, expected_rows=20):

    """
    Load and inspect a summary CSV file.

    The summary file contains the final evaluation results
    for multiple training methods and testing methods.

    Example:

    baseline.csv
    augmented.csv

    We expect 20 rows because:

    5 training methods x 5 testing methods = 25 rows,
    if every possible combination is included.

    NOTE:
    The default expected_rows=20 in this script is
    preserved from the original code. It is only a
    reporting check and does not validate the dataset.

    PARAMETER:

    path = Location of the summary CSV file.

    expected_rows = Expected number of rows.

    RETURNS:

    The loaded pandas DataFrame.
    """


    # Read the summary CSV file.

    df = pd.read_csv(path)


    # Print how many rows are present.
    #
    # This helps us check whether the summary file
    # contains the expected amount of data.

    print(
        f"\n{path}: {len(df)} rows "
        f"(expected {expected_rows})"
    )


    # Print all random seeds used in the experiments.
    #
    # unique() removes repeated values.
    # sorted() arranges the values in ascending order.

    print(
        "  seeds:",
        sorted(df.seed.unique()),
        " epochs:",
        sorted(df.epochs.unique())
    )


    # Return the loaded dataframe so that we can
    # use it later to create matrices and comparisons.

    return df


# ============================================================
# 7. CREATE HEATMAPS
# ============================================================

def heatmap(
    table,
    title,
    out_path,
    cmap="viridis",
    vmin=0,
    vmax=1,
    label="AUC",
    fmt="{:.3f}"
):

    """
    Create and save a heatmap from a performance matrix.

    WHAT IS A HEATMAP?

    A heatmap represents numerical values using colors.

    In our project:

    Rows    = Training methods.
    Columns = Testing methods.
    Cells   = AUC or EER values.

    For example:

                    Test Methods
                  Deepfakes FaceSwap
    Training
    Deepfakes       0.95     0.80
    FaceSwap        0.75     0.94

    This allows us to quickly understand which training
    and testing combinations perform well.

    PARAMETERS:

    table:
        Matrix containing AUC, EER, or performance changes.

    title:
        Title displayed above the heatmap.

    out_path:
        File location where the heatmap will be saved.

    cmap:
        Color scheme used for the heatmap.

    vmin:
        Minimum value represented by the color scale.

    vmax:
        Maximum value represented by the color scale.

    label:
        Label displayed next to the color bar.

    fmt:
        Number formatting for values written inside cells.
    """


    # --------------------------------------------------------
    # STEP 1: CREATE THE FIGURE
    # --------------------------------------------------------

    # figsize controls the width and height of the figure.
    #
    # fig = Entire figure.
    # ax  = Area where the heatmap will be drawn.

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    # --------------------------------------------------------
    # STEP 2: DRAW THE HEATMAP
    # --------------------------------------------------------

    # imshow() converts our numerical matrix into colors.
    #
    # cmap controls the color scheme.
    #
    # vmin and vmax keep the color scale consistent.
    #
    # This is important when comparing multiple heatmaps.

    im = ax.imshow(
        table.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )


    # --------------------------------------------------------
    # STEP 3: SET X-AXIS LABELS
    # --------------------------------------------------------

    # X-axis represents the testing methods.
    #
    # range(table.shape[1]) creates positions
    # for all columns.
    #
    # rotation=30 rotates labels so they do not overlap.

    ax.set_xticks(
        range(table.shape[1])
    )

    ax.set_xticklabels(
        table.columns,
        rotation=30,
        ha="right"
    )


    # --------------------------------------------------------
    # STEP 4: SET Y-AXIS LABELS
    # --------------------------------------------------------

    # Y-axis represents the training methods.
    #
    # Each row corresponds to one training method.

    ax.set_yticks(
        range(table.shape[0])
    )

    ax.set_yticklabels(
        table.index
    )


    # --------------------------------------------------------
    # STEP 5: WRITE NUMERICAL VALUES INSIDE EACH CELL
    # --------------------------------------------------------

    # Instead of showing only colors, we also write
    # the actual numerical values inside the heatmap.
    #
    # Example:
    # 0.923
    # 0.845
    # 0.761
    #
    # This makes it easier to compare exact performance.

    for i in range(table.shape[0]):

        for j in range(table.shape[1]):

            # Get the numerical value at row i, column j.

            v = table.values[i, j]


            # Write the value in the center of the cell.
            #
            # fmt controls the number of decimal places.
            #
            # For AUC:
            # 0.92345 becomes 0.923.
            #
            # For difference:
            # +0.023 or -0.023.

            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",

                # Change text color depending on the
                # background intensity.
                #
                # This helps keep numbers readable.

                color=(
                    "white"
                    if abs(v - (vmin + vmax) / 2)
                    > (vmax - vmin) * 0.3
                    else "black"
                )
            )


    # --------------------------------------------------------
    # STEP 6: ADD AXIS LABELS AND TITLE
    # --------------------------------------------------------

    # Horizontal axis = Testing method.
    ax.set_xlabel("Test method")

    # Vertical axis = Training method.
    ax.set_ylabel("Training method")

    # Title displayed above the heatmap.
    ax.set_title(title)


    # --------------------------------------------------------
    # STEP 7: ADD COLOR BAR
    # --------------------------------------------------------

    # Color bar explains what numerical values
    # the different colors represent.
    #
    # For example:
    # Darker color = Lower value.
    # Brighter color = Higher value.
    #
    # Exact meaning depends on the selected color map.

    fig.colorbar(
        im,
        ax=ax,
        label=label
    )


    # --------------------------------------------------------
    # STEP 8: SAVE THE HEATMAP
    # --------------------------------------------------------

    # tight_layout() adjusts spacing to prevent
    # labels from being cut off.

    fig.tight_layout()


    # Save the heatmap as a PNG image.
    #
    # dpi=150 controls image quality.

    fig.savefig(
        out_path,
        dpi=150
    )


    # Close the figure to release memory.
    #
    # Important when generating many heatmaps.

    plt.close(fig)


    # Print the location where the image was saved.

    print("saved", out_path)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

# This condition means:
#
# Execute the following code only when this file
# is run directly.
#
# It will not automatically execute if metrics.py
# is imported into another Python script.

if __name__ == "__main__":


    # ========================================================
    # STEP 1: CREATE FIGURES DIRECTORY
    # ========================================================

    # Create the figures folder if it does not already exist.
    #
    # exist_ok=True prevents an error if the folder
    # already exists.

    os.makedirs(
        FIG_DIR,
        exist_ok=True
    )


    # ========================================================
    # STEP 2: VERIFY BASELINE AND AUGMENTED RESULTS
    # ========================================================

    # First verify the baseline experiments.
    #
    # This recalculates AUC and EER from predictions.csv
    # and compares them with the saved results.csv.

    verify(BASE_DIR)


    # Now perform the same verification for augmented experiments.

    verify(AUG_DIR)


    # ========================================================
    # STEP 3: LOAD SUMMARY FILES
    # ========================================================

    # Load baseline summary results.
    #
    # This file contains the final baseline AUC and EER
    # for each training and testing combination.

    base = check_summary(
        os.path.join(BASE_DIR, "baseline.csv")
    )


    # Load augmented summary results.
    #
    # This file contains the final augmented AUC and EER.

    aug = check_summary(
        os.path.join(AUG_DIR, "augmented.csv")
    )


    # ========================================================
    # STEP 4: CREATE PERFORMANCE MATRICES
    # ========================================================

    # WHAT IS A MATRIX HERE?
    #
    # A matrix is simply a table where:
    #
    # Rows    = Training methods.
    # Columns = Testing methods.
    # Values  = Evaluation metrics.
    #
    # We create separate matrices for:
    #
    # 1. Baseline AUC.
    # 2. Augmented AUC.
    # 3. Baseline EER.
    # 4. Augmented EER.
    #
    # pivot() converts a normal table into this
    # row-and-column structure.


    # --------------------------------------------------------
    # BASELINE AUC MATRIX
    # --------------------------------------------------------

    # index="train_method":
    # Training methods become the rows.
    #
    # columns="test_method":
    # Testing methods become the columns.
    #
    # values="auc":
    # Each cell contains the AUC value.

    base_auc = base.pivot(
        index="train_method",
        columns="test_method",
        values="auc"
    )


    # --------------------------------------------------------
    # AUGMENTED AUC MATRIX
    # --------------------------------------------------------

    # Same structure, but using augmented results.

    aug_auc = aug.pivot(
        index="train_method",
        columns="test_method",
        values="auc"
    )


    # --------------------------------------------------------
    # BASELINE EER MATRIX
    # --------------------------------------------------------

    # Create a matrix containing baseline EER values.

    base_eer = base.pivot(
        index="train_method",
        columns="test_method",
        values="eer"
    )


    # --------------------------------------------------------
    # AUGMENTED EER MATRIX
    # --------------------------------------------------------

    # Create a matrix containing augmented EER values.

    aug_eer = aug.pivot(
        index="train_method",
        columns="test_method",
        values="eer"
    )


    # ========================================================
    # STEP 5: PRINT AUC MATRICES
    # ========================================================

    # round(3) displays values up to three decimal places.
    #
    # Printing these matrices allows us to inspect
    # numerical results directly in the terminal.

    print(
        "\nBaseline AUC\n",
        base_auc.round(3)
    )

    print(
        "\nAugmented AUC\n",
        aug_auc.round(3)
    )


    # ========================================================
    # STEP 6: GENERATE HEATMAPS
    # ========================================================

    # --------------------------------------------------------
    # HEATMAP 1: BASELINE AUC
    # --------------------------------------------------------

    # Visualizes the performance of the baseline model.
    #
    # Each cell represents AUC for one training/testing pair.

    heatmap(
        base_auc,
        "Baseline: AUC by training and test method",
        os.path.join(FIG_DIR, "baseline_auc.png")
    )


    # --------------------------------------------------------
    # HEATMAP 2: AUGMENTED AUC
    # --------------------------------------------------------

    # Visualizes the performance of the augmented model.
    #
    # This allows us to see how augmentation affects
    # the model's ability to detect different fake methods.

    heatmap(
        aug_auc,
        "Augmented: AUC by training and test method",
        os.path.join(FIG_DIR, "augmented_auc.png")
    )


    # --------------------------------------------------------
    # HEATMAP 3: AUC DIFFERENCE
    # --------------------------------------------------------

    # Calculate the difference between augmented and baseline AUC.
    #
    # Formula:
    #
    # AUC Difference = Augmented AUC - Baseline AUC
    #
    # Example:
    #
    # Baseline AUC = 0.80
    # Augmented AUC = 0.90
    #
    # Difference = 0.90 - 0.80 = +0.10
    #
    # Positive value:
    # Augmented model performs better.
    #
    # Negative value:
    # Baseline model performs better.
    #
    # Zero:
    # Both models perform equally.

    diff_auc = aug_auc - base_auc


    # Generate a heatmap showing these differences.
    #
    # RdBu:
    # Red-Blue color scheme.
    #
    # vmin=-0.3:
    # Minimum color scale value.
    #
    # vmax=0.3:
    # Maximum color scale value.
    #
    # Center value is zero.
    #
    # Positive and negative changes can therefore
    # be visually compared.

    heatmap(
        diff_auc,
        "Change in AUC (augmented minus baseline)",
        os.path.join(FIG_DIR, "auc_difference.png"),
        cmap="RdBu",
        vmin=-0.3,
        vmax=0.3,
        label="Change in AUC",
        fmt="{:+.3f}"
    )


    # ========================================================
    # STEP 7: CREATE BASELINE VS AUGMENTED COMPARISON TABLE
    # ========================================================

    # We want to compare baseline and augmented performance
    # for every training and testing combination.
    #
    # First, select the columns we actually need.

    comp = base[
        ["train_method", "test_method", "auc", "eer"]
    ].merge(

        aug[
            ["train_method", "test_method", "auc", "eer"]
        ],

        # Match rows using the same training and testing methods.
        #
        # This ensures that we compare the same experiment
        # between baseline and augmented results.

        on=["train_method", "test_method"],

        # Add suffixes to distinguish baseline and augmented.
        #
        # auc_base = Baseline AUC.
        # auc_aug  = Augmented AUC.
        #
        # eer_base = Baseline EER.
        # eer_aug  = Augmented EER.

        suffixes=("_base", "_aug")
    )


    # --------------------------------------------------------
    # CALCULATE AUC IMPROVEMENT
    # --------------------------------------------------------

    # Calculate how much AUC changed after augmentation.
    #
    # Formula:
    # Augmented AUC - Baseline AUC
    #
    # Positive = Improvement.
    # Negative = Performance decreased.

    comp["auc_change"] = (
        comp.auc_aug - comp.auc_base
    ).round(4)


    # --------------------------------------------------------
    # CALCULATE EER IMPROVEMENT
    # --------------------------------------------------------

    # Calculate how much EER changed after augmentation.
    #
    # Formula:
    # Augmented EER - Baseline EER
    #
    # IMPORTANT:
    #
    # For EER, negative is generally GOOD.
    #
    # Example:
    #
    # Baseline EER = 0.20
    # Augmented EER = 0.10
    #
    # Change = 0.10 - 0.20 = -0.10
    #
    # This means the error rate decreased by 0.10.

    comp["eer_change"] = (
        comp.eer_aug - comp.eer_base
    ).round(4)


    # --------------------------------------------------------
    # SAVE COMPARISON TABLE
    # --------------------------------------------------------

    # Save all comparison values into comparison.csv.
    #
    # This file is useful for:
    # - Creating research paper tables.
    # - Comparing different methods.
    # - Identifying which methods benefit from augmentation.
    # - Performing additional statistical analysis.

    comp.to_csv(
        os.path.join(RES_DIR, "comparison.csv"),
        index=False
    )


    print(
        "\nSaved results/comparison.csv"
    )


    # ========================================================
    # STEP 8: CALCULATE AVERAGE AUC ON UNSEEN METHODS
    # ========================================================

    # WHAT ARE UNSEEN METHODS?
    #
    # In this project, a model is trained on one fake method.
    #
    # For example:
    #
    # Training method = Deepfakes
    #
    # When testing on Deepfakes:
    # This is a SEEN method because it was used during training.
    #
    # When testing on FaceSwap:
    # This is an UNSEEN method because it was not used
    # during training.
    #
    # This is important because we want to know whether
    # our model can detect fake images generated by
    # methods it has never encountered before.
    #
    # This is called CROSS-METHOD GENERALISATION.
    #
    # Off-diagonal cells represent unseen methods.
    #
    # Diagonal cells represent seen methods.


    print(
        "\nMean AUC on UNSEEN methods "
        "(off-diagonal, FaceShifter included):"
    )


    # We calculate this separately for:
    #
    # 1. Baseline AUC matrix.
    # 2. Augmented AUC matrix.

    for name, mat in [
        ("baseline", base_auc),
        ("augmented", aug_auc)
    ]:


        # ----------------------------------------------------
        # CREATE A MASK
        # ----------------------------------------------------

        # A mask is a table of True and False values.
        #
        # True means:
        # Include this cell in the calculation.
        #
        # False means:
        # Exclude this cell from the calculation.
        #
        # Initially, every cell is True.
        #
        # This means all cells are included.

        mask = pd.DataFrame(
            True,
            index=mat.index,
            columns=mat.columns
        )


        # ----------------------------------------------------
        # EXCLUDE DIAGONAL CELLS
        # ----------------------------------------------------

        # The diagonal contains cells where:
        #
        # Training method = Testing method.
        #
        # These are seen methods.
        #
        # We do not want them when calculating
        # unseen-method performance.
        #
        # Therefore, we change diagonal values to False.

        for t in mat.index:

            if t in mat.columns:

                mask.loc[t, t] = False


        # ----------------------------------------------------
        # CALCULATE MEAN AUC FOR UNSEEN METHODS
        # ----------------------------------------------------

        # mat[mask] selects only cells where mask=True.
        #
        # stack() converts the selected matrix cells
        # into a single series of values.
        #
        # mean() calculates their average.
        #
        # The result tells us how well the model performs
        # on fake methods that were not used during training.

        print(
            f"  {name:10s} "
            f"{mat[mask].stack().mean():.3f}"
        )


    # ========================================================
    # STEP 9: CALCULATE AVERAGE AUC ON SEEN METHODS
    # ========================================================

    # Diagonal cells represent SEEN methods.
    #
    # Example:
    #
    # Training = Deepfakes
    # Testing  = Deepfakes
    #
    # The model has already seen this fake generation
    # method during training.
    #
    # We calculate the average AUC across all diagonal cells.
    #
    # This gives us an idea of how well the model performs
    # on fake methods it was trained to recognize.


    print(
        "\nMean AUC on SEEN methods (diagonal):"
    )


    # Calculate the average seen-method AUC separately
    # for baseline and augmented models.

    for name, mat in [
        ("baseline", base_auc),
        ("augmented", aug_auc)
    ]:


        # Extract diagonal values.
        #
        # mat.loc[t, t] selects the cell where
        # training method and testing method are identical.
        #
        # The if condition ensures that the method exists
        # in both the row and column labels.

        diag = [
            mat.loc[t, t]
            for t in mat.index
            if t in mat.columns
        ]


        # Calculate the average of all diagonal AUC values.
        #
        # np.mean() calculates the arithmetic average.

        print(
            f"  {name:10s} "
            f"{np.mean(diag):.3f}"
        )
