
"""Apply the Member 3 integration to the baseline trainer, then run all four jobs."""

# Reads the optional --train switch from the command line.
import argparse
# Reads each run's metrics and writes the combined CSV.
import csv
# Launches each training run as a separate Python process.
import subprocess
# Provides the path to the Python interpreter running this launcher.
import sys
# Lets us compose file paths and call file operations on them.
from pathlib import Path


# These four methods each get their own training run.
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
# Unpack the four methods and add FaceShifter, which is only evaluated.
TEST_METHODS = (*METHODS, "FaceShifter")
# Go up from this script to models, then one level further to the repository root.
REPO = Path(__file__).resolve().parent.parent
# Use the Kaggle output directory where the augmented trainer saves individual runs.
RUNS = Path("/kaggle/working/results/augmented/runs")


# Add the augmentation wiring to a baseline copy of train.py.
def integrate(trainer):
    # Read the entire trainer into memory before applying edits.
    source = trainer.read_text(encoding="utf-8")
    # Each pair contains an exact fragment from the baseline followed by its replacement. Keep the strings intact because they also identify an already-integrated trainer.
    changes = (
        # Add a flag that defaults to false and import the augmentation helper only when enabled.
        ("args = p.parse_args()", 'p.add_argument("--augment", action="store_true")\n\nargs = p.parse_args()\n\nif args.augment:\n    from augment import augment_training_image'),
        # Store the augmentation setting on each dataset instance.
        ("def __init__(self, d, tf):", "def __init__(self, d, tf, augment=False):\n        self.augment = augment"),
        # Let augmentation update both the image and label before the usual tensor transform.
        ("        return self.tf(img), float(r.label)", "        label = float(r.label)\n        if self.augment:\n            img, label = augment_training_image(img, label)\n        return self.tf(img), label"),
        # Turn augmentation on only for the shuffled loader used during training.
        ("        FrameDS(d, tf),", "        FrameDS(d, tf, augment=args.augment and shuffle),"),
        # Route augmented runs to their own folders, with separate directories for smoke tests.
        ('os.makedirs(run_dir, exist_ok=True)', 'if args.augment:\n    mode = "augmented-dry-run" if args.dry_run else "augmented"\n    run_dir = f"/kaggle/working/results/{mode}/runs/{args.train_method}_seed{args.seed}"\n\nos.makedirs(run_dir, exist_ok=True)'),
    )
    # If every replacement is already present, leave the trainer file alone.
    if all(new in source for _, new in changes):
        return
    # Look for each baseline fragment in turn before replacing it in memory.
    for old, new in changes:
        # Require exactly one occurrence. A missing or repeated fragment signals that the trainer differs from the expected version.
        if source.count(old) != 1:
            # Stop with the fragment that needs attention before anything is written.
            raise ValueError(f"Baseline trainer differs at the expected anchor: {old}")
        # Apply the verified replacement to the in-memory source.
        source = source.replace(old, new)
    # Check Python syntax without running the trainer.
    compile(source, str(trainer), "exec")
    # Save only after all replacements and the syntax check succeed.
    trainer.write_text(source, encoding="utf-8")


# Run the four experiments in order and collect their metric rows.
def run_experiments(trainer):
    # Build the set of 20 required combinations by pairing each training method with every test method.
    expected = {(train, test) for train in METHODS for test in TEST_METHODS}
    # Accumulate the CSV rows across runs in this list.
    rows = []
    # Run once for each training method with the same baseline settings.
    for method in METHODS:
        # Call the trainer with this Python interpreter. Wait for it to finish before reading its outputs.
        subprocess.run(
            # The first two entries select Python and the script, followed by the method and fixed training arguments.
            [sys.executable, str(trainer), "--train_method", method, "--epochs", "5",
             "--batch", "32", "--lr", "1e-4", "--seed", "42", "--augment"],
            # Raise an error if the trainer exits unsuccessfully so aggregation stops.
            check=True,
        )
        # Find this method's output folder, whose name includes the fixed seed.
        run = RUNS / f"{method}_seed42"
        # Require a nonempty predictions file. A missing file also stops execution.
        if not (run / "predictions.csv").stat().st_size:
            raise ValueError(f"Empty predictions for {method}")
        # Open the per-method metrics file and close it automatically after reading.
        with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
            # Append rows as dictionaries keyed by their CSV headers.
            rows.extend(csv.DictReader(handle))

    # Extract the train/test pair from each result row for a coverage check.
    actual = {(row["train_method"], row["test_method"]) for row in rows}
    # Check both the total row count and the exact pair set. Together these detect missing, repeated, or unexpected pairs.
    if len(rows) != 20 or actual != expected:
        raise ValueError("Expected one result for every train/test pair.")
    # CSV readers return strings. Compare the seed and epoch fields to the expected string values.
    if any(row["seed"] != "42" or row["epochs"] != "5" for row in rows):
        raise ValueError("A run used unexpected settings.")
    # Save augmented.csv beside the runs directory.
    output = RUNS.parent / "augmented.csv"
    # Build a temporary filename next to the final CSV.
    temporary = output.with_suffix(".tmp")
    # Write the complete file to the temporary path before replacing the final copy.
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        # Use the first result row's keys as the CSV column headings in their original order.
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        # Write the headings as the first row.
        writer.writeheader()
        # Write all 20 metric rows.
        writer.writerows(rows)
    # Move the completed temporary file into place, replacing any earlier summary.
    temporary.replace(output)
    # Print the finished summary path so the caller knows where to find it.
    print(output)


# Run the command-line workflow only when this file is executed as a script.
if __name__ == "__main__":
    # Use the module's opening description as the CLI help text.
    parser = argparse.ArgumentParser(description=__doc__)
    # Treat --train as an optional on/off flag. Omitting it applies only the trainer integration.
    parser.add_argument("--train", action="store_true", help="Launch the four full Kaggle GPU runs.")
    # Parse the supplied command-line options.
    options = parser.parse_args()
    # Find train.py relative to this launcher's repository location.
    trainer = REPO / "models" / "train.py"
    # Ensure the trainer understands --augment before starting any runs.
    integrate(trainer)
    # Start the GPU experiments only when explicitly requested.
    if options.train:
        # Launch training, validate the output and create the summary.
        run_experiments(trainer)
