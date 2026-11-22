
"""Apply the Member 3 integration to the baseline trainer, then run all four jobs."""

import argparse
import csv
import subprocess
import sys
from pathlib import Path


METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
TEST_METHODS = (*METHODS, "FaceShifter")
REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/kaggle/working/results/augmented/runs")


def integrate(trainer):
    source = trainer.read_text(encoding="utf-8")
    changes = (
        ("args = p.parse_args()", 'p.add_argument("--augment", action="store_true")\n\nargs = p.parse_args()\n\nif args.augment:\n    from augment import augment_training_image'),
        ("def __init__(self, d, tf):", "def __init__(self, d, tf, augment=False):\n        self.augment = augment"),
        ("        return self.tf(img), float(r.label)", "        label = float(r.label)\n        if self.augment:\n            img, label = augment_training_image(img, label)\n        return self.tf(img), label"),
        ("        FrameDS(d, tf),", "        FrameDS(d, tf, augment=args.augment and shuffle),"),
        ('os.makedirs(run_dir, exist_ok=True)', 'if args.augment:\n    mode = "augmented-dry-run" if args.dry_run else "augmented"\n    run_dir = f"/kaggle/working/results/{mode}/runs/{args.train_method}_seed{args.seed}"\n\nos.makedirs(run_dir, exist_ok=True)'),
    )
    if all(new in source for _, new in changes):
        return
    for old, new in changes:
        if source.count(old) != 1:
            raise ValueError(f"Baseline trainer differs at the expected anchor: {old}")
        source = source.replace(old, new)
    compile(source, str(trainer), "exec")
    trainer.write_text(source, encoding="utf-8")


def run_experiments(trainer):
    expected = {(train, test) for train in METHODS for test in TEST_METHODS}
    rows = []
    for method in METHODS:
        subprocess.run(
            [sys.executable, str(trainer), "--train_method", method, "--epochs", "5",
             "--batch", "32", "--lr", "1e-4", "--seed", "42", "--augment"],
            check=True,
        )
        run = RUNS / f"{method}_seed42"
        if not (run / "predictions.csv").stat().st_size:
            raise ValueError(f"Empty predictions for {method}")
        with (run / "results.csv").open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    actual = {(row["train_method"], row["test_method"]) for row in rows}
    if len(rows) != 20 or actual != expected:
        raise ValueError("Expected one result for every train/test pair.")
    if any(row["seed"] != "42" or row["epochs"] != "5" for row in rows):
        raise ValueError("A run used unexpected settings.")
    output = RUNS.parent / "augmented.csv"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Launch the four full Kaggle GPU runs.")
    options = parser.parse_args()
    trainer = REPO / "models" / "train.py"
    integrate(trainer)
    if options.train:
        run_experiments(trainer)
