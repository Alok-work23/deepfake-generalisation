
"""Training-only face augmentation and a private visual review grid."""

# Builds the command-line interface for generating the review image.
import argparse
# Reads each manifest row as a dictionary keyed by column name.
import csv
# Provides an in-memory file for the JPEG encode/decode step.
import io
# Supplies random choices for the transforms and preview sampling.
import random
# Lets us join file and folder paths with the / operator.
from pathlib import Path

# Pillow opens images, draws masks and captions, and applies blur.
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# Make a pseudo-fake from a single real crop using a modified copy of that crop.
def blend_real(image, rng):
    """Feather a shifted, scaled copy of a real crop over the same crop."""
    # Read the crop dimensions so offsets and mask boundaries scale with the image.
    width, height = image.size
    # Pick a zoom factor between 90% and 110%.
    scale = rng.uniform(0.90, 1.10)
    # Build the shifted and scaled copy on an output canvas the same size as the original.
    source = image.transform(
        image.size,
        # Use an affine transform, which maps output pixel coordinates back to input coordinates.
        Image.Transform.AFFINE,
        (
            # The first three numbers map x. The inverse scale centers the zoom with an added random horizontal shift of up to 3.5% of the width.
            1 / scale, 0, width * (1 - 1 / scale) / 2 + rng.uniform(-0.035, 0.035) * width,
            # The next three map y in the same way, with an independent vertical shift.
            0, 1 / scale, height * (1 - 1 / scale) / 2 + rng.uniform(-0.035, 0.035) * height,
        ),
        # Estimate pixels between input coordinates from their neighbors using bicubic interpolation.
        resample=Image.Resampling.BICUBIC,
    )
    # Start with a black grayscale mask. Black keeps the original image and white selects the transformed copy.
    mask = Image.new("L", image.size)
    # Draw an oval around the center of the face crop, using fractions of its width and height.
    ImageDraw.Draw(mask).ellipse(
        (0.22 * width, 0.19 * height, 0.78 * width, 0.82 * height),
        # Choose how strongly the transformed face shows through the center of the mask.
        fill=rng.randint(180, 255),
    )
    # Blur the mask edges so the two crops blend gradually.
    mask = mask.filter(ImageFilter.GaussianBlur(0.035 * min(width, height)))
    # Use the mask as a per-pixel mixing weight and return the combined image.
    return Image.composite(source, image, mask)


# Return the possibly transformed image together with the label the model should learn.
def augment_training_image(image, label, rng=random, *, force_blend=False):
    """Generate pseudo-fakes from real crops and degrade images of either class."""
    # Only real crops can become pseudo-fakes. Force every preview pair to blend, or blend about one in four real samples during training.
    if label == 0 and (force_blend or rng.random() < 0.25):
        # Make the blended crop before applying the optional image degradations.
        image = blend_real(image, rng)
        # Give the synthetic example the same positive label as a real deepfake example.
        label = 1.0

    # Flip a coin for Gaussian blur. This operates on both real and fake examples.
    if rng.random() < 0.5:
        # Choose a strength from 0.3 to 1.5 pixels so blur varies across samples.
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    # Independently flip a coin for JPEG compression, so either, both, or neither degradation may run.
    if rng.random() < 0.5:
        # Keep the temporary JPEG in memory and close its buffer when done.
        with io.BytesIO() as buffer:
            # Encode at a random JPEG quality from 45 to 95 to introduce varying compression artifacts.
            image.save(buffer, format="JPEG", quality=rng.randint(45, 95))
            # Rewind to the start so Pillow can read the JPEG just written.
            buffer.seek(0)
            # Decode the JPEG while the buffer is still open.
            with Image.open(buffer) as decoded:
                # Make a usable RGB image before leaving the file context.
                image = decoded.convert("RGB")
    # Pass both values back so the dataset can transform the pixels and train with the correct target.
    return image, label


# Create a contact sheet of 20 original crops and their augmented counterparts.
def save_review_grid(manifest, output, seed):
    # Use a separate seeded generator so the review can be repeated with the same inputs.
    rng = random.Random(seed)
    # Read the CSV as text and accept an optional UTF-8 byte order mark.
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        # Keep only original, real training frames. Validation and test crops are excluded.
        originals = [
            row for row in csv.DictReader(handle)
            if row["split"] == "train" and row["method"] == "original" and row["label"] == "0"
        ]
    # Build a lookup from each video ID to its candidate crop paths.
    by_video = {}
    for row in originals:
        # Start an empty list on first sight of a video ID, then append the current frame path.
        by_video.setdefault(row["video_id"], []).append(row["path"])
    # Require 20 distinct source videos so the grid shows one frame from each of 20 videos.
    if len(by_video) < 20:
        raise ValueError("The review grid needs at least 20 distinct real training videos.")

    # Allocate 112 pixels per image and an extra 20 below each for the label.
    tile, caption = 112, 20
    # Fit four pairs across (eight tiles) and five rows down on a white canvas.
    grid = Image.new("RGB", (8 * tile, 5 * (tile + caption)), "white")
    # Draw captions directly onto that canvas.
    draw = ImageDraw.Draw(grid)
    # Use Pillow's built-in font so a separate font file is unnecessary.
    font = ImageFont.load_default()
    # Sort IDs before seeded sampling, and number the chosen videos from zero for layout.
    for index, video in enumerate(rng.sample(sorted(by_video), 20)):
        # Pick one crop from this video and resolve its path relative to the CSV directory.
        path = manifest.parent / rng.choice(by_video[video])
        # Open the crop and close its file after converting it to RGB.
        with Image.open(path) as crop:
            real = crop.convert("RGB")
        # Generate one blended counterpart. The underscore discards the returned label because this grid displays pictures.
        fake, _ = augment_training_image(real, 0.0, rng, force_blend=True)
        # Use the remainder to place the pair in one of four columns.
        x = index % 4 * 2 * tile
        # Use whole-number division to move down a row after every four pairs.
        y = index // 4 * (tile + caption)
        # Put the real crop on the left and its augmented copy one tile to the right.
        for offset, image, name in ((0, real, "real"), (tile, fake, "augmented")):
            # Resize the images for display and place them at the computed positions.
            grid.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x + offset, y))
            # Label each tile with a two-digit pair number and whether it is real or augmented.
            draw.text((x + offset + 4, y + tile + 3), f"{index + 1:02d} {name}", fill="black", font=font)
    # Create any missing output directories and accept directories that already exist.
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write the completed canvas to the requested file.
    grid.save(output)


# Run this command-line code when launched directly. Importing the module only makes its functions available.
if __name__ == "__main__":
    # Create the parser and describe the command in its help output.
    parser = argparse.ArgumentParser(description="Save a private 20-pair augmentation review PNG.")
    # Require the path to frames.csv as the first positional argument.
    parser.add_argument("manifest", type=Path)
    # Use the default PNG path unless a different destination is supplied.
    parser.add_argument("--output", type=Path, default=Path("figures/augmentation_review.png"))
    # Use the default seed 42 unless the caller supplies another integer.
    parser.add_argument("--seed", type=int, default=42)
    # Read the actual command-line arguments.
    options = parser.parse_args()
    # Call the grid builder with the parsed paths and seed.
    save_review_grid(options.manifest, options.output, options.seed)
