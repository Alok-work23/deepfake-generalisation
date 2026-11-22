
"""Training-only face augmentation and a private visual review grid."""

import argparse
import csv
import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


def blend_real(image, rng):
    """Feather a shifted, scaled copy of a real crop over the same crop."""
    width, height = image.size
    scale = rng.uniform(0.90, 1.10)
    source = image.transform(
        image.size,
        Image.Transform.AFFINE,
        (
            1 / scale, 0, width * (1 - 1 / scale) / 2 + rng.uniform(-0.035, 0.035) * width,
            0, 1 / scale, height * (1 - 1 / scale) / 2 + rng.uniform(-0.035, 0.035) * height,
        ),
        resample=Image.Resampling.BICUBIC,
    )
    mask = Image.new("L", image.size)
    ImageDraw.Draw(mask).ellipse(
        (0.22 * width, 0.19 * height, 0.78 * width, 0.82 * height),
        fill=rng.randint(180, 255),
    )
    mask = mask.filter(ImageFilter.GaussianBlur(0.035 * min(width, height)))
    return Image.composite(source, image, mask)


def augment_training_image(image, label, rng=random, *, force_blend=False):
    """Generate pseudo-fakes from real crops and degrade images of either class."""
    if label == 0 and (force_blend or rng.random() < 0.25):
        image = blend_real(image, rng)
        label = 1.0

    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.5:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=rng.randint(45, 95))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                image = decoded.convert("RGB")
    return image, label


def save_review_grid(manifest, output, seed):
    rng = random.Random(seed)
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        originals = [
            row for row in csv.DictReader(handle)
            if row["split"] == "train" and row["method"] == "original" and row["label"] == "0"
        ]
    by_video = {}
    for row in originals:
        by_video.setdefault(row["video_id"], []).append(row["path"])
    if len(by_video) < 20:
        raise ValueError("The review grid needs at least 20 distinct real training videos.")

    tile, caption = 112, 20
    grid = Image.new("RGB", (8 * tile, 5 * (tile + caption)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()
    for index, video in enumerate(rng.sample(sorted(by_video), 20)):
        path = manifest.parent / rng.choice(by_video[video])
        with Image.open(path) as crop:
            real = crop.convert("RGB")
        fake, _ = augment_training_image(real, 0.0, rng, force_blend=True)
        x = index % 4 * 2 * tile
        y = index // 4 * (tile + caption)
        for offset, image, name in ((0, real, "real"), (tile, fake, "augmented")):
            grid.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (x + offset, y))
            draw.text((x + offset + 4, y + tile + 3), f"{index + 1:02d} {name}", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save a private 20-pair augmentation review PNG.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("figures/augmentation_review.png"))
    parser.add_argument("--seed", type=int, default=42)
    options = parser.parse_args()
    save_review_grid(options.manifest, options.output, options.seed)
