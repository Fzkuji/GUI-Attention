#!/usr/bin/env python3
"""Convert GroundCUA annotations into the repo's bbox-training JSON format.

Expected GroundCUA layout after `huggingface-cli download`:

  GroundCUA/
    data/
      <App Name>/
        <hash>.json
        <hash>.png

Each GroundCUA JSON file contains a list of element annotations for one image.
This script turns those element-level annotations into training samples of the
form consumed by `load_single_dataset()` in `training/sft.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image


TEXT_TEMPLATES = (
    'Click the "{text}" element.',
    'Select the "{text}" option in the interface.',
    'Find and click "{text}".',
    'Click on "{text}".',
    'Locate "{text}" and click it.',
    'Press the "{text}" control.',
)

CATEGORY_TEMPLATES = (
    "Click the {category}.",
    "Select the {category} control.",
    "Locate and click the {category}.",
    "Press the {category}.",
)

TEXT_AND_CATEGORY_TEMPLATES = (
    'Click the "{text}" {category}.',
    'Select the {category} labeled "{text}".',
    'Find the {category} with text "{text}" and click it.',
    'Press the "{text}" {category}.',
)


def _clean_text(value: str | None) -> str:
    if value is None:
        return ""
    value = " ".join(str(value).strip().split())
    return value


def _category_phrase(value: str | None) -> str:
    value = _clean_text(value)
    if not value:
        return "ui element"
    return value.lower()


def _make_instruction(entry: dict, rng: random.Random) -> str:
    text = _clean_text(entry.get("text"))
    category = _category_phrase(entry.get("category"))
    if text and category and category not in {"others", "ui element"}:
        template = rng.choice(TEXT_AND_CATEGORY_TEMPLATES)
        return template.format(text=text, category=category)
    if text:
        template = rng.choice(TEXT_TEMPLATES)
        return template.format(text=text)
    template = rng.choice(CATEGORY_TEMPLATES)
    return template.format(category=category)


def _normalize_bbox(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = map(float, bbox)
    x1 = max(0.0, min(width, x1))
    x2 = max(0.0, min(width, x2))
    y1 = max(0.0, min(height, y1))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 / width, y1 / height, x2 / width, y2 / height]


def _resolve_data_dir(root: Path) -> Path:
    if (root / "data").exists():
        return root / "data"
    if any(root.glob("*/")) and any(root.rglob("*.json")):
        return root
    raise FileNotFoundError(
        f"GroundCUA data dir not found under {root}. Expected either "
        f"'{root / 'data'}' or annotation folders directly inside '{root}'."
    )


def _iter_annotation_files(data_dir: Path) -> Iterable[Path]:
    yield from sorted(data_dir.rglob("*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groundcua_dir",
        required=True,
        help="Path to downloaded GroundCUA root directory",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Output JSON path, e.g. data/groundcua_bbox.json",
    )
    parser.add_argument(
        "--max_per_image",
        type=int,
        default=4,
        help="Maximum element samples to keep per screenshot (0 = keep all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum total converted samples (0 = keep all after per-image limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for per-image sampling and prompt templates",
    )
    args = parser.parse_args()

    root = Path(args.groundcua_dir).expanduser().resolve()
    data_dir = _resolve_data_dir(root)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    samples = []
    files_seen = 0
    images_seen = 0
    missing_images = 0
    skipped_boxes = 0

    for ann_path in _iter_annotation_files(data_dir):
        files_seen += 1
        with ann_path.open("r", encoding="utf-8") as f:
            entries = json.load(f)
        if not entries:
            continue

        rel_image = entries[0].get("image_path")
        if not rel_image:
            continue
        image_path = data_dir / rel_image
        if not image_path.exists():
            missing_images += 1
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        image_samples = []
        for entry in entries:
            bbox = _normalize_bbox(entry.get("bbox", []), width, height)
            if bbox is None:
                skipped_boxes += 1
                continue
            instruction = _make_instruction(entry, rng)
            image_samples.append(
                {
                    "image": rel_image,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{instruction}"},
                        {"from": "assistant", "value": "", "bbox_gt": bbox},
                    ],
                }
            )

        if not image_samples:
            continue

        images_seen += 1
        if args.max_per_image > 0 and len(image_samples) > args.max_per_image:
            image_samples = rng.sample(image_samples, args.max_per_image)
        samples.extend(image_samples)

    rng.shuffle(samples)
    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"Annotation files scanned: {files_seen}")
    print(f"Images converted: {images_seen}")
    print(f"Missing images skipped: {missing_images}")
    print(f"Invalid bboxes skipped: {skipped_boxes}")
    print(f"Samples written: {len(samples)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
