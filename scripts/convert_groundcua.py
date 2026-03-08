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
import hashlib
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


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


def _iter_image_files(root: Path) -> Iterable[Path]:
    image_root = root / "images"
    search_root = image_root if image_root.exists() else root
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for path in search_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def _build_basename_index(root: Path) -> dict[str, Path | None]:
    image_files = list(_iter_image_files(root))
    index: dict[str, Path | None] = {}
    file_iter = image_files
    if tqdm is not None:
        file_iter = tqdm(image_files, desc="Index GroundCUA images")
    for path in file_iter:
        index.setdefault(path.name, path.resolve())
    return index


def _resolve_image_path(
    *,
    root: Path,
    data_dir: Path,
    ann_path: Path,
    rel_image: str,
    basename_cache: dict[str, Path | None],
    cache_lock: threading.Lock | None = None,
) -> Path | None:
    rel_path = Path(str(rel_image))
    candidates: list[Path] = []
    if rel_path.is_absolute():
        candidates.append(rel_path)
    candidates.extend([
        data_dir / rel_path,
        root / rel_path,
        ann_path.parent / rel_path,
        ann_path.parent / rel_path.name,
        ann_path.with_suffix(rel_path.suffix or ".png"),
        ann_path.with_suffix(".png"),
        ann_path.with_suffix(".jpg"),
        ann_path.with_suffix(".jpeg"),
        ann_path.with_suffix(".webp"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    basename = rel_path.name
    if basename:
        if cache_lock is None:
            if basename not in basename_cache:
                matches = list(root.rglob(basename))
                basename_cache[basename] = matches[0].resolve() if matches else None
            return basename_cache[basename]
        with cache_lock:
            if basename in basename_cache:
                return basename_cache[basename]
        matches = list(root.rglob(basename))
        resolved = matches[0].resolve() if matches else None
        with cache_lock:
            basename_cache.setdefault(basename, resolved)
            return basename_cache[basename]
    return None


def _seed_for_path(base_seed: int, path: Path) -> int:
    digest = hashlib.sha256(f"{base_seed}:{path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _process_annotation_file(
    ann_path: Path,
    *,
    root: Path,
    data_dir: Path,
    max_per_image: int,
    base_seed: int,
    basename_cache: dict[str, Path | None],
    cache_lock: threading.Lock,
) -> dict:
    with ann_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    if not entries:
        return {"samples": [], "converted": False, "missing": 0, "unreadable": 0, "invalid": 0}

    rel_image = entries[0].get("image_path")
    if not rel_image:
        return {"samples": [], "converted": False, "missing": 0, "unreadable": 0, "invalid": 0}

    image_path = _resolve_image_path(
        root=root,
        data_dir=data_dir,
        ann_path=ann_path,
        rel_image=rel_image,
        basename_cache=basename_cache,
        cache_lock=cache_lock,
    )
    if image_path is None or not image_path.exists():
        return {"samples": [], "converted": False, "missing": 1, "unreadable": 0, "invalid": 0}

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError):
        return {"samples": [], "converted": False, "missing": 0, "unreadable": 1, "invalid": 0}

    rng = random.Random(_seed_for_path(base_seed, ann_path))
    image_samples = []
    skipped_boxes = 0
    for entry in entries:
        bbox = _normalize_bbox(entry.get("bbox", []), width, height)
        if bbox is None:
            skipped_boxes += 1
            continue
        instruction = _make_instruction(entry, rng)
        image_samples.append(
            {
                "image": str(image_path),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{instruction}"},
                    {"from": "assistant", "value": "", "bbox_gt": bbox},
                ],
            }
        )

    if not image_samples:
        return {
            "samples": [],
            "converted": False,
            "missing": 0,
            "unreadable": 0,
            "invalid": skipped_boxes,
        }

    if max_per_image > 0 and len(image_samples) > max_per_image:
        image_samples = rng.sample(image_samples, max_per_image)

    return {
        "samples": image_samples,
        "converted": True,
        "missing": 0,
        "unreadable": 0,
        "invalid": skipped_boxes,
    }


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
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4))),
        help="Number of worker threads for annotation/image processing",
    )
    args = parser.parse_args()

    root = Path(args.groundcua_dir).expanduser().resolve()
    data_dir = _resolve_data_dir(root)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    samples = []
    ann_files = list(_iter_annotation_files(data_dir))
    files_seen = len(ann_files)
    images_seen = 0
    missing_images = 0
    unreadable_images = 0
    skipped_boxes = 0
    print("Building GroundCUA image index...")
    basename_cache = _build_basename_index(root)
    cache_lock = threading.Lock()

    print(f"Processing {files_seen} annotation files with {max(1, args.workers)} workers...")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(
                _process_annotation_file,
                ann_path,
                root=root,
                data_dir=data_dir,
                max_per_image=args.max_per_image,
                base_seed=args.seed,
                basename_cache=basename_cache,
                cache_lock=cache_lock,
            )
            for ann_path in ann_files
        ]
        future_iter = as_completed(futures)
        if tqdm is not None:
            future_iter = tqdm(future_iter, total=len(futures), desc="GroundCUA convert")
        for future in future_iter:
            result = future.result()
            images_seen += int(result["converted"])
            missing_images += int(result["missing"])
            unreadable_images += int(result["unreadable"])
            skipped_boxes += int(result["invalid"])
            samples.extend(result["samples"])

    rng.shuffle(samples)
    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples[: args.max_samples]

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)

    print(f"Annotation files scanned: {files_seen}")
    print(f"Images converted: {images_seen}")
    print(f"Missing images skipped: {missing_images}")
    print(f"Unreadable images skipped: {unreadable_images}")
    print(f"Invalid bboxes skipped: {skipped_boxes}")
    print(f"Samples written: {len(samples)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
