from __future__ import annotations

import argparse
import os
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_WORKSPACE = PROJECT_ROOT / "data" / "yolo_wall_finetune"
DEFAULT_BASE_DATASET = PROJECT_ROOT / "data" / "base_tank_dataset"
DEFAULT_BEST_PT = PROJECT_ROOT / "runs" / "detect" / "first_yolo11n" / "weights" / "best.pt"
DEFAULT_PROJECT_DIR = PROJECT_ROOT / "runs" / "detect"
DEFAULT_ROBOFLOW_API_KEY = ""
DEFAULT_EPOCHS = 150

ROBOFLOW_WORKSPACE = "has-workspace-3feui"
ROBOFLOW_PROJECT = "tankkk2"
ROBOFLOW_VERSION = 1
ROBOFLOW_FORMAT = "yolov11"

BASE_CLASSES = ["blue", "car", "red", "rock", "tank"]
FINAL_CLASSES = ["rock", "Tank", "person", "tent", "wall"]
CLASS_ALIASES = {
    "blue": "person",
    "red": "person",
    "person": "person",
    "rock": "rock",
    "tank": "Tank",
    "tent": "tent",
    "wall": "wall",
}
IGNORED_SOURCE_CLASSES = {"car"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class CopyStats:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    ignored_boxes: int = 0


@dataclass
class SplitMoveStats:
    images: int = 0
    boxes: int = 0


@dataclass
class DuplicateStats:
    images: int = 0
    boxes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Roboflow tankkk2 and fine-tune YOLO with tent/wall data."
    )
    parser.add_argument("--api-key", default=os.getenv("ROBOFLOW_API_KEY", DEFAULT_ROBOFLOW_API_KEY))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_BASE_DATASET)
    parser.add_argument(
        "--best-pt",
        type=Path,
        default=DEFAULT_BEST_PT,
        help="Existing trained best.pt to use as the fine-tuning starting model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs. Default is {DEFAULT_EPOCHS}; increase this for longer fine-tuning.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--box-loss-gain", type=float, default=8.5)
    parser.add_argument("--cls-loss-gain", type=float, default=0.75)
    parser.add_argument("--dfl-loss-gain", type=float, default=1.7)
    parser.add_argument("--close-mosaic", type=int, default=15)
    parser.add_argument("--val-iou", type=float, default=0.75)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--name", default="finetune_tankkk2")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--min-valid-boxes-per-class",
        type=int,
        default=100,
        help="Move train samples into valid until each available class has at least this many validation boxes.",
    )
    parser.add_argument(
        "--focus-classes",
        nargs="*",
        default=["wall"],
        help="Classes to oversample in train. Default emphasizes wall.",
    )
    parser.add_argument(
        "--focus-copy-factor",
        type=int,
        default=2,
        help="How many extra train copies to make for samples containing focus classes.",
    )
    parser.add_argument(
        "--contrast-classes",
        nargs="*",
        default=["rock"],
        help="Classes to sample as contrast negatives for the focus classes.",
    )
    parser.add_argument(
        "--contrast-ratio",
        type=float,
        default=0.5,
        help="How many contrast samples to copy relative to focus sample count.",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is not installed. Run: pip install -r requirements.txt") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_names(names: object) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError(f"Unsupported names format in data.yaml: {names!r}")


def resolve_images_dir(dataset_root: Path, data_yaml: dict, output_split: str) -> Path | None:
    yaml_key = "val" if output_split == "valid" else output_split
    yaml_value = data_yaml.get(yaml_key)
    candidates: list[Path] = []

    if yaml_value:
        yaml_path = Path(str(yaml_value))
        if yaml_path.is_absolute():
            candidates.append(yaml_path)
        else:
            path_root = Path(str(data_yaml.get("path", dataset_root)))
            if not path_root.is_absolute():
                path_root = dataset_root / path_root
            candidates.append((path_root / yaml_path).resolve())
            candidates.append((dataset_root / yaml_path).resolve())

    candidates.append(dataset_root / output_split / "images")
    if output_split == "valid":
        candidates.append(dataset_root / "val" / "images")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def labels_dir_for(images_dir: Path) -> Path:
    if images_dir.name == "images":
        return images_dir.parent / "labels"
    return images_dir.with_name("labels")


def iter_images(images_dir: Path | None) -> list[Path]:
    if images_dir is None or not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def canonical_class_name(source_name: str) -> str | None:
    normalized = source_name.strip().lower()
    if normalized in IGNORED_SOURCE_CLASSES:
        return None
    if normalized not in CLASS_ALIASES:
        raise ValueError(
            f"Unsupported source class {source_name!r}. "
            f"Supported final classes are {FINAL_CLASSES}; ignored classes are {sorted(IGNORED_SOURCE_CLASSES)}."
        )
    return CLASS_ALIASES[normalized]


def build_class_id_map(source_names: list[str]) -> dict[int, int | None]:
    class_id_map: dict[int, int | None] = {}
    for source_id, source_name in enumerate(source_names):
        canonical_name = canonical_class_name(source_name)
        class_id_map[source_id] = None if canonical_name is None else FINAL_CLASSES.index(canonical_name)
    return class_id_map


def remap_label_line(line: str, source_label: Path, class_id_map: dict[int, int | None]) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None

    parts = stripped.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line in {source_label}: {line!r}")

    source_id = int(float(parts[0]))
    if source_id not in class_id_map:
        raise ValueError(f"Class id {source_id} from {source_label} is not in data.yaml names.")

    target_id = class_id_map[source_id]
    if target_id is None:
        return None

    return " ".join([str(target_id), *to_detection_box(parts[1:], source_label)])


def to_detection_box(coords: list[str], source_label: Path) -> list[str]:
    if len(coords) == 4:
        x_center, y_center, width, height = (float(value) for value in coords)
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        return corners_to_yolo_box(x_min, y_min, x_max, y_max)

    values = [float(value) for value in coords]
    if len(values) < 6 or len(values) % 2 != 0:
        raise ValueError(f"Invalid detection/segment coordinates in {source_label}: {coords!r}")

    xs = values[0::2]
    ys = values[1::2]
    return corners_to_yolo_box(min(xs), min(ys), max(xs), max(ys))


def corners_to_yolo_box(x_min: float, y_min: float, x_max: float, y_max: float) -> list[str]:
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    return [
        f"{x_center:.6f}",
        f"{y_center:.6f}",
        f"{width:.6f}",
        f"{height:.6f}",
    ]


def copy_split(
    source_root: Path,
    source_data_yaml: dict,
    output_root: Path,
    output_split: str,
    prefix: str,
    class_id_map: dict[int, int | None],
) -> CopyStats:
    images_dir = resolve_images_dir(source_root, source_data_yaml, output_split)
    source_labels_dir = labels_dir_for(images_dir) if images_dir else None
    output_images_dir = output_root / output_split / "images"
    output_labels_dir = output_root / output_split / "labels"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = CopyStats()
    for image_path in iter_images(images_dir):
        output_stem = f"{prefix}_{output_split}_{image_path.stem}"
        output_image_path = output_images_dir / f"{output_stem}{image_path.suffix.lower()}"
        output_label_path = output_labels_dir / f"{output_stem}.txt"
        source_label_path = source_labels_dir / f"{image_path.stem}.txt" if source_labels_dir else None

        shutil.copy2(image_path, output_image_path)
        stats.images += 1

        label_lines: list[str] = []
        if source_label_path and source_label_path.exists():
            for line in source_label_path.read_text(encoding="utf-8").splitlines():
                remapped = remap_label_line(line, source_label_path, class_id_map)
                if remapped is None:
                    if line.strip():
                        stats.ignored_boxes += 1
                    continue
                label_lines.append(remapped)
                stats.boxes += 1
            stats.labels += 1

        label_lines = dedupe_label_lines(label_lines)
        output_label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

    return stats


def dedupe_label_lines(label_lines: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for line in label_lines:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
    return deduped


def write_combined_yaml(output_root: Path, class_names: list[str]) -> Path:
    data_yaml = output_root / "data.yaml"
    names_yaml = "\n".join(f"  - {name}" for name in class_names)
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: train/images",
                "val: valid/images",
                "test: test/images",
                f"nc: {len(class_names)}",
                "names:",
                names_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return data_yaml


def label_counts(label_path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            counts[int(float(parts[0]))] += 1
    return counts


def split_counts(dataset_root: Path, split: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    labels_dir = dataset_root / split / "labels"
    if not labels_dir.exists():
        return counts
    for label_path in labels_dir.glob("*.txt"):
        counts.update(label_counts(label_path))
    return counts


def find_matching_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def move_train_sample_to_valid(dataset_root: Path, label_path: Path) -> int:
    train_images_dir = dataset_root / "train" / "images"
    train_labels_dir = dataset_root / "train" / "labels"
    valid_images_dir = dataset_root / "valid" / "images"
    valid_labels_dir = dataset_root / "valid" / "labels"
    valid_images_dir.mkdir(parents=True, exist_ok=True)
    valid_labels_dir.mkdir(parents=True, exist_ok=True)

    image_path = find_matching_image(train_images_dir, label_path.stem)
    if image_path is None:
        raise FileNotFoundError(f"Image for label not found: {label_path}")

    box_count = sum(label_counts(label_path).values())
    shutil.move(str(image_path), str(valid_images_dir / image_path.name))
    shutil.move(str(label_path), str(valid_labels_dir / label_path.name))

    empty_train_label = train_labels_dir / label_path.name
    if empty_train_label.exists():
        empty_train_label.unlink()
    return box_count


def ensure_validation_coverage(
    dataset_root: Path,
    class_names: list[str],
    min_boxes_per_class: int,
) -> SplitMoveStats:
    if min_boxes_per_class <= 0:
        return SplitMoveStats()

    stats = SplitMoveStats()
    valid_counts = split_counts(dataset_root, "valid")
    train_labels_dir = dataset_root / "train" / "labels"

    for class_id, class_name in enumerate(class_names):
        if valid_counts[class_id] >= min_boxes_per_class:
            continue

        train_candidates: list[tuple[int, Path]] = []
        for label_path in sorted(train_labels_dir.glob("*.txt")):
            counts = label_counts(label_path)
            if counts[class_id] > 0:
                train_candidates.append((counts[class_id], label_path))

        if not train_candidates:
            print(f"Warning: no train samples available for validation class {class_name!r}")
            continue

        train_candidates.sort(key=lambda item: (-item[0], item[1].name))
        for _, label_path in train_candidates:
            if not label_path.exists():
                continue
            moved_counts = label_counts(label_path)
            moved_boxes = move_train_sample_to_valid(dataset_root, label_path)
            valid_counts.update(moved_counts)
            stats.images += 1
            stats.boxes += moved_boxes
            if valid_counts[class_id] >= min_boxes_per_class:
                break

        if valid_counts[class_id] < min_boxes_per_class:
            print(
                f"Warning: validation class {class_name!r} has only "
                f"{valid_counts[class_id]} boxes after holdout."
            )

    if stats.images:
        print(f"Validation holdout: moved {stats.images} train images/{stats.boxes} boxes into valid.")
    print_split_distribution(dataset_root, class_names)
    return stats


def print_split_distribution(dataset_root: Path, class_names: list[str]) -> None:
    for split in ("train", "valid", "test"):
        counts = split_counts(dataset_root, split)
        total = sum(counts.values())
        parts = ", ".join(f"{name}={counts[class_id]}" for class_id, name in enumerate(class_names))
        print(f"{split} label distribution ({total} boxes): {parts}")


def class_ids_for_names(class_names: list[str], selected_names: list[str]) -> set[int]:
    normalized_to_id = {name.lower(): class_id for class_id, name in enumerate(class_names)}
    class_ids: set[int] = set()
    for selected_name in selected_names:
        normalized = selected_name.strip().lower()
        if not normalized:
            continue
        if normalized not in normalized_to_id:
            raise ValueError(f"Unknown class {selected_name!r}; expected one of {class_names}")
        class_ids.add(normalized_to_id[normalized])
    return class_ids


def duplicate_train_samples(
    dataset_root: Path,
    class_names: list[str],
    focus_classes: list[str],
    focus_copy_factor: int,
    contrast_classes: list[str],
    contrast_ratio: float,
) -> DuplicateStats:
    if focus_copy_factor <= 0:
        return DuplicateStats()

    focus_ids = class_ids_for_names(class_names, focus_classes)
    contrast_ids = class_ids_for_names(class_names, contrast_classes)
    if not focus_ids:
        return DuplicateStats()

    train_images_dir = dataset_root / "train" / "images"
    train_labels_dir = dataset_root / "train" / "labels"
    label_paths = sorted(train_labels_dir.glob("*.txt"))

    focus_label_paths: list[Path] = []
    contrast_label_paths: list[Path] = []
    for label_path in label_paths:
        counts = label_counts(label_path)
        has_focus = any(counts[class_id] > 0 for class_id in focus_ids)
        has_contrast = any(counts[class_id] > 0 for class_id in contrast_ids)
        if has_focus:
            focus_label_paths.append(label_path)
        elif has_contrast:
            contrast_label_paths.append(label_path)

    stats = DuplicateStats()
    for copy_index in range(focus_copy_factor):
        for label_path in focus_label_paths:
            stats.boxes += duplicate_train_sample(
                train_images_dir,
                train_labels_dir,
                label_path,
                f"focus{copy_index + 1}",
            )
            stats.images += 1

    contrast_limit = max(0, int(len(focus_label_paths) * contrast_ratio))
    for label_path in contrast_label_paths[:contrast_limit]:
        stats.boxes += duplicate_train_sample(
            train_images_dir,
            train_labels_dir,
            label_path,
            "contrast1",
        )
        stats.images += 1

    if stats.images:
        print(
            f"Rock/wall focus oversampling: copied {stats.images} train images/"
            f"{stats.boxes} boxes (focus={focus_classes}, contrast={contrast_classes})."
        )
        print_split_distribution(dataset_root, class_names)
    return stats


def duplicate_train_sample(
    train_images_dir: Path,
    train_labels_dir: Path,
    label_path: Path,
    prefix: str,
) -> int:
    image_path = find_matching_image(train_images_dir, label_path.stem)
    if image_path is None:
        raise FileNotFoundError(f"Image for label not found: {label_path}")

    output_stem = f"{prefix}_{label_path.stem}"
    shutil.copy2(image_path, train_images_dir / f"{output_stem}{image_path.suffix}")
    shutil.copy2(label_path, train_labels_dir / f"{output_stem}.txt")
    return sum(label_counts(label_path).values())


def download_roboflow_dataset(api_key: str, download_dir: Path) -> Path:
    if not api_key:
        raise SystemExit(
            "ROBOFLOW_API_KEY is not set. Set it first, or pass --api-key."
        )

    import requests

    api_url = (
        f"https://api.roboflow.com/{ROBOFLOW_WORKSPACE}/"
        f"{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}/{ROBOFLOW_FORMAT}"
    )
    response = requests.get(api_url, params={"api_key": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()

    download_url = payload.get("export", {}).get("link")
    if not download_url:
        raise RuntimeError(f"Roboflow export link not found in response: {payload}")

    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    zip_path = download_dir.with_suffix(".zip")
    with requests.get(download_url, stream=True, timeout=300) as zip_response:
        zip_response.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in zip_response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path) as dataset_zip:
        dataset_zip.extractall(download_dir)

    return find_dataset_root(download_dir)


def find_dataset_root(path: Path) -> Path:
    data_yaml_files = sorted(path.rglob("data.yaml"))
    if not data_yaml_files:
        raise FileNotFoundError(f"data.yaml not found under {path}")
    return data_yaml_files[0].parent.resolve()


def build_combined_dataset(args: argparse.Namespace) -> tuple[Path, list[str]]:
    workspace = args.workspace.resolve()
    download_dir = workspace / "roboflow_download"
    combined_dir = workspace / "combined"

    if args.skip_download:
        new_dataset_root = find_dataset_root(download_dir.resolve())
    else:
        new_dataset_root = download_roboflow_dataset(args.api_key, download_dir)

    base_root = args.base_dataset.resolve()
    base_yaml_path = base_root / "data.yaml"
    new_yaml_path = new_dataset_root / "data.yaml"

    if not base_yaml_path.exists():
        raise FileNotFoundError(f"Base data.yaml not found: {base_yaml_path}")
    if not new_yaml_path.exists():
        raise FileNotFoundError(f"Roboflow data.yaml not found: {new_yaml_path}")

    base_yaml = read_yaml(base_yaml_path)
    new_yaml = read_yaml(new_yaml_path)
    base_names = normalize_names(base_yaml.get("names", BASE_CLASSES))
    new_names = normalize_names(new_yaml.get("names"))
    class_names = [*FINAL_CLASSES]

    base_map = build_class_id_map(base_names)
    new_map = build_class_id_map(new_names)

    reset_dir(combined_dir)
    for split in ("train", "valid", "test"):
        base_stats = copy_split(base_root, base_yaml, combined_dir, split, "base", base_map)
        new_stats = copy_split(new_dataset_root, new_yaml, combined_dir, split, "rf", new_map)
        print(
            f"{split}: base {base_stats.images} images/{base_stats.boxes} boxes "
            f"({base_stats.ignored_boxes} ignored), "
            f"roboflow {new_stats.images} images/{new_stats.boxes} boxes "
            f"({new_stats.ignored_boxes} ignored)"
        )

    ensure_validation_coverage(combined_dir, class_names, args.min_valid_boxes_per_class)
    duplicate_train_samples(
        combined_dir,
        class_names,
        args.focus_classes,
        args.focus_copy_factor,
        args.contrast_classes,
        args.contrast_ratio,
    )

    combined_yaml = write_combined_yaml(combined_dir, class_names)
    print(f"Combined classes: {class_names}")
    print(f"Combined data.yaml: {combined_yaml}")
    return combined_yaml, class_names


def resolve_best_pt(path: Path) -> Path:
    best_pt = path.resolve()
    if not best_pt.exists():
        raise FileNotFoundError(
            f"Fine-tuning source best.pt not found: {best_pt}\n"
            "Train the base model first, or pass --best-pt with the correct path."
        )
    if best_pt.name != "best.pt":
        print(f"Warning: source model is not named best.pt: {best_pt}")
    return best_pt


def fine_tune(args: argparse.Namespace, data_yaml: Path) -> Path:
    from ultralytics import YOLO

    best_pt = resolve_best_pt(args.best_pt)
    print(f"Fine-tuning from: {best_pt}")
    model = YOLO(str(best_pt))

    started_at = time.time()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        box=args.box_loss_gain,
        cls=args.cls_loss_gain,
        dfl=args.dfl_loss_gain,
        close_mosaic=args.close_mosaic,
        iou=args.val_iou,
        project=str(args.project_dir),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"

    if not args.skip_val:
        val_model = YOLO(str(best_weights))
        val_model.val(data=str(data_yaml), device=args.device)

    elapsed = time.time() - started_at
    print(f"Fine-tuned weights: {best_weights}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    return best_weights


def main() -> None:
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    data_yaml, _ = build_combined_dataset(args)

    if args.build_only:
        return

    fine_tune(args, data_yaml)


if __name__ == "__main__":
    main()
