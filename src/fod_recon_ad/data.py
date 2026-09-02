from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
Layout = Literal["auto", "original", "mvtec"]
EvalSplit = Literal["val", "test"]


@dataclass(frozen=True)
class FODSample:
    image_path: Path
    mask_path: Optional[Path]
    label: int
    distance: str
    split: str


def normalize_distance(distance: str | int) -> str:
    text = str(distance)
    if text.startswith("fod"):
        text = text.replace("fod", "").replace("_rgb", "")
    return f"{int(text):02d}"


def list_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS and not item.name.startswith(".")
    )


class ImageMaskResolver:
    """Resolve either one static mask or one mask per input image.

    A directory may be either the annotation ``masks`` directory or the root of
    a dataset view containing sibling ``roi`` directories.  Resolution first
    checks the sibling ROI next to the image and then falls back to a recursive
    filename/stem index.  Ambiguous fallback matches fail loudly instead of
    silently assigning a mask from another camera frame.
    """

    def __init__(self, source: str | Path | None) -> None:
        self.source = Path(source).resolve() if source is not None else None
        self.static_path: Optional[Path] = None
        self._by_name: Dict[str, List[Path]] = {}
        self._by_stem: Dict[str, List[Path]] = {}
        if self.source is None:
            return
        if self.source.is_file():
            self.static_path = self.source
            return
        if not self.source.is_dir():
            raise FileNotFoundError(self.source)

        candidates = sorted(
            path
            for path in self.source.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        roi_candidates = [
            path for path in candidates if path.parent.name.lower() in {"roi", "valid_roi"}
        ]
        if roi_candidates:
            candidates = roi_candidates
        if not candidates:
            raise RuntimeError(f"No ROI masks found under {self.source}")
        for path in candidates:
            self._by_name.setdefault(path.name, []).append(path)
            self._by_stem.setdefault(path.stem, []).append(path)

    @staticmethod
    def _unique(paths: List[Path], image_path: Path) -> Path:
        unique = sorted({path.resolve() for path in paths})
        if len(unique) != 1:
            raise RuntimeError(
                f"Ambiguous ROI masks for {image_path}: "
                + ", ".join(str(path) for path in unique)
            )
        return unique[0]

    def resolve(self, image_path: str | Path) -> Optional[Path]:
        if self.source is None:
            return None
        if self.static_path is not None:
            return self.static_path
        image_path = Path(image_path)

        # Compatible dataset views store rgb and roi as sibling directories.
        for suffix in (image_path.suffix, ".png"):
            sibling = image_path.parent.parent / "roi" / f"{image_path.stem}{suffix}"
            if sibling.is_file():
                try:
                    sibling.resolve().relative_to(self.source)
                except ValueError:
                    pass
                else:
                    return sibling.resolve()

        matches = self._by_name.get(image_path.name, [])
        if matches:
            return self._unique(matches, image_path)
        matches = self._by_stem.get(image_path.stem, [])
        if matches:
            return self._unique(matches, image_path)
        raise FileNotFoundError(f"No ROI mask found for {image_path} under {self.source}")

    def load(self, image_path: str | Path, size: Tuple[int, int] | None = None) -> Image.Image:
        path = self.resolve(image_path)
        if path is None:
            raise RuntimeError("Cannot load an ROI mask from an empty resolver")
        with Image.open(path) as image:
            mask = image.convert("L")
            if size is not None and mask.size != size:
                mask = mask.resize(size, Image.NEAREST)
            return mask.copy()


def resolve_valid_mask_for_image(
    cached_valid_mask: np.ndarray,
    image_path: str | Path,
    output_shape: Tuple[int, int],
    resolver: ImageMaskResolver,
) -> np.ndarray:
    """Return a cached valid mask or the matching per-image ROI override."""

    if resolver.source is None:
        return cached_valid_mask.astype(np.uint8, copy=False)
    height, width = (int(output_shape[0]), int(output_shape[1]))
    return (
        np.asarray(
            resolver.load(image_path, (width, height)),
            dtype=np.uint8,
        )
        > 0
    ).astype(np.uint8)


def detect_layout(root: Path, distance: str) -> Literal["original", "mvtec"]:
    distance = normalize_distance(distance)
    if (root / distance / "train" / "good" / "rgb").exists():
        return "original"
    if (root / f"fod{distance}_rgb" / "train" / "good").exists():
        return "mvtec"
    raise FileNotFoundError(
        f"Cannot detect FOD layout under {root}. Expected either "
        f"{root / distance / 'train/good/rgb'} or {root / f'fod{distance}_rgb' / 'train/good'}."
    )


def resolve_layout(root: Path, distance: str, layout: Layout) -> Literal["original", "mvtec"]:
    if layout == "auto":
        return detect_layout(root, distance)
    return layout


def get_transforms(image_size: int, crop_size: int):
    image_ops = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)]
    mask_ops = [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)]
    if crop_size and crop_size != image_size:
        image_ops.append(transforms.CenterCrop(crop_size))
        mask_ops.append(transforms.CenterCrop(crop_size))
    image_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    mask_ops.append(transforms.ToTensor())
    return transforms.Compose(image_ops), transforms.Compose(mask_ops)


def get_tensor_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def resize_preserve_height(image: Image.Image, target_height: int, interpolation: int) -> Image.Image:
    if target_height <= 0:
        return image
    width, height = image.size
    target_width = max(int(round(width * float(target_height) / float(height))), 1)
    return image.resize((target_width, int(target_height)), interpolation)


def window_starts(length: int, crop_size: int, stride: int) -> List[int]:
    if length <= crop_size:
        return [0]
    stride = max(int(stride), 1)
    starts = list(range(0, length - crop_size + 1, stride))
    last = length - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_windows(width: int, height: int, crop_size: int, stride: int) -> List[Tuple[int, int]]:
    xs = window_starts(width, crop_size, stride)
    ys = window_starts(height, crop_size, stride)
    return [(x, y) for y in ys for x in xs]


def _original_train_samples(root: Path, distance: str, modality: str) -> List[FODSample]:
    image_dir = root / distance / "train" / "good" / modality
    return [
        FODSample(path, None, 0, distance=distance, split="train")
        for path in list_images(image_dir)
    ]


def _mvtec_train_samples(root: Path, distance: str) -> List[FODSample]:
    item = f"fod{distance}_rgb"
    image_dir = root / item / "train" / "good"
    return [
        FODSample(path, None, 0, distance=distance, split="train")
        for path in list_images(image_dir)
    ]


def _original_eval_samples(root: Path, distance: str, split: EvalSplit, modality: str) -> List[FODSample]:
    samples: List[FODSample] = []
    for label_name, label in (("good", 0), ("anomaly", 1)):
        image_dir = root / distance / split / label_name / modality
        mask_dir = root / distance / split / label_name / "gt"
        for image_path in list_images(image_dir):
            mask_path = mask_dir / image_path.name
            samples.append(
                FODSample(
                    image_path=image_path,
                    mask_path=mask_path if mask_path.exists() else None,
                    label=label,
                    distance=distance,
                    split=split,
                )
            )
    return samples


def _mvtec_test_samples(root: Path, distance: str) -> List[FODSample]:
    item = f"fod{distance}_rgb"
    samples: List[FODSample] = []
    for label_name, label in (("good", 0), ("anomaly", 1)):
        image_dir = root / item / "test" / label_name
        mask_dir = root / item / "ground_truth" / label_name
        for image_path in list_images(image_dir):
            mask_path = mask_dir / image_path.name
            samples.append(
                FODSample(
                    image_path=image_path,
                    mask_path=mask_path if mask_path.exists() else None,
                    label=label,
                    distance=distance,
                    split="test",
                )
            )
    return samples


def _mvtec_category_train_samples(root: Path, category: str) -> List[FODSample]:
    image_dir = root / category / "train" / "good"
    return [
        FODSample(path, None, 0, distance=category, split="train")
        for path in list_images(image_dir)
    ]


def _mvtec_category_test_samples(root: Path, category: str) -> List[FODSample]:
    samples: List[FODSample] = []
    test_root = root / category / "test"
    gt_root = root / category / "ground_truth"
    for defect_dir in sorted(test_root.iterdir() if test_root.exists() else []):
        if not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for image_path in list_images(defect_dir):
            mask_path = None
            if label:
                candidate = gt_root / defect_type / f"{image_path.stem}_mask.png"
                if not candidate.exists():
                    candidate = gt_root / defect_type / image_path.name
                mask_path = candidate if candidate.exists() else None
            samples.append(
                FODSample(
                    image_path=image_path,
                    mask_path=mask_path,
                    label=label,
                    distance=category,
                    split="test",
                )
            )
    return samples


class FODTrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        distance: str = "05",
        image_size: int = 448,
        crop_size: int = 448,
        modality: str = "rgb",
        layout: Layout = "auto",
    ) -> None:
        self.root = Path(root)
        self.distance = normalize_distance(distance)
        self.layout = resolve_layout(self.root, self.distance, layout)
        self.image_transform, _ = get_transforms(image_size, crop_size)
        if self.layout == "original":
            self.samples = _original_train_samples(self.root, self.distance, modality)
        else:
            self.samples = _mvtec_train_samples(self.root, self.distance)
        if not self.samples:
            raise RuntimeError(f"No FOD train images found: root={self.root}, distance={self.distance}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return self.image_transform(image), torch.tensor(0, dtype=torch.long), str(sample.image_path)


class FODCropTrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        distance: str = "05",
        full_height: int = 672,
        crop_size: int = 448,
        output_size: int | None = None,
        stride: int = 224,
        modality: str = "rgb",
        layout: Layout = "auto",
        roi_mask_path: str | Path | None = None,
        min_roi_coverage: float = 0.0,
        return_roi_mask: bool = False,
        roi_erode_pixels: int = 0,
    ) -> None:
        self.root = Path(root)
        self.distance = normalize_distance(distance)
        self.full_height = int(full_height)
        self.crop_size = int(crop_size)
        self.output_size = int(output_size) if output_size else self.crop_size
        self.stride = int(stride)
        self.layout = resolve_layout(self.root, self.distance, layout)
        self.image_transform = get_tensor_transform()
        # This may be one legacy static mask, the manual mask directory, or a
        # clean-ROI batch-view root containing per-sample ``roi`` directories.
        self.roi_mask_path = Path(roi_mask_path) if roi_mask_path else None
        self.roi_mask_resolver = ImageMaskResolver(self.roi_mask_path)
        self.min_roi_coverage = float(min_roi_coverage)
        self.return_roi_mask = bool(return_roi_mask)
        self.roi_erode_pixels = int(roi_erode_pixels)
        if self.return_roi_mask and self.roi_mask_path is None:
            raise ValueError("return_roi_mask requires roi_mask_path.")
        if self.roi_erode_pixels < 0:
            raise ValueError("roi_erode_pixels must be non-negative.")
        if self.layout == "original":
            samples = _original_train_samples(self.root, self.distance, modality)
        else:
            samples = _mvtec_train_samples(self.root, self.distance)
        if not samples:
            raise RuntimeError(f"No FOD train images found: root={self.root}, distance={self.distance}")

        self.crops: List[Tuple[FODSample, int, int]] = []
        for sample in samples:
            with Image.open(sample.image_path) as image:
                resized = resize_preserve_height(image, self.full_height, Image.BILINEAR)
                roi = None
                if self.roi_mask_path is not None:
                    roi = self.roi_mask_resolver.load(sample.image_path, resized.size)
                    roi = (np.asarray(roi) > 0).astype(np.uint8)
                    if self.roi_erode_pixels > 0:
                        radius = self.roi_erode_pixels
                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
                        )
                        roi = cv2.erode(roi, kernel, iterations=1)
                    roi = torch.from_numpy(roi.astype(np.float32))
                for x, y in sliding_windows(resized.size[0], resized.size[1], self.crop_size, self.stride):
                    if roi is not None and self.min_roi_coverage > 0:
                        coverage = float(roi[y : y + self.crop_size, x : x + self.crop_size].mean())
                        if coverage < self.min_roi_coverage:
                            continue
                    self.crops.append((sample, x, y))
        if not self.crops:
            raise RuntimeError(f"No FOD crop train samples produced: root={self.root}, distance={self.distance}")

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, index: int):
        sample, x, y = self.crops[index]
        image = Image.open(sample.image_path).convert("RGB")
        image = resize_preserve_height(image, self.full_height, Image.BILINEAR)
        crop = image.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.output_size != self.crop_size:
            crop = crop.resize((self.output_size, self.output_size), Image.BILINEAR)
        image_tensor = self.image_transform(crop)
        path = f"{sample.image_path}:{x}:{y}"
        if not self.return_roi_mask:
            return image_tensor, torch.tensor(0, dtype=torch.long), path
        roi = self.roi_mask_resolver.load(sample.image_path, image.size)
        roi_array = (np.asarray(roi) > 0).astype(np.uint8)
        if self.roi_erode_pixels > 0:
            radius = self.roi_erode_pixels
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
            )
            roi_array = cv2.erode(roi_array, kernel, iterations=1)
        roi_crop = Image.fromarray(roi_array * 255).crop(
            (x, y, x + self.crop_size, y + self.crop_size)
        )
        if self.output_size != self.crop_size:
            roi_crop = roi_crop.resize(
                (self.output_size, self.output_size), Image.NEAREST
            )
        roi_tensor = torch.from_numpy(
            (np.asarray(roi_crop) > 0).astype(np.float32)
        ).unsqueeze(0)
        return image_tensor, torch.tensor(0, dtype=torch.long), path, roi_tensor


class FODTestDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        distance: str = "05",
        image_size: int = 448,
        crop_size: int = 448,
        modality: str = "rgb",
        layout: Layout = "auto",
        split: EvalSplit = "test",
    ) -> None:
        self.root = Path(root)
        self.distance = normalize_distance(distance)
        self.split = split
        self.layout = resolve_layout(self.root, self.distance, layout)
        self.image_transform, self.mask_transform = get_transforms(image_size, crop_size)
        if self.layout == "original":
            self.samples = _original_eval_samples(self.root, self.distance, split, modality)
        else:
            if split != "test":
                raise ValueError("MVTec-style FOD layout only supports split='test'.")
            self.samples = _mvtec_test_samples(self.root, self.distance)
        if not self.samples:
            raise RuntimeError(f"No FOD {split} images found: root={self.root}, distance={self.distance}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_pil = Image.open(sample.image_path).convert("RGB")
        image = self.image_transform(image_pil)
        if sample.mask_path is not None and sample.mask_path.exists():
            mask_pil = Image.open(sample.mask_path).convert("L")
        else:
            mask_pil = Image.new("L", image_pil.size, 0)
        mask = (self.mask_transform(mask_pil) > 0).float()
        return image, mask, torch.tensor(sample.label, dtype=torch.long), str(sample.image_path)

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor([sample.label for sample in self.samples], dtype=torch.long)


class MVTecCategoryTrainDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        image_size: int = 448,
        crop_size: int = 392,
        physical_crop_size: int = 0,
    ) -> None:
        self.root = Path(root)
        self.category = category
        if physical_crop_size > 0:
            self.image_transform = transforms.Compose(
                [
                    transforms.RandomCrop(
                        (physical_crop_size, physical_crop_size),
                        pad_if_needed=True,
                        padding_mode="reflect",
                    ),
                    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.image_transform, _ = get_transforms(image_size, crop_size)
        self.samples = _mvtec_category_train_samples(self.root, self.category)
        if not self.samples:
            raise RuntimeError(f"No MVTec train images found: root={self.root}, category={self.category}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        return self.image_transform(image), torch.tensor(0, dtype=torch.long), str(sample.image_path)


class MVTecCategoryTestDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        category: str,
        image_size: int = 448,
        crop_size: int = 392,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.image_transform, self.mask_transform = get_transforms(image_size, crop_size)
        self.samples = _mvtec_category_test_samples(self.root, self.category)
        if not self.samples:
            raise RuntimeError(f"No MVTec test images found: root={self.root}, category={self.category}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_pil = Image.open(sample.image_path).convert("RGB")
        image = self.image_transform(image_pil)
        if sample.mask_path is not None and sample.mask_path.exists():
            mask_pil = Image.open(sample.mask_path).convert("L")
        else:
            mask_pil = Image.new("L", image_pil.size, 0)
        mask = (self.mask_transform(mask_pil) > 0).float()
        return image, mask, torch.tensor(sample.label, dtype=torch.long), str(sample.image_path)


def make_train_loader(
    root: str | Path,
    distance: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    workers: int,
    modality: str = "rgb",
    layout: Layout = "auto",
    shuffle: bool = True,
) -> DataLoader:
    dataset = FODTrainDataset(root, distance, image_size, crop_size, modality, layout)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_crop_train_loader(
    root: str | Path,
    distance: str,
    full_height: int,
    crop_size: int,
    output_size: int | None,
    stride: int,
    batch_size: int,
    workers: int,
    modality: str = "rgb",
    layout: Layout = "auto",
    shuffle: bool = True,
    drop_last: bool = False,
    roi_mask_path: str | Path | None = None,
    min_roi_coverage: float = 0.0,
    return_roi_mask: bool = False,
    roi_erode_pixels: int = 0,
) -> DataLoader:
    dataset = FODCropTrainDataset(
        root,
        distance,
        full_height,
        crop_size,
        output_size,
        stride,
        modality,
        layout,
        roi_mask_path,
        min_roi_coverage,
        return_roi_mask,
        roi_erode_pixels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_test_loader(
    root: str | Path,
    distance: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    workers: int,
    modality: str = "rgb",
    layout: Layout = "auto",
    split: EvalSplit = "test",
) -> DataLoader:
    dataset = FODTestDataset(root, distance, image_size, crop_size, modality, layout, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_mvtec_train_loader(
    root: str | Path,
    category: str,
    image_size: int,
    crop_size: int,
    physical_crop_size: int,
    batch_size: int,
    workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = MVTecCategoryTrainDataset(root, category, image_size, crop_size, physical_crop_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def make_mvtec_test_loader(
    root: str | Path,
    category: str,
    image_size: int,
    crop_size: int,
    batch_size: int,
    workers: int,
) -> DataLoader:
    dataset = MVTecCategoryTestDataset(root, category, image_size, crop_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def default_distances() -> Tuple[str, ...]:
    return ("05", "10", "15", "20", "25", "30")


def normalize_distances(distances: Iterable[str | int]) -> List[str]:
    return [normalize_distance(distance) for distance in distances]
