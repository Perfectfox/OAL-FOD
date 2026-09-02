from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional speedup
    cv2 = None  # type: ignore[assignment]


METRIC_KEYS = ["I-AUROC", "I-AP", "I-F1", "P-AUROC", "P-AP", "P-F1", "AUC-PRO"]
NEIGHBORS_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = values.astype(np.float32)
    min_v = float(values.min())
    max_v = float(values.max())
    if max_v <= min_v + eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - min_v) / (max_v - min_v + eps)).astype(np.float32)


def safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if np.unique(y_true).size < 2:
        return float("nan")
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = original_ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def safe_average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if np.asarray(y_true).sum() == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    precision = tp / ranks
    return float(precision[y_sorted == 1].mean())


def f1_score_max(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if np.unique(y_true).size < 2:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y_true.sum()), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-7)
    return float(f1.max()) if f1.size else 0.0


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


def connected_components(mask: np.ndarray) -> List[tuple[np.ndarray, np.ndarray, float]]:
    mask = mask.astype(bool)
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            ys, xs = [], []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            yy = np.asarray(ys, dtype=np.int64)
            xx = np.asarray(xs, dtype=np.int64)
            components.append((yy, xx, float(len(ys))))
    return components


def sparse_connected_components(mask: np.ndarray, min_area: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return []
    if cv2 is not None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        components: List[Tuple[np.ndarray, np.ndarray]] = []
        for label_idx in range(1, int(count)):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x0 = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y0 = int(stats[label_idx, cv2.CC_STAT_TOP])
            width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            yy, xx = np.nonzero(labels[y0 : y0 + height, x0 : x0 + width] == label_idx)
            components.append((yy.astype(np.int64) + y0, xx.astype(np.int64) + x0))
        return components
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: List[Tuple[np.ndarray, np.ndarray]] = []
    for y0, x0 in coords:
        y0 = int(y0)
        x0 = int(x0)
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        ys: List[int] = []
        xs: List[int] = []
        while stack:
            y, x = stack.pop()
            ys.append(y)
            xs.append(x)
            for dy, dx in NEIGHBORS_8:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(ys) >= min_area:
            components.append((np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)))
    return components


class StreamingPROHistogram:
    """Approximate AUPRO with constant memory using score histograms."""

    def __init__(self, bins: int) -> None:
        if bins < 16:
            raise ValueError("AUPRO histogram bins should be at least 16.")
        self.bins = int(bins)
        self.min_score = None
        self.max_score = None
        self.region_hist_sum = np.zeros(self.bins, dtype=np.float64)
        self.bg_hist = np.zeros(self.bins, dtype=np.float64)
        self.region_count = 0
        self.bg_total = 0.0

    def _rebin(self, new_min: float, new_max: float) -> None:
        if self.min_score is None or self.max_score is None:
            self.min_score = new_min
            self.max_score = new_max
            return
        if new_min >= self.min_score and new_max <= self.max_score:
            return
        old_min, old_max = self.min_score, self.max_score
        old_region = self.region_hist_sum
        old_bg = self.bg_hist
        self.min_score = min(self.min_score, new_min)
        self.max_score = max(self.max_score, new_max)
        self.region_hist_sum = np.zeros(self.bins, dtype=np.float64)
        self.bg_hist = np.zeros(self.bins, dtype=np.float64)
        if old_max <= old_min:
            self.region_hist_sum[0] += old_region.sum()
            self.bg_hist[0] += old_bg.sum()
            return
        old_centers = np.linspace(old_min, old_max, self.bins, dtype=np.float64)
        scale = (self.bins - 1) / max(self.max_score - self.min_score, 1e-12)
        new_idx = np.clip(((old_centers - self.min_score) * scale).round().astype(np.int64), 0, self.bins - 1)
        np.add.at(self.region_hist_sum, new_idx, old_region)
        np.add.at(self.bg_hist, new_idx, old_bg)

    def _indices(self, scores: np.ndarray) -> np.ndarray:
        assert self.min_score is not None and self.max_score is not None
        if self.max_score <= self.min_score:
            return np.zeros(scores.size, dtype=np.int64)
        scale = (self.bins - 1) / (self.max_score - self.min_score)
        return np.clip(((scores.reshape(-1) - self.min_score) * scale).astype(np.int64), 0, self.bins - 1)

    def update(self, score_map: np.ndarray, mask: np.ndarray) -> None:
        scores = np.asarray(score_map, dtype=np.float32)
        finite = np.isfinite(scores)
        if not finite.all():
            raise ValueError(f"Prediction map contains {int((~finite).sum())} non-finite values.")
        self._rebin(float(scores.min()), float(scores.max()))
        labels = np.asarray(mask).astype(bool)
        bg_scores = scores[~labels]
        self.bg_total += float(bg_scores.size)
        if bg_scores.size:
            self.bg_hist += np.bincount(self._indices(bg_scores), minlength=self.bins).astype(np.float64)
        for yy, xx in sparse_connected_components(labels, min_area=1):
            region_scores = scores[yy, xx]
            if region_scores.size == 0:
                continue
            hist = np.bincount(self._indices(region_scores), minlength=self.bins).astype(np.float64)
            self.region_hist_sum += hist / max(float(region_scores.size), 1.0)
            self.region_count += 1

    def compute(self) -> float:
        if self.region_count <= 0 or self.bg_total <= 0:
            return float("nan")
        pro = np.cumsum(self.region_hist_sum[::-1]) / max(float(self.region_count), 1.0)
        fpr = np.cumsum(self.bg_hist[::-1]) / max(float(self.bg_total), 1.0)
        keep = fpr < 0.3
        if int(keep.sum()) < 2:
            return 0.0
        fpr = fpr[keep]
        pro = pro[keep]
        max_fpr = float(fpr.max())
        if max_fpr <= 0:
            return float(pro.max())
        order = np.argsort(fpr)
        return float(np.trapz(pro[order], fpr[order] / max_fpr))


def _precompute_regions(masks: np.ndarray):
    regions = []
    inverse_masks = 1 - masks.astype(np.uint8)
    inverse_area = int(inverse_masks.sum())
    for mask in masks:
        regions.append(connected_components(mask.astype(np.uint8)))
    return regions, inverse_masks.astype(bool), inverse_area


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_thresholds: int = 200) -> float:
    assert masks.ndim == 3 and amaps.ndim == 3
    masks = (masks > 0.5).astype(np.uint8)
    if masks.sum() == 0:
        return 0.0
    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if max_th <= min_th:
        return 0.0
    regions, inverse_masks, inverse_area = _precompute_regions(masks)
    pros, fprs = [], []
    for threshold in np.linspace(min_th, max_th, num_thresholds, endpoint=False):
        binary = amaps > threshold
        region_scores = []
        for binary_amap, image_regions in zip(binary, regions):
            for rr, cc, area in image_regions:
                region_scores.append(float(binary_amap[rr, cc].sum()) / area)
        fp_pixels = np.logical_and(inverse_masks, binary).sum()
        pros.append(float(np.mean(region_scores)) if region_scores else 0.0)
        fprs.append(float(fp_pixels) / max(inverse_area, 1))
    fprs_arr = np.asarray(fprs, dtype=np.float32)
    pros_arr = np.asarray(pros, dtype=np.float32)
    keep = fprs_arr < 0.3
    if keep.sum() < 2:
        return 0.0
    fprs_arr = fprs_arr[keep]
    pros_arr = pros_arr[keep]
    max_fpr = float(fprs_arr.max())
    if max_fpr <= 0:
        return float(pros_arr.max())
    return auc_trapezoid(fprs_arr / max_fpr, pros_arr)


def evaluate_predictions(
    labels: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    item: str,
    distance: str,
    seconds: float,
    pro_thresholds: int = 200,
    skip_pro: bool = False,
) -> Dict[str, float]:
    if not np.isfinite(maps).all():
        bad_count = int((~np.isfinite(maps)).sum())
        raise ValueError(f"Prediction maps contain {bad_count} non-finite values.")
    pred_maps = normalize_np(maps)
    image_scores = normalize_np(pred_maps.reshape(pred_maps.shape[0], -1).max(axis=1))
    masks = (masks > 0.5).astype(np.uint8)
    return {
        "item": item,
        "distance_m": distance,
        "test_good": int((labels == 0).sum()),
        "test_anomaly": int((labels == 1).sum()),
        "seconds": round(seconds, 2),
        "I-AUROC": safe_roc_auc_score(labels, image_scores),
        "I-AP": safe_average_precision_score(labels, image_scores),
        "I-F1": f1_score_max(labels, image_scores),
        "P-AUROC": safe_roc_auc_score(masks.reshape(-1), pred_maps.reshape(-1)),
        "P-AP": safe_average_precision_score(masks.reshape(-1), pred_maps.reshape(-1)),
        "P-F1": f1_score_max(masks.reshape(-1), pred_maps.reshape(-1)),
        "AUC-PRO": float("nan") if skip_pro else compute_pro(masks, pred_maps, pro_thresholds),
    }


def add_mean_row(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    mean_row: Dict[str, float] = {
        "item": "mean",  # type: ignore[assignment]
        "distance_m": "mean",  # type: ignore[assignment]
        "test_good": sum(int(row["test_good"]) for row in rows),
        "test_anomaly": sum(int(row["test_anomaly"]) for row in rows),
        "seconds": round(sum(float(row["seconds"]) for row in rows), 2),
    }
    for key in METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        mean_row[key] = float(np.nan) if np.isnan(values).all() else float(np.nanmean(values))
    return rows + [mean_row]
