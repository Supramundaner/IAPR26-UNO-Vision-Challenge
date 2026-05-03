from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CARD_COLORS_BGR = {
    "red": (51, 47, 229),
    "yellow": (1, 200, 237),
    "blue": (211, 154, 0),
    "green": (65, 186, 97),
    "black": (38, 44, 47),
}

TARGET_RGB = {
    "black": (47, 44, 38),
    "blue": (0, 154, 211),
    "yellow": (237, 200, 1),
    "red": (229, 47, 51),
    "green": (97, 186, 65),
}

TARGET_HSV = {
    # OpenCV HSV: hue is 0..179, saturation/value are 0..255.
    "black": (20, 49, 47),
    "blue": (98, 255, 211),
    "yellow": (25, 254, 237),
    "red": (179, 203, 229),
    "green": (52, 166, 186),
}


@dataclass(frozen=True)
class ComponentStats:
    image_id: str
    color: str
    kept: bool
    reason: str
    area: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    fill_ratio: float
    aspect_ratio: float
    white_support: float
    inner_white_support: float
    dark_support: float
    color_support: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment UNO card regions using target RGB/HSV and card-like components."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Image file or directory containing jpg/png images.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/classical_segmentation"),
        type=Path,
        help="Directory for masks, overlays, and component CSV files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images when --input is a directory.",
    )
    parser.add_argument(
        "--rgb-tolerance",
        type=int,
        default=64,
        help="Per-channel RGB tolerance around the target non-black card colors.",
    )
    parser.add_argument(
        "--blue-rgb-tolerance",
        type=int,
        default=110,
        help="Per-channel RGB tolerance around the target blue card color.",
    )
    parser.add_argument(
        "--yellow-rgb-tolerance",
        type=int,
        default=88,
        help="Per-channel RGB tolerance around the target yellow card color.",
    )
    parser.add_argument(
        "--black-rgb-tolerance",
        type=int,
        default=34,
        help="Per-channel RGB tolerance around the target black card color.",
    )
    parser.add_argument(
        "--max-black-rgb",
        type=int,
        default=88,
        help="Maximum RGB channel value for neutral dark black-card pixels.",
    )
    parser.add_argument(
        "--max-black-channel-spread",
        type=int,
        default=28,
        help="Maximum max(R,G,B)-min(R,G,B) spread for neutral dark black-card pixels.",
    )
    parser.add_argument(
        "--hue-tolerance",
        type=int,
        default=10,
        help="OpenCV hue tolerance around the target non-black card colors.",
    )
    parser.add_argument(
        "--blue-hue-tolerance",
        type=int,
        default=22,
        help="OpenCV hue tolerance around the target blue card color.",
    )
    parser.add_argument(
        "--yellow-hue-tolerance",
        type=int,
        default=16,
        help="OpenCV hue tolerance around the target yellow card color.",
    )
    parser.add_argument(
        "--min-color-saturation",
        type=int,
        default=100,
        help="Minimum HSV saturation for red/yellow/blue/green card pixels.",
    )
    parser.add_argument(
        "--min-color-value",
        type=int,
        default=85,
        help="Minimum HSV value for red/yellow/blue/green card pixels.",
    )
    parser.add_argument(
        "--max-black-value",
        type=int,
        default=92,
        help="Maximum HSV value for black wild-card pixels.",
    )
    parser.add_argument(
        "--max-black-saturation",
        type=int,
        default=85,
        help="Maximum HSV saturation for black wild-card pixels.",
    )
    parser.add_argument(
        "--min-black-area-frac",
        type=float,
        default=0.00022,
        help="Reject black components smaller than this image-area fraction.",
    )
    parser.add_argument(
        "--component-mode",
        choices=("card_like", "raw"),
        default="card_like",
        help="Whether to filter connected components or keep every thresholded pixel.",
    )
    parser.add_argument(
        "--max-component-area-frac",
        type=float,
        default=0.0065,
        help="Reject non-black components larger than this fraction of image area.",
    )
    parser.add_argument(
        "--max-black-component-area-frac",
        type=float,
        default=0.012,
        help="Reject black components larger than this fraction of image area.",
    )
    parser.add_argument(
        "--min-white-support",
        type=float,
        default=0.18,
        help="Minimum nearby white-card-pixel ratio for a component.",
    )
    parser.add_argument(
        "--min-inner-white-support",
        type=float,
        default=0.30,
        help="Minimum white-card-pixel ratio inside a normal color component bbox.",
    )
    parser.add_argument(
        "--min-dark-support",
        type=float,
        default=0.08,
        help="Minimum nearby dark-card-pixel ratio for small wild-card color components.",
    )
    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.18,
        help="Minimum component area divided by its bounding box area.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Odd kernel size for light morphological cleanup. Use 0 to disable.",
    )
    parser.add_argument(
        "--hole-close-kernel",
        type=int,
        default=7,
        help="Odd kernel size for filling small bright gaps inside thresholded card areas. Use 0 to disable.",
    )
    parser.add_argument(
        "--post-min-region-area",
        type=int,
        default=5500,
        help="Remove final same-color connected regions smaller than this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--post-dilate-kernel",
        type=int,
        default=10,
        help="Odd kernel size for final same-color dilation after small-region removal. Use 0 to disable.",
    )
    parser.add_argument(
        "--post-dilate-iterations",
        type=int,
        default=1,
        help="Number of final dilation iterations after small-region removal. Use 0 to disable.",
    )
    parser.add_argument(
        "--satellite-padding-frac",
        type=float,
        default=0.08,
        help="Extra bbox padding used to keep small number/symbol components near a kept card-face component.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Mask opacity in overlay images.",
    )
    return parser.parse_args()


def iter_images(input_path: Path, limit: int | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in suffixes)
    if limit is not None:
        images = images[:limit]
    return images


def rgb_window(
    name: str, r: np.ndarray, g: np.ndarray, b: np.ndarray, tolerance: int
) -> np.ndarray:
    target_r, target_g, target_b = TARGET_RGB[name]
    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    return (
        (np.abs(r16 - target_r) <= tolerance)
        & (np.abs(g16 - target_g) <= tolerance)
        & (np.abs(b16 - target_b) <= tolerance)
    )


def hue_window(name: str, h: np.ndarray, tolerance: int) -> np.ndarray:
    target_h = TARGET_HSV[name][0]
    hue_delta = np.abs(h.astype(np.int16) - target_h)
    circular_delta = np.minimum(hue_delta, 180 - hue_delta)
    return circular_delta <= tolerance


def cleanup_masks(
    masks: dict[str, np.ndarray], morph_kernel: int, hole_close_kernel: int
) -> dict[str, np.ndarray]:
    if morph_kernel <= 1 and hole_close_kernel <= 1:
        return masks

    denoise_kernel = None
    if morph_kernel > 1:
        kernel_size = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1
        denoise_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    hole_kernel = None
    if hole_close_kernel > 1:
        kernel_size = (
            hole_close_kernel if hole_close_kernel % 2 == 1 else hole_close_kernel + 1
        )
        hole_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

    cleaned = {}
    for color, mask in masks.items():
        mask_u8 = mask.astype(np.uint8) * 255
        if denoise_kernel is not None:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, denoise_kernel, iterations=1)
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, denoise_kernel, iterations=1)
        if hole_kernel is not None:
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, hole_kernel, iterations=1)
        cleaned[color] = mask_u8 > 0
    return cleaned


def build_color_masks(
    image_bgr: np.ndarray,
    rgb_tolerance: int,
    blue_rgb_tolerance: int,
    yellow_rgb_tolerance: int,
    black_rgb_tolerance: int,
    max_black_rgb: int,
    max_black_channel_spread: int,
    hue_tolerance: int,
    blue_hue_tolerance: int,
    yellow_hue_tolerance: int,
    min_color_saturation: int,
    min_color_value: int,
    max_black_value: int,
    max_black_saturation: int,
    morph_kernel: int,
    hole_close_kernel: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b, g, r = cv2.split(image_bgr)

    color_gate = (s >= min_color_saturation) & (v >= min_color_value)
    masks = {}
    for color in ("red", "yellow", "blue", "green"):
        if color == "blue":
            rgb_tol = blue_rgb_tolerance
            hue_tol = blue_hue_tolerance
        elif color == "yellow":
            rgb_tol = yellow_rgb_tolerance
            hue_tol = yellow_hue_tolerance
        else:
            rgb_tol = rgb_tolerance
            hue_tol = hue_tolerance
        masks[color] = (
            color_gate
            & rgb_window(color, r, g, b, rgb_tol)
            & hue_window(color, h, hue_tol)
        )
    max_channel = np.maximum.reduce([r, g, b])
    min_channel = np.minimum.reduce([r, g, b])
    target_black = rgb_window("black", r, g, b, black_rgb_tolerance)
    tightly_neutral_black = (
        (max_channel <= max_black_rgb)
        & ((max_channel.astype(np.int16) - min_channel.astype(np.int16)) <= max_black_channel_spread)
    )
    masks["black"] = target_black & tightly_neutral_black & (v <= max_black_value) & (
        s <= max_black_saturation
    )

    masks = cleanup_masks(masks, morph_kernel, hole_close_kernel)
    color_support_mask = (
        masks["red"] | masks["yellow"] | masks["blue"] | masks["green"]
    )
    white_support_mask = (s <= 55) & (v >= 145) & (r >= 130) & (g >= 130) & (b >= 125)
    dark_support_mask = (v <= max_black_value) & (s <= max_black_saturation)
    return masks, white_support_mask, dark_support_mask, color_support_mask


def filter_components(
    image_id: str,
    color_masks: dict[str, np.ndarray],
    white_support_mask: np.ndarray,
    dark_support_mask: np.ndarray,
    color_support_mask: np.ndarray,
    component_mode: str,
    max_component_area_frac: float,
    max_black_component_area_frac: float,
    min_white_support: float,
    min_inner_white_support: float,
    min_dark_support: float,
    min_fill_ratio: float,
    satellite_padding_frac: float,
    min_black_area_frac: float,
) -> tuple[dict[str, np.ndarray], list[ComponentStats]]:
    if component_mode == "raw":
        return color_masks, []

    height, width = white_support_mask.shape
    image_area = height * width
    min_area = max(90, int(image_area * 0.000018))
    min_black_area = max(min_area, int(image_area * min_black_area_frac))
    output_masks: dict[str, np.ndarray] = {}
    rows: list[ComponentStats] = []

    for color, mask in color_masks.items():
        num_labels, labels, component_stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        filtered = np.zeros_like(mask, dtype=bool)
        max_area = int(
            image_area
            * (
                max_black_component_area_frac
                if color == "black"
                else max_component_area_frac
            )
        )

        component_decisions = []
        primary_bboxes = []

        for label in range(1, num_labels):
            x, y, w, h, area = component_stats[label]
            bbox_area = max(1, int(w * h))
            fill_ratio = float(area / bbox_area)
            aspect_ratio = float(max(w / max(h, 1), h / max(w, 1)))

            pad = int(max(w, h) * 0.35) + 6
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(width, x + w + pad)
            y1 = min(height, y + h + pad)
            white_support = float(white_support_mask[y0:y1, x0:x1].mean())
            inner_white_support = float(white_support_mask[y : y + h, x : x + w].mean())
            dark_support = float(dark_support_mask[y0:y1, x0:x1].mean())
            color_support = float(color_support_mask[y0:y1, x0:x1].mean())
            small_wild_like = area <= int(image_area * 0.0030)
            if color == "black":
                has_card_support = (
                    white_support >= max(min_white_support, 0.22)
                    and inner_white_support >= 0.08
                    and fill_ratio <= 0.85
                )
            else:
                has_card_support = inner_white_support >= min_inner_white_support or (
                    small_wild_like and dark_support >= min_dark_support
                )

            kept = True
            reason = "kept"
            if color == "black" and area < min_black_area:
                kept = False
                reason = "black_too_small"
            elif area < min_area:
                kept = False
                reason = "too_small"
            elif area > max_area:
                kept = False
                reason = "too_large"
            elif aspect_ratio > (3.0 if color == "black" else 2.25):
                kept = False
                reason = "too_thin"
            elif fill_ratio < min_fill_ratio:
                kept = False
                reason = "too_sparse"
            elif white_support < min_white_support:
                kept = False
                reason = "low_white_support"
            elif not has_card_support:
                kept = False
                reason = "low_card_support"

            if kept:
                filtered[labels == label] = True
                primary_bboxes.append((x, y, w, h))

            component_decisions.append(
                {
                    "label": label,
                    "kept": kept,
                    "reason": reason,
                    "area": int(area),
                    "bbox_x": int(x),
                    "bbox_y": int(y),
                    "bbox_w": int(w),
                    "bbox_h": int(h),
                    "fill_ratio": fill_ratio,
                    "aspect_ratio": aspect_ratio,
                    "white_support": white_support,
                    "inner_white_support": inner_white_support,
                    "dark_support": dark_support,
                    "color_support": color_support,
                }
            )

        for decision in component_decisions:
            if decision["kept"]:
                continue

            x = decision["bbox_x"]
            y = decision["bbox_y"]
            w = decision["bbox_w"]
            h = decision["bbox_h"]
            cx = x + w / 2.0
            cy = y + h / 2.0
            satellite = False
            for px, py, pw, ph in primary_bboxes:
                pad = int(max(pw, ph) * satellite_padding_frac) + 4
                if (px - pad) <= cx <= (px + pw + pad) and (py - pad) <= cy <= (py + ph + pad):
                    satellite = True
                    break

            if satellite:
                decision["kept"] = True
                decision["reason"] = "kept_satellite"
                filtered[labels == decision["label"]] = True

        for decision in component_decisions:
            rows.append(
                ComponentStats(
                    image_id=image_id,
                    color=color,
                    kept=decision["kept"],
                    reason=decision["reason"],
                    area=decision["area"],
                    bbox_x=decision["bbox_x"],
                    bbox_y=decision["bbox_y"],
                    bbox_w=decision["bbox_w"],
                    bbox_h=decision["bbox_h"],
                    fill_ratio=decision["fill_ratio"],
                    aspect_ratio=decision["aspect_ratio"],
                    white_support=decision["white_support"],
                    inner_white_support=decision["inner_white_support"],
                    dark_support=decision["dark_support"],
                    color_support=decision["color_support"],
                )
            )

        output_masks[color] = filtered

    return output_masks, rows


def combine_masks(color_masks: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.zeros(next(iter(color_masks.values())).shape, dtype=np.uint8)
    for index, color in enumerate(CARD_COLORS_BGR, start=1):
        combined[color_masks[color]] = index
    return combined


def odd_kernel(size: int) -> tuple[int, int] | None:
    if size <= 1:
        return None
    kernel_size = size if size % 2 == 1 else size + 1
    return (kernel_size, kernel_size)


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask.copy()

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(mask, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            filtered[labels == label] = True
    return filtered


def postprocess_masks(
    color_masks: dict[str, np.ndarray],
    min_region_area: int,
    dilate_kernel: int,
    dilate_iterations: int,
) -> dict[str, np.ndarray]:
    kernel_size = odd_kernel(dilate_kernel)
    dilation_kernel = None
    if kernel_size is not None and dilate_iterations > 0:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    output_masks: dict[str, np.ndarray] = {}
    for color, mask in color_masks.items():
        processed = remove_small_regions(mask, min_region_area)
        if dilation_kernel is not None:
            processed_u8 = processed.astype(np.uint8) * 255
            processed_u8 = cv2.dilate(
                processed_u8, dilation_kernel, iterations=dilate_iterations
            )
            processed = processed_u8 > 0
        output_masks[color] = processed
    return output_masks


def colorize_mask(label_mask: np.ndarray) -> np.ndarray:
    colorized = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    for index, color in enumerate(CARD_COLORS_BGR, start=1):
        colorized[label_mask == index] = CARD_COLORS_BGR[color]
    return colorized


def make_overlay(image_bgr: np.ndarray, label_mask: np.ndarray, alpha: float) -> np.ndarray:
    colorized = colorize_mask(label_mask)
    overlay = image_bgr.copy()
    active = label_mask > 0
    overlay[active] = cv2.addWeighted(
        image_bgr[active], 1.0 - alpha, colorized[active], alpha, 0
    )
    return overlay


def save_component_csv(path: Path, rows: list[ComponentStats]) -> None:
    fieldnames = list(ComponentStats.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def process_image(image_path: Path, args: argparse.Namespace) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_id = image_path.stem
    color_masks, white_support_mask, dark_support_mask, color_support_mask = build_color_masks(
        image_bgr=image_bgr,
        rgb_tolerance=args.rgb_tolerance,
        blue_rgb_tolerance=args.blue_rgb_tolerance,
        yellow_rgb_tolerance=args.yellow_rgb_tolerance,
        black_rgb_tolerance=args.black_rgb_tolerance,
        max_black_rgb=args.max_black_rgb,
        max_black_channel_spread=args.max_black_channel_spread,
        hue_tolerance=args.hue_tolerance,
        blue_hue_tolerance=args.blue_hue_tolerance,
        yellow_hue_tolerance=args.yellow_hue_tolerance,
        min_color_saturation=args.min_color_saturation,
        min_color_value=args.min_color_value,
        max_black_value=args.max_black_value,
        max_black_saturation=args.max_black_saturation,
        morph_kernel=args.morph_kernel,
        hole_close_kernel=args.hole_close_kernel,
    )
    raw_label_mask = combine_masks(color_masks)
    filtered_masks, component_rows = filter_components(
        image_id=image_id,
        color_masks=color_masks,
        white_support_mask=white_support_mask,
        dark_support_mask=dark_support_mask,
        color_support_mask=color_support_mask,
        component_mode=args.component_mode,
        max_component_area_frac=args.max_component_area_frac,
        max_black_component_area_frac=args.max_black_component_area_frac,
        min_white_support=args.min_white_support,
        min_inner_white_support=args.min_inner_white_support,
        min_dark_support=args.min_dark_support,
        min_fill_ratio=args.min_fill_ratio,
        satellite_padding_frac=args.satellite_padding_frac,
        min_black_area_frac=args.min_black_area_frac,
    )
    filtered_masks = postprocess_masks(
        color_masks=filtered_masks,
        min_region_area=args.post_min_region_area,
        dilate_kernel=args.post_dilate_kernel,
        dilate_iterations=args.post_dilate_iterations,
    )
    label_mask = combine_masks(filtered_masks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_dir / f"{image_id}_raw_mask.png"), colorize_mask(raw_label_mask))
    cv2.imwrite(str(args.output_dir / f"{image_id}_mask.png"), colorize_mask(label_mask))
    cv2.imwrite(
        str(args.output_dir / f"{image_id}_overlay.jpg"),
        make_overlay(image_bgr, label_mask, args.overlay_alpha),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    save_component_csv(args.output_dir / f"{image_id}_components.csv", component_rows)

    raw_pixels = int((raw_label_mask > 0).sum())
    filtered_pixels = int((label_mask > 0).sum())
    print(
        f"{image_id}: raw_pixels={raw_pixels} "
        f"filtered_pixels={filtered_pixels} -> {args.output_dir}"
    )


def main() -> None:
    args = parse_args()
    images = iter_images(args.input, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    for image_path in images:
        process_image(image_path, args)


if __name__ == "__main__":
    main()
