from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CARD_COLORS_BGR = {
    "red": (45, 45, 235),
    "yellow": (35, 215, 245),
    "blue": (225, 150, 30),
    "green": (80, 190, 75),
    "black": (35, 35, 35),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment UNO card printed regions with strict RGB + HSV thresholds."
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
        help="Directory for masks and overlays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images when --input is a directory.",
    )
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=70,
        help="Minimum HSV saturation for red/yellow/blue/green card pixels.",
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=90,
        help="Minimum HSV value for red/yellow/blue/green card pixels.",
    )
    parser.add_argument(
        "--max-black-value",
        type=int,
        default=95,
        help="Maximum HSV value for black wild-card pixels.",
    )
    parser.add_argument(
        "--max-black-saturation",
        type=int,
        default=120,
        help="Maximum HSV saturation for black wild-card pixels.",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Odd kernel size for light morphological cleanup. Use 0 to disable.",
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


def build_color_masks(
    image_bgr: np.ndarray,
    min_saturation: int,
    min_value: int,
    max_black_value: int,
    max_black_saturation: int,
    morph_kernel: int,
) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    b, g, r = cv2.split(image_bgr)

    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    saturated = (s >= min_saturation) & (v >= min_value)

    red = (
        saturated
        & ((h <= 10) | (h >= 170))
        & (r >= 145)
        & ((r16 - g16) >= 35)
        & ((r16 - b16) >= 35)
    )
    yellow = (
        saturated
        & (h >= 18)
        & (h <= 40)
        & (r >= 145)
        & (g >= 125)
        & (b <= 170)
        & (np.abs(r16 - g16) <= 80)
        & ((np.minimum(r16, g16) - b16) >= 30)
    )
    blue = (
        saturated
        & (h >= 88)
        & (h <= 112)
        & (b >= 105)
        & ((b16 - r16) >= 30)
        & ((g16 - r16) >= 5)
    )
    green = (
        saturated
        & (h >= 48)
        & (h <= 82)
        & (g >= 105)
        & ((g16 - r16) >= 15)
        & ((g16 - b16) >= 5)
    )
    black = (
        (v <= max_black_value)
        & (s <= max_black_saturation)
        & (r <= 105)
        & (g <= 105)
        & (b <= 105)
    )

    masks = {
        "red": red,
        "yellow": yellow,
        "blue": blue,
        "green": green,
        "black": black,
    }

    if morph_kernel <= 1:
        return masks

    kernel_size = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    cleaned = {}
    for color, mask in masks.items():
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        cleaned[color] = mask_u8 > 0
    return cleaned


def combine_masks(color_masks: dict[str, np.ndarray]) -> np.ndarray:
    combined = np.zeros(next(iter(color_masks.values())).shape, dtype=np.uint8)
    for index, color in enumerate(CARD_COLORS_BGR, start=1):
        combined[color_masks[color]] = index
    return combined


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


def process_image(
    image_path: Path,
    output_dir: Path,
    min_saturation: int,
    min_value: int,
    max_black_value: int,
    max_black_saturation: int,
    morph_kernel: int,
    overlay_alpha: float,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_id = image_path.stem
    masks = build_color_masks(
        image_bgr=image_bgr,
        min_saturation=min_saturation,
        min_value=min_value,
        max_black_value=max_black_value,
        max_black_saturation=max_black_saturation,
        morph_kernel=morph_kernel,
    )
    label_mask = combine_masks(masks)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{image_id}_mask.png"), colorize_mask(label_mask))
    cv2.imwrite(
        str(output_dir / f"{image_id}_overlay.jpg"),
        make_overlay(image_bgr, label_mask, overlay_alpha),
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )

    pixel_counts = ", ".join(f"{name}={int(mask.sum())}" for name, mask in masks.items())
    print(f"{image_id}: {pixel_counts} -> {output_dir}")


def main() -> None:
    args = parse_args()
    images = iter_images(args.input, args.limit)
    if not images:
        raise SystemExit(f"No images found in {args.input}")

    for image_path in images:
        process_image(
            image_path=image_path,
            output_dir=args.output_dir,
            min_saturation=args.min_saturation,
            min_value=args.min_value,
            max_black_value=args.max_black_value,
            max_black_saturation=args.max_black_saturation,
            morph_kernel=args.morph_kernel,
            overlay_alpha=args.overlay_alpha,
        )


if __name__ == "__main__":
    main()
