from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1]))

from uno_vision.config import CONDITION_TOKENS, RAW_DATA_DIR
from uno_vision.data import crop_by_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, default=RAW_DATA_DIR / "train.csv")
    parser.add_argument("--image-dir", type=Path, default=RAW_DATA_DIR / "train_images")
    parser.add_argument("--output", type=Path, default=Path("outputs/roi_crops.png"))
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop-width", type=int, default=220)
    parser.add_argument("--crop-height", type=int, default=160)
    return parser.parse_args()


def draw_label(image: Image.Image, label: str) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((0, 0, box[2] + 8, box[3] + 8), fill=(255, 255, 255))
    draw.text((4, 4), label, fill=(0, 0, 0), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.train_csv).sample(n=args.num_images, random_state=args.seed)
    tokens = CONDITION_TOKENS
    cell_width = args.crop_width
    cell_height = args.crop_height
    header_height = 28
    row_label_width = 110

    sheet = Image.new(
        "RGB",
        (row_label_width + cell_width * len(tokens), header_height + cell_height * len(df)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col, token in enumerate(tokens):
        x = row_label_width + col * cell_width
        draw.text((x + 6, 8), token, fill=(0, 0, 0), font=font)

    for row_index, row in enumerate(df.itertuples(index=False)):
        image_id = str(row.image_id)
        image = Image.open(args.image_dir / f"{image_id}.jpg").convert("RGB")
        y = header_height + row_index * cell_height
        draw.text((6, y + 8), image_id, fill=(0, 0, 0), font=font)
        for col, token in enumerate(tokens):
            crop = crop_by_condition(image, token).resize((cell_width, cell_height), Image.Resampling.LANCZOS)
            crop = draw_label(crop, token)
            x = row_label_width + col * cell_width
            sheet.paste(crop, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
