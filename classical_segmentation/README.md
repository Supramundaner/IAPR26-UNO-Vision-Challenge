# Classical RGB + HSV + Component Segmentation

This folder contains a non-CNN baseline for segmenting UNO card printed regions.

The method first uses target RGB/HSV thresholds for red, yellow, blue, green, and black card areas. It then applies connected-component filtering with simple card-structure cues, mainly nearby white card border/oval pixels and dark support for wild-card details.

After a main card-face component is kept, small same-color components inside or very near that component's bounding box are also kept as satellites. This preserves card numbers and symbols that are separated from the main colored region by the white oval.

The final filtered mask is also converted into rough card bounding boxes. The bbox stage first extracts candidates separately per color class, then merges candidate boxes only when they overlap or contain each other. This avoids global all-color mask fusion while still connecting same-card fragments such as wild-card black regions, colored quadrants, and printed digits.

## Target Colors

```text
black:  rgb(47, 44, 38)
blue:   rgb(0, 154, 211)
yellow: rgb(237, 200, 1)
red:    rgb(229, 47, 51)
green:  rgb(97, 186, 65)
```

OpenCV HSV is derived from these RGB values inside the script. OpenCV hue uses `0..179`.

## Run

This script needs OpenCV in addition to the repository's existing NumPy dependency:

```powershell
pip install -r classical_segmentation/requirements.txt
```

From the repository root:

```powershell
python classical_segmentation/segment_card_colors.py --input data/raw/train_images/L1000902.jpg --output-dir outputs/classical_segmentation/L1000902
```

For a small batch of training images:

```powershell
python classical_segmentation/segment_card_colors.py --input data/raw/train_images --limit 8 --output-dir outputs/classical_segmentation/train_preview
```

Useful threshold options:

```powershell
--rgb-tolerance 64
--blue-rgb-tolerance 92
--black-rgb-tolerance 34
--max-black-rgb 88
--max-black-channel-spread 28
--hue-tolerance 10
--blue-hue-tolerance 16
--min-color-saturation 100
--min-color-value 85
--max-black-value 92
--max-black-saturation 85
--morph-kernel 3
--hole-close-kernel 7
```

Useful component-filter options:

```powershell
--component-mode card_like
--component-mode raw
--min-white-support 0.18
--min-inner-white-support 0.30
--max-component-area-frac 0.0065
--max-black-component-area-frac 0.012
--satellite-padding-frac 0.08
```

Useful bbox options:

```powershell
--bbox-close-kernel 95
--bbox-dilate-kernel 21
--min-box-area-frac 0.00045
--max-box-area-frac 0.04
--max-box-aspect-ratio 4.0
--box-merge-overlap-frac 0.18
```

`raw` keeps every thresholded pixel and is useful for debugging the color thresholds. `card_like` is the default and removes many background or token components.

## Outputs

For each image, the script writes:

- `<image_id>_raw_mask.png`: RGB/HSV threshold result before component filtering.
- `<image_id>_mask.png`: filtered card-like segmentation mask.
- `<image_id>_overlay.jpg`: mask overlaid on the original image.
- `<image_id>_components.csv`: component statistics and keep/drop reasons. `kept_satellite` means a small number/symbol component was recovered because it lies inside or near a kept card-face component.
- `<image_id>_card_boxes.csv`: rough bounding boxes extracted from per-color candidates; merged boxes have color values like `black+blue+green+red+yellow`.
- `<image_id>_card_boxes.jpg`: original image with bounding boxes drawn.
- `<image_id>_bbox_merge_mask.png`: intermediate mask used for bbox connected components.
