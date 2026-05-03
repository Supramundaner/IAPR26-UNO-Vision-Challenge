# Classical RGB + HSV Segmentation

This folder contains a first non-CNN baseline for segmenting UNO card printed regions.

The current goal is deliberately narrow: find pixels that look like red, yellow, blue, green, and black UNO card areas. It does not classify card values yet, and it does not try to recover the full game state.

## Method

`segment_card_colors.py` uses strict RGB + HSV thresholds:

- Red/yellow/blue/green use hue windows plus RGB dominance checks.
- Black wild-card regions use low HSV value, low saturation, and low RGB channel checks.
- A small optional morphology pass removes isolated pixels.

There is intentionally no learned model and no complex connected-component filtering in this baseline.

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

Useful options:

```powershell
--min-saturation 70
--min-value 90
--max-black-value 95
--max-black-saturation 120
--morph-kernel 3
```

## Outputs

For each image, the script writes:

- `<image_id>_mask.png`: colorized segmentation mask.
- `<image_id>_overlay.jpg`: mask overlaid on the original image.

