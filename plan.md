# Project Plan: Conditional CNN for UNO Game-State Recovery

## Objective

Build a competition-compliant baseline for the IAPR 26 UNO Vision Challenge. Given one full-table UNO image, the system must predict:

1. the center card
2. the active player
3. the cards held by each of four players

The approach uses only the provided competition data, trains from scratch, and keeps every learning model below the 12 million parameter limit.

## High-Level Architecture

The final prediction pipeline has two separate neural components:

1. Conditional card CNN
   - Input: one RGB image plus one conditional token.
   - Conditional token classes: `center`, `top`, `right`, `bottom`, `left`.
   - Output: a multi-label vector over the UNO card vocabulary plus an empty-hand confidence.
   - Purpose: predict the center card or the set of cards held in a spatial player region, and detect empty hands for player regions.

2. Active-player CNN
   - Input: one RGB image.
   - Output: one of `p1`, `p2`, `p3`, `p4`.
   - Purpose: detect the active-player token independently from card recognition.

This split keeps the active-player problem clean while letting the conditional card model learn the closely related card-presence and empty-hand decisions.

## Label Design

### Card Vocabulary

The card vocabulary is inferred from `train.csv` by collecting every card string appearing in:

- `center_card`
- `player_1_cards`
- `player_2_cards`
- `player_3_cards`
- `player_4_cards`

`EMPTY` is not a card class. It is represented by a separate binary empty-hand target on player-hand conditional samples.

Card examples include:

- `r_5`
- `b_skip`
- `y_reverse`
- `wild`
- `draw_4`

### Conditional Card Samples

Each training image produces five card-training samples:

| Conditional token | Card target | Empty target |
| --- | --- |
| `center` | one-hot center-card vector | ignored |
| `top` | multi-hot vector for one player's cards | 1 if that hand is `EMPTY`, else 0 |
| `right` | multi-hot vector for one player's cards | 1 if that hand is `EMPTY`, else 0 |
| `bottom` | multi-hot vector for one player's cards | 1 if that hand is `EMPTY`, else 0 |
| `left` | multi-hot vector for one player's cards | 1 if that hand is `EMPTY`, else 0 |

The implementation uses a fixed mapping from player IDs to spatial tokens:

| Player | Conditional token |
| --- | --- |
| `p1` | `bottom` |
| `p2` | `left` |
| `p3` | `top` |
| `p4` | `right` |

This mapping is intentionally centralized in code so it can be adjusted after visual inspection if the table layout differs.

### Active Player Samples

Each training image produces one active-player sample:

- input: image
- target: integer class for `p1`, `p2`, `p3`, or `p4`

## Model Design

### Shared Design Principles

- No pre-trained weights.
- Small convolutional networks trained from scratch.
- Parameter count must be below 12 million for each model.
- Use resized images to make training feasible on a laptop.
- Keep the implementation simple enough to debug and improve quickly.

### Conditional Card CNN

The conditional CNN contains:

1. a convolutional image encoder
2. a conditional-token embedding
3. a small fusion MLP
4. a sigmoid multi-label card head
5. a sigmoid empty-hand head

Conditioning is done by concatenating the pooled image feature with the learned token embedding. This is simple, parameter-efficient, and easy to inspect.

Loss:

```text
card BCEWithLogitsLoss + empty_loss_weight * masked empty BCEWithLogitsLoss
```

The empty loss is masked out for the `center` token because the center-card prediction is never an empty-hand decision.

Inference:

- for `center`, choose the highest-probability card, because exactly one center card is expected
- for player hands, first use the empty-hand confidence
- if empty confidence is above the empty threshold, output `EMPTY`
- otherwise choose all cards above the card threshold
- if the model predicts non-empty but no card crosses the card threshold, fall back to `EMPTY` as a valid CSV value

Threshold selection:

- default threshold: `0.5`
- default empty threshold: `0.5`
- later improvement: tune per-card or global threshold on a validation split using multiset F1

### Active-Player CNN

The active-player model is a smaller CNN with a four-class linear head.

Loss:

```text
CrossEntropyLoss
```

Inference:

- choose the highest-probability class among `p1`, `p2`, `p3`, `p4`

## Training Strategy

### Data Split

Use a deterministic train/validation split over image IDs:

- default validation fraction: 20%
- seed: 42

Splitting is done by image ID before expanding conditional card samples. This prevents samples from the same image appearing in both train and validation sets.

### Augmentation

Allowed augmentations must use only provided training images. Initial baseline augmentations:

- resize
- small color jitter
- small affine perturbation
- optional horizontal/vertical flips are disabled by default because they can invalidate the spatial player-token mapping

### Optimization

Suggested defaults:

- optimizer: AdamW
- image size: 384
- batch size: 8 for card model, 8 for active-player model
- epochs: 20
- learning rate: `1e-3`
- weight decay: `1e-4`

These are baseline values; actual settings can be adjusted based on GPU/CPU availability.

## Evaluation

Validation should track:

1. center-card accuracy from the conditional model with token `center`
2. active-player accuracy from the active-player model
3. card multiset F1 over the four player-hand predictions
4. final weighted score:

```text
Score = 0.1 * CenterAcc + 0.1 * ActiveAcc + 0.8 * F1
```

The first implementation focuses on producing a trainable pipeline and valid submission CSV. More exact competition-style validation can be expanded after baseline training works.

## Submission Generation

For each test image:

1. run the conditional card CNN with token `center`
2. run the conditional card CNN with tokens `bottom`, `left`, `top`, and `right`
3. map spatial predictions back to `player_1_cards` through `player_4_cards`
4. run the active-player CNN
5. write one CSV row:

```csv
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
```

Player-card fields are semicolon-separated. Empty hands are written as `EMPTY`.

## Initial Implementation Tasks

1. Build card vocabulary and label encoders from `train.csv`.
2. Implement PyTorch datasets for conditional card/empty samples and active-player samples.
3. Implement small CNN backbones and heads, including the conditional model's card and empty heads.
4. Add scripts for:
   - parameter counting
   - card-model training
   - active-player training
   - submission generation
5. Add smoke tests that verify:
   - data can be loaded
   - models can run a forward pass
   - parameter counts are below 12M
   - a sample submission file can be generated from randomly initialized models

## Expected First Baseline

The first model is not expected to be leaderboard-competitive immediately. It is intended to create a clean, legal, end-to-end baseline. The strongest next improvements will likely come from:

- verifying the spatial player-token mapping
- adding local crops or attention to card regions
- tuning thresholds for player hands
- improving image resolution and augmentations
- using reference images to build card-specific validation diagnostics
- adding post-processing constraints based on known UNO game structure
