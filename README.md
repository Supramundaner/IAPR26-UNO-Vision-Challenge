# IAPR26 UNO Vision Challenge

This repository is for the Kaggle competition [IAPR 26 UNO Vision Challenge](https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge).

## Challenge Goal

The task is to recover the complete state of a multiplayer UNO game from each image. For every test image, the model must predict:

- the center card on the table
- the active player, meaning whose turn it is
- the set of cards held by each player

Each image is a table snapshot containing the center card, an active-player token, and cards held by four players.

## Data

The competition data explorer lists approximately 2.61 GB of files:

- `reference_images/`
- `test_images/`
- `train_images/`
- `sample_submission.csv`
- `train.csv`

Data is not tracked in Git. By default, use:

```powershell
python scripts/download_data.py
```

The script downloads the competition files through KaggleHub into `data/raw/` and copies the expected competition files there.

Kaggle competitions normally require authentication and accepting the competition rules before downloads are allowed. Do not commit Kaggle tokens to this repository.

If download fails, create a Kaggle API token from Kaggle account settings and save it outside the repository at:

```text
C:\Users\<your-user>\.kaggle\access_token
```

The file should contain only the token value. Then accept the rules on the competition page and rerun the script.

## Rules and Constraints

The challenge permits classical image processing, deep learning, or a combination of both. The following competition rules must be respected:

- External datasets are strictly forbidden.
- Test images must not be used for training, fine-tuning, or augmentation.
- Pre-trained models are strictly forbidden.
- Learning models must be trained from scratch using only the provided training and/or reference images.
- Models with more than 12 million parameters are strictly forbidden.

Compliance may be checked by the teaching assistants. Rule violations can disqualify the Kaggle submission and cause a project-grade penalty.

## Evaluation

Submissions are evaluated with a weighted score combining three components:

- Center card accuracy: 10%
- Active player accuracy: 10%
- Card prediction multiset F1 score: 80%

The final score is:

```text
Score = 0.1 * CenterAcc + 0.1 * ActiveAcc + 0.8 * F1
```

For card prediction, cards are evaluated as multisets across all players, so order does not matter and duplicate cards are allowed.

## Submission Format

The submission file must be a CSV with the following columns:

```csv
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
```

Requirements:

- `image_id` must exactly match the IDs in the test set.
- Cards are strings combining the first letter of the color and the value/action.
- Multiple cards in one hand are separated with semicolons.
- Empty hands must be written as `EMPTY`, not left blank.
- `active_player` must be one of `p1`, `p2`, `p3`, or `p4`.

Example card list:

```text
r_5;b_skip;y_2
```

Example rows:

```csv
image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards
L1000777,r_7,p2,wild;b_4,r_reverse;y_3,r_7;b_draw_2,EMPTY
L1000778,b_9,p4,EMPTY,b_1;y_7,b_3;g_2;y_reverse,b_5;r_2
```

## Suggested Project Direction

Two baseline implementation families are mentioned in the competition description:

- Classical pipeline: segmentation, object descriptors or feature extraction, then classification and game-state inference.
- Deep learning pipeline: architecture, loss, and data processing choices, trained from scratch under the parameter limit.

For either approach, the final notebook and presentation should include failure-case analysis and explain likely reasons for errors.

## Current Baseline

The implemented baseline follows [plan.md](plan.md):

- `ConditionalCardCNN`: predicts cards and an empty-hand confidence from an image plus one conditional token: `center`, `top`, `right`, `bottom`, or `left`. `EMPTY` is not a card class.
- `ActivePlayerCNN`: separately predicts the active player as one of `p1`, `p2`, `p3`, `p4`.
- Both models are trained from scratch and checked against the 12M parameter limit.

Run model and data-flow checks:

```powershell
python scripts/smoke_test.py
```

Train the card model:

```powershell
python scripts/train_card_model.py --epochs 20 --batch-size 8 --image-size 512 --device cuda:0 --wandb --wandb-run-name card-empty-cnn-512-loss-balanced --num-workers 2 --output artifacts/card_model.pt
```

Train the active-player model:

```powershell
python scripts/train_active_player.py --epochs 20 --batch-size 8 --image-size 384
```

Generate a submission after training:

```powershell
python scripts/generate_submission.py --output submissions/submission.csv
```

Training scripts default to `--device cuda:0`. Add `--wandb` to log runs to Weights & Biases.
