# FreeFlow

Fast video object segmentation by predicting the attention output of SAM 2's
Hiera encoder instead of computing it, without retraining, distillation, or
checkpoint modification.

## What it does

SAM 2's Hiera image encoder runs on every frame and dominates inference cost.
Within a specific operating regime, the attention output of its six deepest
blocks (blocks 42–47) is temporally smooth enough to be predicted from the
recent history rather than computed from scratch. FreeFlow fits a small
ridge regression per block during a short warmup and replaces the attention
sub-call on subsequent frames.

## How it works

1. **Warmup.** The first 8 frames run the unmodified SAM 2 video predictor.
   For each of the 6 replaced blocks, the real attention output is recorded
   and decomposed into a unit-norm *phase* and a scalar *amplitude*.

2. **Fit.** Two closed-form ridge regressions per block are solved:

   - **Phase model** — predicts the current phase as a weighted sum of the
     last K = 4 phases, with weights that are affine in a 5-dimensional
     crop statistic (brightness, contrast, edge energy, mean flow magnitude,
     relative residual between crops).
   - **Log-amplitude model** — same structure, applied to the logarithm of
     the amplitude, with an additional bias term.

   The fit uses a Coupled Smoothness-Regularized Ridge (CSRR) that penalizes
   differences between adjacent lags and between adjacent blocks, so the six
   block solutions are jointly regularized.

3. **Inference.** From frame 9 onward, each replaced block's attention
   output is estimated as the sum of two terms:
   - a PACAP reconstruction `amplitude × phase`, and
   - a flow-warped copy of the previous frame's attention output.

   The two are blended with equal weight. The prediction is passed to the
   remaining layers of the block (layer norm, residual, MLP) exactly as the
   real attention output would be.

4. **ROI crop.** Both the encoder input and the prompt mask are restricted
   to a bounding box around the previous frame's mask, expanded by 30
   pixels and resized to 256×256. The decoder pastes the predicted mask
   back into full-frame coordinates.

Memory attention and the memory bank are bypassed on predicted frames.

## Scope

FreeFlow is designed for **single-object video object segmentation** on
commodity hardware, where:

- the object's appearance and position change within bounded per-frame
  bounds, and
- any transient occlusion resolves within a few frames.

Within this envelope, the predictor matches the encoder+decoder-only
baseline while also replacing the encoder's internal attention, and its
cost scales with the crop size rather than with the full frame.

## Limitations

The current method does not handle:

- **Multi-object scenes with heavy overlap** — the ROI crop merges the
  instances, and the single-channel mask prompt cannot separate them.
- **Long-range target reappearance** — if the object leaves the crop and
  returns at a distant location more than K = 4 frames later, the
  predictor's history no longer contains the target.
- **Sustained occlusion** — if the target is hidden for more than K
  consecutive frames, the same issue occurs.
- **Large per-frame motion** — if the object moves outside the bounding
  box plus margin between consecutive frames, the mask is lost entirely
  rather than degraded gracefully.

The boundary metric F suffers more than the region metric J when the
attention output is predicted, because the predictor produces contours
that are structurally correct but slightly smoother than the real ones.

## Usage

Edit `VIDEO_PATH` and `GT_MASK_DIR` at the top of `FreeFlow.py`. Both are
optional in the sense that if `GT_MASK_DIR` is missing, metrics are
skipped and only timing is reported.

```bash
python FreeFlow.py
