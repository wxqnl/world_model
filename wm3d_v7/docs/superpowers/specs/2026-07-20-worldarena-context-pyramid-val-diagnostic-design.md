# WorldArena Context-Pyramid Validation Diagnostic

## Purpose

Test whether a fixed context-pyramid renderer can recover high-frequency appearance from the initial observation without weakening WM3D motion. This is a validation-only diagnostic. It cannot read, generate, score, or select parameters on WorldArena episodes 40 through 49.

## Data boundary

- Source manifest: `manifests/benchmarks/worldarena_clean50_paper_v1.jsonl`.
- Allowed episodes: 36, 37, 38, and 39. These records belong to the training partition and serve as the held-out validation range declared in the WorldArena adaptation checkpoint.
- Forbidden episodes: 40 through 49. Encountering one is a fatal protocol error before model loading or video decoding.
- Fixed five-record panel: sort the 50 task names lexicographically, select task indices 0, 12, 24, 36, and 49, and assign episodes 36, 37, 38, 39, and 36 in that order.
- Inference may read the first RGB frame, task instruction, and physical actions. It may not read a future GT frame.
- GT future frames are available only to the post-generation PSNR scorer.

The diagnostic writes the five selected identities and the SHA-256 of their manifest rows before generation. Every result carries this selection audit.

## Alternatives considered

### Hard motion mask

Threshold the native RGB difference and dilate the binary mask. This is cheap, but small threshold changes move sharp boundaries and can create temporal flicker.

### Soft motion mask with one Laplacian band

Use a smooth ramp over the native RGB difference, blur it spatially, and inject one context residual band only outside motion. This is the selected design. It has two interpretable thresholds and one residual strength.

### Optical-flow-warped context residual

Warp high-frequency context with optical flow before blending. This may preserve detail on moving objects, but it adds another model, failure modes, and tunable confidence rules. It is outside this quick diagnostic.

## Renderer

For the initial RGB observation `C` and a native prediction `P_t`:

1. Resize `C` to the requested output resolution to obtain `C_high`.
2. Downsample `C` to 64 by 64 with area interpolation, then resize it to the output resolution with bicubic interpolation to obtain `C_low_up`.
3. Convert `P_t` to 64 by 64 if needed, then resize it to the output resolution with bicubic interpolation to obtain `P_low_up_t`.
4. Compute the fixed high-frequency residual `R = C_high - C_low_up`.
5. At 64 by 64, compute `D_t = mean_channel(abs(P_t - C_64))`.
6. Convert `D_t` into a soft motion mask with a linear ramp from `low` to `high`, clamp to `[0, 1]`, and apply Gaussian blur with sigma 1.0 native pixel.
7. Resize the mask to output resolution and render:

   `Y_t = clip(P_low_up_t + alpha * (1 - M_t) * R, 0, 1)`

The first output frame stays identical to the current renderer. The renderer has no GT input and no learned parameters.

## Locked parameter grid

The grid contains exactly six global configurations:

- `alpha`: 0.50, 0.75, or 1.00.
- motion ramp `(low, high)`: `(0.02, 0.08)` or `(0.04, 0.12)` in RGB `[0, 1]` units.
- Gaussian sigma: 1.0 native pixel for all configurations.
- Native comparison resolution: 64 by 64 for all configurations.

The implementation rejects a grid with any additional entry. It does not support per-task, per-video, or per-frame parameter overrides.

## Comparison pipeline

1. Generate the five native factual rollouts once from `formal1000_diverse3/ckpt/step_00001000.pt`.
2. Save the native RGB arrays and an inference audit. This cache contains no future GT RGB.
3. Render the current baseline and all six context-pyramid configurations from the same native arrays.
4. Run the official WorldArena implementations of image quality, JEPA similarity, dynamic degree, and motion smoothness on each five-video set.
5. Compute frame-index-aligned GT PSNR after generation. Resize each GT frame to the generated 640 by 480 resolution with `cv2.INTER_AREA`, matching the existing long-episode comparator. The scorer verifies equal frame counts and records that it is evaluation-only.
6. Produce a contact sheet for each of the five records with initial frame, GT, baseline, and selected candidate keyframes. The sheet is a reporting artifact and cannot affect selection.

## Aggregate selection and GO gate

Aggregate each metric over all five records before selection. The diagnostic never selects from an individual task result.

A candidate passes only when all conditions hold:

- Mean GT PSNR improves by at least 0.25 dB over the current renderer.
- Official mean image quality is not lower than the current renderer.
- JEPA similarity is at least 97 percent of the current renderer value.
- Dynamic degree is at least 97 percent of the current renderer value.
- Motion smoothness is at least 97 percent of the current renderer value.

Among passing candidates, select the largest mean PSNR gain. Treat gains within 0.02 dB as tied, then choose the smaller `alpha`, followed by lexicographic `(low, high)` order. If no configuration passes, report `NO-GO`. A `GO` result authorizes review only. It does not authorize a 500-video test rerun.

## Failure handling

- Reject any selected row outside episodes 36 through 39.
- Reject duplicate panel identities or any panel size other than five.
- Reject non-finite native RGB, renderer output, or metric values.
- Reject frame-count mismatches before PSNR computation.
- Reject missing official metric coverage or coverage other than five unique videos.
- Record subprocess commands, local model paths, hashes, and return codes.
- Run with local weights and offline Hugging Face settings.
- Do not overwrite the completed formal test result card.

## Tests

Unit tests cover:

- deterministic five-record selection and rejection of episodes 40 through 49;
- the exact six-entry grid;
- identity when `alpha` is zero;
- no context residual inside a full motion mask;
- exact residual injection inside a zero motion mask;
- finite output and `[0, 1]` bounds;
- aggregate-only GO logic and deterministic tie-break;
- PSNR frame-count validation;
- result audits that contain no future-GT inference field.

An integration smoke test renders synthetic native frames and builds a seven-way baseline-plus-grid report without loading the model or official test records.

## Artifacts

Write all outputs under:

`results/benchmarks/worldarena_bimanual_adapt/context_pyramid_val5/`

The directory contains the locked panel, native rollout cache, seven rendered video sets, official metric outputs, GT PSNR details, selection report, command provenance, and contact sheets. The final report includes `GO` or `NO-GO`, the selected global parameters when applicable, aggregate deltas, sample identities, and hashes.

## Non-goals

- No WorldArena episodes 40 through 49.
- No test-500 generation or scoring.
- No checkpoint update or renderer training.
- No optical flow, Wan renderer, or per-task tuning.
- No changes to the completed formal result card.
