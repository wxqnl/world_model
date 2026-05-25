# Data

Three buckets. Total budget ≤50 GB. Manifests under `manifests/` are the ground
truth — raw frames live outside this repo (see `.gitignore`). Manifest entries
use absolute paths so the pipeline is portable across runs.

## Set A — robot manipulation (primary)

- Target: 30–50 clips, 60–150 frames each (2–5 s @ 30 fps).
- Source candidates:
  - `lerobot/droid_100` on Hugging Face (preferred: small, preselected).
  - Subset of Open X-Embodiment from https://robotics-transformer-x.github.io/.
- Selection criteria:
  - Third-person camera views only (no wrist-cam-only clips).
  - Balanced across: grasping, pushing, pouring, articulated-object manipulation.
  - Skip clips with severe motion blur or camera cuts.

## Set B — static scenes (baseline)

- Target: 20 scenes × ~20 frames each.
- Source candidates:
  - ScanNet small subset (preferred: has GT depth).
  - 7-Scenes.
- GT depth path recorded per frame under `gt_depth:` in the manifest entry.

## Set C — autonomous driving (cross-domain probe)

- Target: 10–20 clips × ~60 frames.
- Source candidates: nuScenes mini, KITTI subset, Waymo Open mini.
- Third-person, forward-facing camera; clips with visible dynamic agents preferred.

## Manifest format

```json
[
  {
    "clip_id": "droid_kitchen_0001",
    "set": "A",
    "frames": ["/abs/path/frame_0000.jpg", "/abs/path/frame_0001.jpg", ...],
    "fps": 30,
    "meta": {"dataset": "droid", "task": "pour_cup"},
    "gt_depth": null
  }
]
```

For Set B, `gt_depth` is a parallel list of GT depth PNG paths (ScanNet: uint16 mm).

## License notes

- **DROID / Open X-Embodiment**: CC-BY variants; citation required.
- **ScanNet**: requires TOS acceptance and a signed form returned by email.
- **nuScenes**: requires free account + accepting non-commercial license.
- **Waymo Open**: requires Google account + TOS.

Manifests should not include proprietary frames; link to the canonical source
so the diagnostic is reproducible by anyone who has accepted the TOS.
