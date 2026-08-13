# Python package guide

- `wm3d.data` owns manifests, adapters, cache formats, grouped robot semantics, normalization, and
  deterministic sampling.
- `wm3d.encoders` owns frozen observation/task encoder boundaries.
- `wm3d.models` owns the single native world-model factory shared by all capacities.
- `wm3d.training` owns Stage0 objectives, distributed runtime, DCP, launch qualification, and eval.
- `wm3d.stage1_planner` owns the frozen-Stage0 planning system and must remain action-blind.
- Imports use `wm3d.*`; do not introduce versioned package aliases.
