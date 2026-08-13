# 5B site configuration

Create a site file for one of the four schedules, then edit the storage, token, model snapshot,
and rendezvous paths:

```bash
./run_wm3d.sh 5b init validation10k /shared/wm3d/control/h200_5b.env
./run_wm3d.sh 5b doctor /shared/wm3d/control/h200_5b.env
```

Available presets are `canary1k`, `validation10k`, `validation100k`, and `formal600k`.
The wrapper derives the runtime profile, run identity, final checkpoint, and eval path from the
selected preset.

The complete procedure is in `docs/WM3D_5B_SCALING.md`. Never commit the filled site file:
it can contain private filesystem and token locations.
