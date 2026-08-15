# 5B site configuration

Create a site file for one of the four schedules, then edit the storage, token, model snapshot,
and rendezvous paths:

```bash
./run_wm3d.sh 5b init validation10k /data/wm3d/control/h200_5b.env
./run_wm3d.sh 5b data-template /data/wm3d/control/h200_5b.env
./run_wm3d.sh 5b doctor /data/wm3d/control/h200_5b.env
```

Available presets are `canary1k`, `validation10k`, `validation100k`, and `formal600k`.
The wrapper derives the runtime profile, run identity, final checkpoint, and eval path from the
selected preset.

The default data contract includes the official LeRobot OXE collection and excludes AgiBotWorld
Beta. Set `INCLUDE_AGIBOT_BETA=YES` before `data-template` to include Beta.

The complete procedure is in `docs/WM3D_5B_SCALING.md`. Never commit the filled site file:
it can contain private filesystem and token locations.
