# Cluster site configuration

For single-node 1B direct training on 8×H100:

```bash
./run_wm3d.sh 1b init canary1k /data/wm3d_1b_oxe/control/1b_canary.env
./run_wm3d.sh 1b data-template /data/wm3d_1b_oxe/control/1b_canary.env
./run_wm3d.sh 1b doctor /data/wm3d_1b_oxe/control/1b_canary.env
```

The 1B presets are `canary1k` and `formal100k`. The default uses the no-latent-cache
`direct_raw` path with P64 geometry and per-view P256 appearance, all official OXE
sources after DROID de-duplication, and no AgiBot options. See
`docs/WM3D_DIRECT_RAW.md`.

For multi-node 5B:

Create a site file for one of the three schedules, then edit the storage, token, model snapshot,
and rendezvous paths:

```bash
./run_wm3d.sh 5b init canary1k /data/wm3d/control/h200_5b.env direct_raw
./run_wm3d.sh 5b data-template /data/wm3d/control/h200_5b.env
./run_wm3d.sh 5b doctor /data/wm3d/control/h200_5b.env
```

Available presets are `canary1k`, `validation100k`, and `formal600k`. The required validation run
is `canary1k`; `validation100k` is an optional intermediate run.
The wrapper derives the runtime profile, run identity, final checkpoint, and eval path from the
selected preset.

`direct_raw` is the recommended no-visual-cache option. The final `init` argument can instead
be `streaming_raw` or `episode_cache`; omitting it keeps the `direct_raw` default. The selected
mode is sealed into the generated site file so colleagues do not need to edit wrapper internals.

The default data contract includes the official LeRobot OXE collection and excludes AgiBotWorld
Beta. Set `INCLUDE_AGIBOT_BETA=YES` before `data-template` to include Beta.

The direct data path is in `docs/WM3D_DIRECT_RAW.md`; the complete 5B procedure is in
`docs/WM3D_5B_SCALING.md`. Never commit the filled site file:
it can contain private filesystem and token locations.
