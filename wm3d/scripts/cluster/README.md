# Cluster helpers

`wm3d_1b.sh` and `wm3d_5b.sh` read one site-local environment file and compose the existing
WM3D commands. The 1B wrapper selects the same implementation with the 1B model/runtime defaults;
it does not create a second trainer.
It handles source download, task/cache preparation, runtime materialization, distributed launch,
status, and final verification. It does not bypass the human adapter audit.

Run `./run_wm3d.sh 1b help` or `./run_wm3d.sh 5b help`. The ordered procedures are in
`docs/WM3D_1B_STREAMING.md` and `docs/WM3D_5B_SCALING.md`.
