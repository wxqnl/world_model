# 5B cluster helpers

`wm3d_5b.sh` reads one site-local environment file and composes the existing WM3D commands.
It handles source download, task/cache preparation, runtime materialization, distributed launch,
status, and final verification. It does not bypass the human adapter audit.

Run `./run_wm3d.sh 5b help` for the command list and see
`docs/WM3D_5B_SCALING.md` for the ordered multi-node procedure.
