# 5B site configuration

Copy `h200_5b.env.example` outside the Git checkout, edit the storage, token, model snapshot,
and rendezvous paths, then run:

```bash
./run_wm3d.sh 5b doctor /shared/wm3d/control/h200_5b.env
```

The complete procedure is in `docs/WM3D_5B_SCALING.md`. Never commit the filled site file:
it can contain private filesystem and token locations.
