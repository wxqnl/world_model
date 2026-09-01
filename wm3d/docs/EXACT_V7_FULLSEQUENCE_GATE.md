# Exact V7 Full-Sequence Runtime Gate

`run_exact_v7_fullsequence_gate.py` is a fail-closed, no-training topology
check. It runs one sealed current-data batch and does not inspect source text,
change the model, rebuild caches, or start an optimizer.

It requires all of the following runtime behavior:

- factual StateStream and ActionStream enter block 0 as full
  `task + T*P + K*G` sequences;
- every full block preserves that sequence, runs bidirectionally, and masks
  invalid `K*G` tokens as both keys and queries;
- every configured bridge runs exactly once in the frozen schedule and both
  updates read the same pre-bridge state/action pair;
- decoder memory is the complete post-block StateStream;
- multi-group query/RGB conditioning preserves the group axis and never uses a
  raw group mean;
- the policy path stays rank-4 factorized and is exactly invariant to factual
  future-action zeroing or shuffling.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/tools/run_exact_v7_fullsequence_gate.py \
  --v8-code-root /path/to/wm3d \
  --runtime /path/to/runtime.yaml \
  --batch /path/to/sealed_current_data_batch.pt \
  --output /path/to/new_receipt.json \
  --device cuda:0
```

The output path must not already exist. Success exits `0`; any missing hook,
ambiguous mask, topology mismatch, leakage, or forward error writes a receipt
and exits `2`.

This gate has been calibrated with the same runtime and batch: the earlier
factorized factual candidate fails, while the exact full-sequence candidate
passes. It proves implementation topology only. It does not prove held-out
learning, final RGB quality, or closed-loop VLA success.
