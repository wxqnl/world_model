# Data script guide

- Keep download, inspection, adapter audit, inventory, cache, and seal stages separate.
- Every input revision and payload is SHA-bound before it can enter a downstream stage.
- Raw schemas may be inspected automatically; physical semantics require an explicit adapter
  contract and operator confirmation.
- Writers are deterministic, no-clobber, and resumable by immutable task identity.
- Legacy inputs are converted into current WM3D manifests; they never bypass the current ABI.
