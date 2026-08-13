# Environment guide

- `requirements.lock` is the Python dependency lock used by the bootstrap script.
- PyTorch/CUDA wheels are installed explicitly by `bootstrap_environment.sh` because the cluster
  driver determines the wheel index.
- Environment reuse is allowed only when the sealed receipt matches exactly.
- Never store tokens, private indexes, host credentials, or machine-specific paths here.
