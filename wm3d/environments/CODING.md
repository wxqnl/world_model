# Environment guide

- `requirements.lock` is the Python dependency lock used by the bootstrap script.
- PyTorch/CUDA wheels are installed explicitly by `bootstrap_environment.sh` because the cluster
  driver determines the wheel index.
- Environment reuse is allowed only when the sealed receipt matches exactly.
- `bootstrap_foundation_environment.sh` applies the tracked `wsa_action_time_padding.patch`
  before importing WSA. It excludes masked command slots from the existing action MSE and
  its diagnostic reductions; it does not add a loss or change source weighting. Apply it
  before training, never to an active training checkout. Model loading rejects unpatched WSA.
- Never store tokens, private indexes, host credentials, or machine-specific paths here.
