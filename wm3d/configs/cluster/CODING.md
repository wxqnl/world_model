# 5B cluster configuration guide

- Files in this directory are operator templates, not model or data contracts.
- Keep secrets out of Git. Token files must live outside the checkout and use mode `0600`.
- Paths must be absolute after the template is copied to a site-local file.
- A site file may select an existing sealed data profile, but it must never invent adapter
  semantics or silently replace a source receipt.
- The same site file is shared by all nodes. Override `NODE_RANK` in the launcher environment.
