"""Phase 2 — generative head G_theta + physics inference g_phi.

STATUS: scaffolding only. No training runs in this snapshot.

We ship working *unit-testable* architectures and an empty training loop; when
a future session has (text/image, scene) pairs and the Phase 1 predictor
frozen checkpoint, everything below should drop in.

Modules::

    generative  — token-space flow-matching G_theta
    physics     — g_phi latent physics inference
    coupling    — consistency losses that tie G_theta and D_psi via g_phi

See ``PHASE2.md`` at the repo root for the design rationale and the exact
open questions that are still unresolved.
"""
