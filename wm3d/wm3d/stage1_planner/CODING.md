# Stage1 planner guide

- Stage1 freezes a committed Stage0 DCP and performs one native rollout with `0 < H <= K`.
- The learned planner never receives candidate actions. Action cost stays outside learned logits.
- Candidate evidence must be regenerated from the same Stage0/encoder/data lineage and real
  simulator outcomes.
- Train, exact resume, and evaluation receipts bind branch seal, rollout audit, Stage0 DCP, and
  launch qualification.
- Evaluation covers the complete sealed split and records all four structural gates.
