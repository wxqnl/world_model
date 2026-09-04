#!/usr/bin/env python3
"""Export actual K8 diagnostic predictions, with source timestamps and no teacher input."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageDraw
from scripts.tools import audit_action_owned_transport_checkpoint as audit
from scripts.tools.run_action_owned_transport_gate import _to_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--batches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit._configure_reproducibility(7340)
    batches = []
    grouped = {}
    for path in sorted(args.batches.glob("source*_normal.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=False)
        grouped.setdefault(int(batch["source_id"][0]), []).append(batch)
    for source, values in sorted(grouped.items()):
        batch = {}
        for key, value in values[0].items():
            if torch.is_tensor(value):
                batch[key] = torch.cat([item[key] for item in values])
            else:
                if any(item[key] != value for item in values):
                    raise RuntimeError("Non-tensor batch metadata differs")
                batch[key] = value
        batches.append(batch)
    predictions = {}
    for label, path in (("direct", args.direct), ("transport", args.transport)):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("diagnostic_only") is not True:
            raise ValueError("Expected a diagnostic model-only snapshot")
        with torch.device("meta"):
            model = audit.build_world_model(state["model_profile"])
        model.load_state_dict(state["model"], strict=True, assign=True)
        del state
        model = model.cuda().eval()
        for batch in batches:
            source = int(batch["source_id"][0])
            gpu_batch = audit._batch_to_device(batch, torch.device("cuda:0"))
            variants, _, valid, _ = audit.build_action_variants(gpu_batch, step=0, minimum_distance=0.05)
            if not bool(valid.all()):
                raise ValueError("Invalid real-action mismatch pair")
            base = None
            for variant, value in variants.items():
                output = audit._forward_eval(model, value)
                if base is None:
                    base = output
                elif not all(audit._policy_invariants(base, output).values()):
                    raise RuntimeError("Future action leaked into policy/action-free")
                predictions[(label, source, variant)] = output["rgb"].detach().float().cpu()
            del base, output, gpu_batch, variants
        del model
        torch.cuda.empty_cache()
    manifest = {"scope": "fixed-batch fitting diagnostic, not held-out evaluation", "clips": []}
    for batch in batches:
        source = int(batch["source_id"][0])
        for sample in range(len(batch["source_id"])):
            times = batch["future_world_boundaries_dt"][sample, 1:].tolist()
            if len(times) != 8 or any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError("Expected strictly ordered real K8 future timestamps")
            intervals = [round((b - a) * 1000) for a, b in zip(times, times[1:])]
            durations = intervals + [intervals[-1]]
            for view in range(batch["context_rgb"].shape[1]):
                if not bool(batch["context_rgb_mask"][sample, view]):
                    continue
                if not bool(batch["target_rgb_mask"][sample, :, view].all()):
                    continue
                prefix = f"source{source}_sample{int(batch['sample_index'][sample])}_view{view}"
                tensors = {
                    "target": batch["target_rgb"][sample, :, view],
                    "direct": predictions[("direct", source, "normal")][sample, :, view],
                    "transport": predictions[("transport", source, "normal")][sample, :, view],
                    "copy-last": batch["context_rgb"][sample, view].unsqueeze(0).expand(8, -1, -1, -1),
                    "direct-noop": predictions[("direct", source, "physical_noop")][sample, :, view],
                    "direct-wrong": predictions[("direct", source, "distant_mismatch")][sample, :, view],
                }
                frames = []
                contact = Image.new("RGB", (4 * 256, 8 * 280), "white")
                for horizon in range(8):
                    panel = Image.new("RGB", (6 * 256, 280), "white")
                    draw = ImageDraw.Draw(panel)
                    for column, (name, tensor) in enumerate(tensors.items()):
                        panel.paste(_to_image(tensor[horizon]), (column * 256, 24))
                        draw.text((column * 256 + 4, 4), f"{name} h{horizon+1} t={times[horizon]:.2f}s", fill="black")
                    frames.append(panel)
                    contact.paste(panel.crop((0, 0, 4 * 256, 280)), (0, horizon * 280))
                gif = args.output / (prefix + ".gif")
                frames[0].save(gif, save_all=True, append_images=frames[1:], duration=durations,
                               loop=0, optimize=False, disposal=2)
                contact.save(args.output / (prefix + ".png"))
                manifest["clips"].append({"name": prefix, "times_seconds": times,
                                          "durations_ms": durations, "gif": str(gif)})
    (args.output / "timeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "export_complete", "clips": len(manifest["clips"]),
                      "output": str(args.output)}), flush=True)

if __name__ == "__main__":
    main()
