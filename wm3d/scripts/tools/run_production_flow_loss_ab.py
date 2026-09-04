#!/usr/bin/env python3
"""Controlled full-size, real-batch loss-unit A/B; not a generalization gate."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
from scripts.tools import audit_action_owned_transport_checkpoint as audit
from wm3d.training.native_objective import objective_config_from_mapping, compute_native_objective, build_rgb_perceptual_model
from wm3d.training.pretrain import _forward_with_action_counterfactual, _learning_rate
from wm3d.training.runtime_contract import load_materialized_runtime

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--batches",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pixel-units",action="store_true")
    p.add_argument("--real-negative-every-step",action="store_true")
    p.add_argument("--occlusion-completion", action="store_true")
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--seed",type=int,default=7340)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    config,_=load_materialized_runtime(args.runtime)
    device=torch.device("cuda:0")
    audit._configure_reproducibility(args.seed)
    profile=dict(config["model_profile"]); body=dict(profile["model"])
    runtime=config["runtime_profile"]
    body["activation_checkpointing"]=runtime["train"].get("activation_checkpointing",body["activation_checkpointing"])
    body["rgb_decode_chunk_size"]=runtime["train"].get("rgb_decode_chunk_size",body["rgb_decode_chunk_size"])
    profile["model"]=body
    if args.occlusion_completion:
        body["rgb_transport_occlusion_completion"] = True
        final_channels = max(32, int(body["rgb_hidden"]) // 8)
        added_parameters = (final_channels * 9 + 1) + ((final_channels + 3) * 32 * 9 + 32) + 32 * 3
        profile["expected_parameter_count"] = int(profile["expected_parameter_count"]) + added_parameters
    with torch.device(device):
        model=audit.build_world_model(profile)
    count=sum(p.numel() for p in model.parameters())
    if count<1_000_000_000 or model.cfg.rgb_size!=256 or model.cfg.K!=8:
        raise RuntimeError("A/B requires the full production 1B/K8/RGB256 model")
    objective=replace(objective_config_from_mapping(config["objective_profile"]["objective"]),
                      rgb_flow_teacher_pixel_units=args.pixel_units)
    if args.real_negative_every_step:
        objective=replace(objective,action_counterfactual_token_advantage=0.0,
            context_pixel_action_rank_every=1,context_pixel_action_rank_ramp_steps=0)
    objective.validate()
    if args.occlusion_completion:
        objective = replace(objective, rgb_disocclusion_bce=0.03,
                            rgb_disocclusion_dice=0.03)
        objective.validate()
    perceptual=build_rgb_perceptual_model(objective,device=device)
    oc=runtime["optimizer"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(oc["peak_lr"]),
        betas=tuple(oc["betas"]),eps=float(oc["eps"]),weight_decay=float(oc["weight_decay"]),foreach=False)
    grouped={}
    for path in sorted(args.batches.glob("source*_normal.pt")):
        value=torch.load(path,map_location="cpu",weights_only=False)
        source=int(value["source_id"].reshape(-1)[0])
        grouped.setdefault(source,[]).append(value)
    if len(grouped)<3: raise RuntimeError("Need at least three real sources")
    batches=[]
    for source,values in sorted(grouped.items()):
        if len(values)<2: raise RuntimeError("Need a real action mismatch pair per source")
        batch={}
        for key,value in values[0].items():
            if torch.is_tensor(value): batch[key]=torch.cat([v[key] for v in values],dim=0)
            else:
                if any(v[key]!=value for v in values): raise RuntimeError("Non-tensor batch metadata differs")
                batch[key]=value
        batches.append(audit._batch_to_device(batch,device))
    reports=[]; training=[]
    def visibility_metrics(output, batch):
        target = batch["target_rgb"].float()
        valid = batch["target_rgb_mask"].bool() & batch["context_rgb_mask"].bool()[:, None, :, None, None, None]
        occ = batch["rgb_disocclusion_target"].float()
        occ = torch.nn.functional.interpolate(
            occ.flatten(0, 2), size=target.shape[-2:], mode="nearest"
        ).reshape(*target.shape[:3], 1, *target.shape[-2:]) >= 0.5
        error = (output["rgb"].float() - target).abs()
        values = {}
        for name, mask in (("visible", valid & ~occ), ("disoccluded", valid & occ)):
            weights = torch.broadcast_to(mask, error.shape)
            values[name + "_l1"] = float((error * weights).sum() / weights.sum().clamp_min(1))
        values["disocclusion_target_fraction"] = float(audit._masked_mean(occ.float(), valid))
        values["disocclusion_prediction_fraction"] = float(audit._masked_mean(
            torch.sigmoid(output["rgb_disocclusion_logit"].float()), valid))
        return values
    def evaluate(step):
        model.eval(); records=[]
        for batch in batches:
            variants,_,valid,distance=audit.build_action_variants(batch,step=0,minimum_distance=0.05)
            if not bool(valid.all()): raise RuntimeError("Real mismatch pair is invalid")
            outputs={label:audit._forward_eval(model,value) for label,value in variants.items()}
            invariants={label:audit._policy_invariants(outputs["normal"],outputs[label])
                        for label in ("physical_noop","distant_mismatch")}
            if not all(v for d in invariants.values() for v in d.values()):
                raise RuntimeError("Future action leaked into policy/action-free")
            records.append({"source_id":int(batch["source_id"][0]),
                "sample_indices":batch["sample_index"].tolist(),
                "visibility":{label:visibility_metrics(out,batch) for label,out in outputs.items()},
                "variants":{label:audit.variant_metrics(out,batch,motion_threshold=objective.rgb_motion_threshold)
                            for label,out in outputs.items()},
                "responses":{label:audit._response_rms(outputs["normal"],outputs[label],batch)
                            for label in ("physical_noop","distant_mismatch")},
                "invariants":invariants})
        report={"step":step,"sources":records}
        reports.append(report)
        print(json.dumps({"event":"evaluation",**report}),flush=True)
        model.train()
    evaluate(0)
    start=time.monotonic()
    for step in range(args.steps):
        batch=batches[step%len(batches)]
        optimizer.zero_grad(set_to_none=True)
        lr=_learning_rate(step,runtime)
        for group in optimizer.param_groups: group["lr"]=lr
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=_forward_with_action_counterfactual(model,batch,appearance_teacher_ratio=0.0,
                objective=objective,step=step)
            loss=compute_native_objective(output=output,batch=batch,config=objective,
                perceptual_model=perceptual,rgb_perceptual_chunk_size=runtime["train"].get("rgb_perceptual_chunk_size",4))
        if not all(bool(torch.isfinite(x).all()) for x in loss.values() if torch.is_tensor(x)):
            raise RuntimeError("Nonfinite production objective")
        loss["total"].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(runtime["train"]["gradient_clip"]))
        if not bool(torch.isfinite(norm)): raise RuntimeError("Nonfinite gradient")
        if step==0:
            owners={}
            for prefix in ("factual_action","factual_token_output","rgb_head.image_decoder.flow_head","rgb_head.image_decoder.occlusion_head","rgb_head.image_decoder.occlusion_completion","action_head"):
                parameters=[p for name,p in model.named_parameters() if name.replace("_checkpoint_wrapped_module.","").startswith(prefix) and p.grad is not None]
                owners[prefix]={"tensors":len(parameters),"norm":float(torch.stack([p.grad.float().square().sum() for p in parameters]).sum().sqrt()) if parameters else None}
            print(json.dumps({"event":"gradient_owners","owners":owners}),flush=True)
        optimizer.step()
        if (step+1)%8==0:
            row={"step":step+1,"source":int(batch["source_id"][0]),"lr":lr,"grad_norm":float(norm),
                **{key:float(loss[key].detach()) for key in ("total","rgb_l1","rgb_motion_region_l1","rgb_flow_teacher","rgb_flow_epe","rgb_flow_prediction_magnitude","rgb_flow_target_magnitude","action_fine") if key in loss}}
            training.append(row);print(json.dumps({"event":"train",**row}),flush=True)
        del output,loss
        if (step+1)%32==0 or step+1==args.steps: evaluate(step+1)
    payload={"scope":"controlled optimization diagnostic, not held-out generalization or formal qualification",
        "fresh_initialization":True,"checkpoint_loaded":False,"parameter_count":count,
        "pixel_units":args.pixel_units,"seed":args.seed,"steps":args.steps,
        "real_negative_every_step":args.real_negative_every_step,
        "occlusion_completion":args.occlusion_completion,
        "batch_size":int(batches[0]["source_id"].shape[0]),"source_balanced_diagnostic":True,
        "production_lr_schedule":True,"elapsed_seconds":time.monotonic()-start,
        "evaluations":reports,"training":training}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f: json.dump(payload,f,indent=2)
if __name__=="__main__": main()
