"""AgiBot2026 recorded-channel layout; no guessed unit/frame conversion.

All native action columns are reversible. Kinematic conditioning selects the
measured arm joints/grippers and head/waist coordinates explicitly; wrench,
velocity, redundant end poses, chassis pose and camera extrinsics remain in
the original record, not padded into the fixed state slots or called controls.
"""
from __future__ import annotations
from typing import Any, Mapping


def agibot_native_contract(info: Mapping[str, Any]):
    features = info["features"]
    action = features["action"]
    state = features["observation.state"]
    width, state_width = int(action["shape"][0]), int(state["shape"][0])
    variants = {(40,162): 240, (40,169): 241, (44,169): 242, (44,193): 243}
    if (width,state_width) not in variants:
        raise ValueError("unreviewed AgiBot field layout")
    af, sf = action["field_descriptions"], state["field_descriptions"]
    def indices(fields, key, count):
        row = fields[key]; out = list(row["indices"])
        if len(out) != count or int(row["dimensions"]) != count:
            raise ValueError(f"field dimensions changed: {key}")
        return out
    agl=indices(af,"action/left_effector/position",1)
    agr=indices(af,"action/right_effector/position",1)
    aj=indices(af,"action/joint/position",14)
    ap=indices(af,"action/end/position",6)
    aq=indices(af,"action/end/orientation",8)
    ah=indices(af,"action/head/position",3)
    aw=indices(af,"action/waist/position",5)
    av=indices(af,"action/robot/velocity",width-38)
    if agl+agr+ap+aq+aj+ah+aw+av != list(range(width)):
        raise ValueError("AgiBot action column identity/order changed")
    sj=indices(sf,"state/joint/position",14)
    sgl=indices(sf,"state/left_effector/position",1)
    sgr=indices(sf,"state/right_effector/position",1)
    sh=indices(sf,"state/head/position",3)
    sw=indices(sf,"state/waist/position",5)
    constants = {"40":0.0,"41":0.0,"42":0.0} if width==44 else {}
    action_columns = [aj[:7]+ap[:3]+aq[:4], aj[7:]+ap[3:]+aq[4:],
                      agl+agr+ah+aw+([38,39,43] if width==44 else av)]
    state_columns = [sj[:7]+sgl, sj[7:]+sgr, sh+sw]
    name=f"agibot2026_native_a{width}_s{state_width}"
    groups=[]; specs=[]
    for i,(group,ac,sc) in enumerate(zip(
        ("left_arm","right_arm","grippers_head_waist_base"),action_columns,state_columns
    )):
        term=lambda key,cols: dict(key=key,columns=cols,scale=[1]*len(cols),offset=[0]*len(cols))
        groups.append(dict(group=group,supervision="fine_command",
            action=[term("action",ac)],state=[term("observation.state",sc)],
            action_time_key="timestamp",state_time_key="timestamp",
            world_interval_index_key=None,action_transform="identity",
            state_transform="identity",action_value_mask=[True]*len(ac),
            state_value_mask=[True]*len(sc),action_alignment="as_recorded"))
        specs.append(dict(name=group,group_id=30+i,
            action_semantics=["controller_command"]*len(ac),
            state_semantics=["controller_state"]*len(sc),
            action_frame="G2_published_action_channels_native_coordinates_as_recorded",
            state_frame="G2_published_measured_joint_gripper_head_waist_native_coordinates",
            composition_operators=["last"]*len(ac)))
    views = [("head","observation.images.top_head"),
             ("left_wrist","observation.images.hand_left"),
             ("right_wrist","observation.images.hand_right")]
    if any(key not in features or features[key].get("dtype") != "video" for _,key in views):
        raise ValueError("AgiBot selected RGB streams changed")
    adapter=dict(schema="wm3d_source_adapter_v7",name=name,raw_format="lerobot_parquet_video",
        observation_time_key="timestamp",
        views=[dict(name=n,key=k,color_order="rgb") for n,k in views],
        groups=groups,raw_action_layout=dict(key="action",width=width,constant_columns=constants))
    template=dict(schema="wm3d_admitted_source_template_v1",
        sources=[dict(name=name,embodiment=name)],
        embodiments=[dict(name=name,embodiment_id=variants[(width,state_width)],groups=specs)])
    review=dict(native_action_columns=width,
        packed_action_columns=sum(map(len,action_columns)),constant_columns=constants,
        selected_state_fields=["state/joint/position","state/left_effector/position",
            "state/right_effector/position","state/head/position","state/waist/position"],
        state_fields_not_conditioned=sorted(set(sf)-{"state/joint/position",
            "state/left_effector/position","state/right_effector/position",
            "state/head/position","state/waist/position"}),
        no_control_truncation=True,no_gripper_clipping=True,no_unit_conversion=True,
        calibrated_metric_units_claim=False,quaternion_convention_claim=False,
        recorded_command_timing="Published action and state fields share the actual per-row timestamp; no row shift or synthetic clock.",
        hardware_execution_or_closed_loop_certified=False)
    return adapter,template,review
