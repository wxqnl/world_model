from wm3d_v3.eval.worldvla_libero_protocol import _extract_traj_spec, _parse_suites


def test_extract_worldvla_trajectory_spec_for_libero_10():
    spec = _extract_traj_spec(
        "10",
        3,
        "../processed_data/libero_10_image_state_action_t_512/"
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/trj_5",
    )
    assert spec.suite == "10"
    assert spec.episode_index == 3
    assert spec.dataset_dir == "libero_10_image_state_action_t_512"
    assert spec.task_name == "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
    assert spec.trj_name == "trj_5"
    assert spec.trj_index == 5


def test_parse_suites_normalizes_long_alias():
    assert _parse_suites("long,goal,object,spatial") == ["10", "goal", "object", "spatial"]
