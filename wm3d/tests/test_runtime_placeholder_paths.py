from wm3d.training.runtime_contract import _PLACEHOLDER

def test_real_droid_repository_path_is_not_a_placeholder():
    assert not _PLACEHOLDER.search("/shared/Minko/datasets/hf_world_model_pretrain/lerobot__droid_1.0.1")

def test_unresolved_runtime_fields_are_still_rejected():
    for value in ("__EXPANDED_SELECTION__", "__COUNT_SELECTED_WINDOWS__", "PENDING", "MATERIALIZE_REQUIRED"):
        assert _PLACEHOLDER.search(value)
