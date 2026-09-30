import json

import pytest
import yaml

from wm3d.training.runtime_contract import canonical_sha256, load_yaml


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_runtime_numeric_types_survive_document_roundtrip(tmp_path, suffix):
    value = {"optimizer": {"eps": 1e-8, "min_lr": 1e-5},
             "stat": {"mtime_ns": 1790792449020385797}}
    path = tmp_path / ("runtime" + suffix)
    path.write_text(json.dumps(value) if suffix == ".json" else yaml.safe_dump(value))
    result = load_yaml(path)
    assert result == value
    assert canonical_sha256(result) == canonical_sha256(value)


def test_json_extension_does_not_silently_accept_yaml(tmp_path):
    path = tmp_path / "runtime.json"; path.write_text("eps: 1.0e-08\n")
    with pytest.raises(json.JSONDecodeError): load_yaml(path)
