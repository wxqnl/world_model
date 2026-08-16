from __future__ import annotations

import pytest

from wm3d.data.formal_cache_adapter import FormalCacheError, _git_object_id


@pytest.mark.parametrize("length", (40, 64))
def test_git_object_id_accepts_sha1_and_sha256_repositories(length: int) -> None:
    value = "a" * length
    assert _git_object_id(value, field="commit") == value


@pytest.mark.parametrize("value", ("a" * 39, "A" * 40, "g" * 40, "a" * 65))
def test_git_object_id_rejects_noncanonical_values(value: str) -> None:
    with pytest.raises(FormalCacheError, match="lowercase Git object ID"):
        _git_object_id(value, field="commit")
