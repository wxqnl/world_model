"""Reuse an existing whole-episode selection without touching data or validation."""
from collections import Counter
import json
from pathlib import Path


def select_training_entries(entries, identities, selection_path, metadata_root):
    document = json.loads(Path(selection_path).read_text())
    if document.get("schema") != "v7_training_episode_selection_v1":
        raise ValueError("unsupported whole-episode selection")
    if Path(document["data_root"]).resolve() != Path(metadata_root).resolve().parent:
        raise ValueError("episode selection belongs to another data root")
    requested = document["episodes"]
    sources = {entry.source for entry in entries}
    if set(requested) != sources or set(document["expected_windows"]) != sources:
        raise ValueError("selection must preserve all training sources")
    allowed = {}
    for name, ids in requested.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{name}: empty or duplicate selected episodes")
        allowed[name] = set(ids)
    selected, seen = [], {name: set() for name in sources}
    original_counts = Counter(entry.source for entry in entries)
    for entry in entries:
        if entry.split != "train":
            raise ValueError("training selection must never filter validation")
        source, episode = identities[entry.feature_shard]
        if source != entry.source:
            raise ValueError("episode/window source identity differs")
        if episode in allowed[source]:
            selected.append(entry)
            seen[source].add(episode)
    counts = Counter(entry.source for entry in selected)
    for name in sources:
        if seen[name] != allowed[name] or counts[name] != document["expected_windows"][name]:
            raise ValueError(f"{name}: selected episode/window coverage changed")
        if name.startswith("oxe_") and counts[name] != original_counts[name]:
            raise ValueError(f"{name}: the saved selection must retain all OXE windows")
    # Exactly the existing project's rule: count selected source windows,
    # without equalizing sources or selecting for motion/model quality.
    # ExactSourceSchedule consumes integer weights, not normalized fractions.
    # Counts preserve the same ratios without rounding any source to zero.
    return tuple(selected), dict(counts)
