"""Explicit episode subset for a bounded fresh run; production tensors are unchanged."""
from collections.abc import Mapping

SCHEMA = "wm3d_small_episode_selection_v1"


def validate_selection(value):
    fields = {"schema", "source", "task_ids_by_split", "window_count_by_split",
              "train_episodes", "val_episodes", "task_texts"}
    if not isinstance(value, Mapping) or set(value) != fields or value["schema"] != SCHEMA:
        raise ValueError("invalid small-data selection contract")
    if not isinstance(value["source"], str) or not value["source"]:
        raise ValueError("small-data source missing")
    ids = value["task_ids_by_split"]
    counts = value["window_count_by_split"]
    if set(ids) != {"train", "val"} or set(counts) != {"train", "val"}:
        raise ValueError("small-data train/val splits required")
    episodes = {}
    for split in ("train", "val"):
        if not isinstance(ids[split], dict) or not ids[split]:
            raise ValueError("small-data tasks must be explicit")
        if any(not isinstance(k, str) or not isinstance(v, str) or not k or not v
               for k, v in ids[split].items()):
            raise ValueError("invalid task/episode identity")
        episodes[split] = set(ids[split].values())
        if len(episodes[split]) != len(ids[split]) or len(ids[split]) != value[split + "_episodes"]:
            raise ValueError("small-data episode identities/counts differ")
        if type(counts[split]) is not int or counts[split] <= 0:
            raise ValueError("small-data windows must be positive")
    if set(ids["train"]) & set(ids["val"]) or episodes["train"] & episodes["val"]:
        raise ValueError("small-data episode leakage")
    if len(value["task_texts"]) != 4 or len(set(value["task_texts"])) != 4:
        raise ValueError("four frozen tasks required")
    return value


def select_entries(dataset, selection, split):
    validate_selection(selection)
    if split not in ("train", "val"):
        raise ValueError("small-data loader split is not train/val")
    wanted = selection["task_ids_by_split"][split]
    selected = tuple(e for e in dataset.entries if e.task_id in wanted)
    if len(selected) != selection["window_count_by_split"][split]:
        raise ValueError("selected windows differ from sealed small-data count")
    seen = {e.task_id for e in selected}
    if seen != set(wanted):
        raise ValueError("selected task is absent from its real split")
    for entry in selected:
        if entry.split != split or entry.source != selection["source"]:
            raise ValueError("selected window split/source mismatch")
    for task_id in seen:
        task = dataset._source._load_task(task_id)
        if (task.split != split or task.source != selection["source"]
                or task.episode_id != wanted[task_id]
                or task.task_text not in selection["task_texts"]):
            raise ValueError("real episode identity differs from selected task")
    dataset.entries = selected
    dataset._source_names = (selection["source"],)
