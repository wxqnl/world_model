"""Window-only metadata for stat-bound direct raw training; no feature cache."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping
from .file_binding import require_stat_binding, verify_stat_asset
from .manifest_contract import ADMITTED_CACHE_INDEX_SCHEMA


@dataclass(frozen=True)
class AdmittedWindowEntry:
    sample_id: str
    source: str
    split: str
    embodiment: str
    task_id: str
    leading_feature_row: int
    context_feature_rows: tuple[int,...]
    future_feature_rows: tuple[int,...]

    @property
    def feature_shard(self) -> str:
        """Compatibility key for direct-raw lookups, never a feature-file path."""
        return self.task_id


def window_from_mapping(row: Mapping[str,Any]) -> AdmittedWindowEntry:
    fields={"schema","sample_id","source","split","embodiment","task_id",
        "leading_feature_row","context_feature_rows","future_feature_rows"}
    if not isinstance(row,dict) or set(row)!=fields or row["schema"]!=ADMITTED_CACHE_INDEX_SCHEMA:
        raise ValueError("admitted window fields/schema mismatch")
    for key in ("sample_id","source","embodiment"):
        if not isinstance(row[key],str) or not row[key].strip():
            raise ValueError("empty admitted window identity")
    if row["split"] not in {"train","val","test"} or not re.fullmatch(r"task[0-9]{9}",row["task_id"]):
        raise ValueError("invalid admitted window split/task")
    left,right=row["context_feature_rows"],row["future_feature_rows"]
    if not isinstance(left,list) or not isinstance(right,list) or not left or not right:
        raise ValueError("admitted context/future rows must be nonempty")
    all_rows=[row["leading_feature_row"],*left,*right]
    if (any(type(x)is not int or x<0 for x in all_rows)
            or any(a>=b for a,b in zip(all_rows,all_rows[1:]))):
        raise ValueError("admitted rows must be nonnegative/strictly increasing")
    return AdmittedWindowEntry(**{k:row[k] for k in fields-{"schema","context_feature_rows","future_feature_rows"}},
        context_feature_rows=tuple(left),future_feature_rows=tuple(right))


def parse_admitted_window_index(path:Path):
    entries=[]; identities=set()
    with path.open() as f:
        for number,line in enumerate(f,1):
            entry=window_from_mapping(json.loads(line))
            if entry.sample_id in identities:
                raise ValueError(f"duplicate admitted window at line {number}")
            identities.add(entry.sample_id);entries.append(entry)
    if not entries:raise ValueError("admitted window index is empty")
    return tuple(entries)


def load_admitted_window_index(path:Path,*,expected_binding:Mapping[str,Any]):
    from .parsed_index_cache import load_parsed_window_index
    path=Path(path).absolute()
    binding=require_stat_binding(expected_binding)
    check=lambda:verify_stat_asset(path.parent,path.name,binding)
    check()
    return load_parsed_window_index(path,None,source_binding=binding,
        verify_source=check,build=lambda:parse_admitted_window_index(path))
