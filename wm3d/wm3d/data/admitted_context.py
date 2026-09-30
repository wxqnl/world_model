"""Reuse immutable base text contexts and append only newly encoded instructions."""
from pathlib import Path
import json
import torch
from .file_binding import require_stat_binding,verify_stat_asset

SCHEMA="wm3d_admitted_foundation_context_union_v1"

class ContextUnion:
    def __init__(self,stores):
        self.stores=tuple(stores);self._owner={}
        for store in stores:
            for text in store.tasks:
                if text in self._owner:raise ValueError("text appears in more than one context part")
                self._owner[text]=store
        self.tasks=tuple(self._owner)
    def lookup(self,tasks):
        values=[]
        for raw in tasks:
            text=str(raw).strip()
            if text not in self._owner:raise ValueError("instruction absent from admitted context union")
            values.append(self._owner[text].lookup([text]))
        return torch.cat([v[0] for v in values]),torch.cat([v[1] for v in values])

def load_context_union(root,payload,store_type):
    if set(payload)!={"schema","admission_id","parts"} or not payload["parts"]:
        raise ValueError("context union fields differ")
    stores=[]
    for part in payload["parts"]:
        if set(part)!={"root","index_binding","tensor_binding"}:raise ValueError("invalid context part")
        directory=Path(part["root"])
        if not directory.is_absolute() or directory.resolve()==root.resolve():raise ValueError("invalid context root")
        for name,key in (("index.json","index_binding"),("context.safetensors","tensor_binding")):
            if require_stat_binding(part[key])["admission_id"]!=payload["admission_id"]:
                raise ValueError("context part admission differs")
            verify_stat_asset(directory,name,part[key])
        child=json.loads((directory/"index.json").read_text())
        if child.get("schema")!="wm3d_foundation_context_store_v1":raise ValueError("nested context union forbidden")
        store=store_type.from_directory(directory)
        for name,key in (("index.json","index_binding"),("context.safetensors","tensor_binding")):
            verify_stat_asset(directory,name,part[key])
        if stores and store.context.shape[1:]!=stores[0].context.shape[1:]:
            raise ValueError("text context parts use different dimensions")
        stores.append(store)
    return ContextUnion(stores)
