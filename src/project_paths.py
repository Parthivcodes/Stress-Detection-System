"""Resolve archived paths without rewriting frozen experiment manifests."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def resolve_path(path):
    path=Path(path)
    if not path.is_absolute():path=ROOT/path
    try:rel=path.relative_to(ROOT).as_posix()
    except ValueError:return path
    cfg=json.loads((ROOT/'models/path_aliases.json').read_text())
    for old,new in sorted(cfg['prefixes'].items(),key=lambda kv:-len(kv[0])):
        if rel==old or rel.startswith(old+'/'):
            rel=new+rel[len(old):];break
    return ROOT.joinpath(*(cfg['components'].get(s,s) for s in rel.split('/')))
