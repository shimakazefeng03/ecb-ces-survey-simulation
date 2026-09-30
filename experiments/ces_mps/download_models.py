"""Resolve revisions and acquire native Transformers weights for MPS."""
from pathlib import Path
import datetime as dt
import json
from huggingface_hub import HfApi, snapshot_download

HERE = Path(__file__).resolve().parent
MODELS = json.loads((HERE/'env-spec.json').read_text())['models']
manifest = HERE / 'model_manifest.json'
rows = json.loads(manifest.read_text()) if manifest.exists() else {}
for model in MODELS:
    if model not in rows:
        info = HfApi().model_info(model)
        rows[model] = {'revision':info.sha,'resolved_at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'downloading'}
        manifest.write_text(json.dumps(rows,indent=2))
    print('DOWNLOAD',model,rows[model]['revision'],flush=True)
    path = snapshot_download(model,revision=rows[model]['revision'],
                             allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','README.md','LICENSE'],max_workers=2)
    weights=list(Path(path).glob('*.safetensors'))
    assert weights and all(p.stat().st_size>0 for p in weights)
    rows[model].update(path=path,weight_bytes=sum(p.stat().st_size for p in weights),status='downloaded')
    manifest.write_text(json.dumps(rows,indent=2))
    print('READY',model,rows[model]['weight_bytes'],flush=True)
