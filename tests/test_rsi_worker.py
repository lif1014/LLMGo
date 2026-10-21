"""Execute arbitrary research jobs locally, including receipt identity and replay."""
from pathlib import Path
import sys

from llmgo.rsi.artifacts import source_snapshot,write_manifest
from llmgo.rsi.contracts import REPO,atomic_json,digest,file_hash,read_json
from llmgo.rsi.worker import execute


def test_free_form_worker_export_receipt_and_idempotent_replay(tmp_path):
    source=source_snapshot(REPO,tmp_path/'protected')
    candidate=tmp_path/'editable'
    candidate.mkdir()
    (candidate/'custom_trainer.py').write_text('''import json,os
from pathlib import Path
out=Path(os.environ['LLMGO_OUTPUT']);model=out/'model';model.mkdir()
(model/'config.json').write_text('{"model_type":"qwen3_5"}')
(model/'model.safetensors').write_bytes(b'fixture export, not real weights')
(out/'submission.json').write_text(json.dumps({'model_path':str(model),'method':'arbitrary trainer'}))
(out/'counter').write_text('once')
print('custom research task')
''')
    frozen=source_snapshot(candidate,tmp_path/'candidate')
    output=tmp_path/'output'
    site={'user_root':str(tmp_path),'base_model':str(tmp_path/'base'),'image':'fixture','gpu_count':1,'python':sys.executable}
    payload={'source':source,'candidate_source':frozen,'site':site,'output':str(output),
             'kind':'research','job_id':'j1','model_path':site['base_model'],
             'command':{'argv':['{python}','custom_trainer.py'],'resources':{'gpu_count':0}},
             'resources':{'gpu_count':0}}
    path=tmp_path/'manifest.json';wrapper=write_manifest(path,payload)
    result=execute(path,wrapper['sha256'])
    receipt=read_json(result['artifact'])
    assert receipt['model_identity']['sha256']
    assert receipt['method']=='arbitrary trainer'
    assert result['allocated_gpu_count']==0
    assert 'custom research task' in (output/'research.log').read_text()
    before=(output/'research.log').read_text()
    assert execute(path,wrapper['sha256'])==result
    assert (output/'research.log').read_text()==before
