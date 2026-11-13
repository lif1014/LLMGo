"""Current allocation authorization at both Meta and Student submission boundaries."""
import json
from pathlib import Path
import pytest
from llmgo.remote.resource_policy import validate_submission,SUBMISSION_POOLS
from llmgo.remote.cctl import CctlExecutor
from llmgo.rsi.contracts import REPO,read_json


@pytest.mark.parametrize('pool',SUBMISSION_POOLS)
def test_student_request_uses_the_authorized_pool_and_priority(monkeypatch,pool):
    site={'cluster':'beijing','resource_pool':pool,'priority':'PREEMPTABLE','user_root':'/user/lyh4170',
          'gpu_count':8,'gpu_model':'h800','cpu':64,'memory_gib':512,'python':'python','project':'rsi','billing_account_id':'fixture','image':'fixture'}
    payload={'site':site,'source':{'root':'/user/lyh4170/source'},'campaign_id':'test','job_id':'j000001','resources':{}}
    monkeypatch.setattr('llmgo.remote.cctl.load_manifest',lambda p:{'payload':payload,'sha256':'bound'})
    args,_=CctlExecutor(site,None).request('/user/lyh4170/manifest.json')
    assert args[args.index('--resource-pool')+1]==pool
    assert args[args.index('--priority')+1]=='PREEMPTABLE'


@pytest.mark.parametrize('pool,priority',[('model-autosft','HIGH'),('model-autosft','PREEMPTABLE'),('model-new','HIGH'),('model-new','NORMAL')])
def test_current_policy_rejects_all_other_submission_envelopes(pool,priority):
    with pytest.raises(ValueError,match='model-autosft is disabled'):validate_submission(pool,priority)


def test_new_meta_profile_has_double_context_and_separate_student_identity():
    site=read_json(REPO/'configs/rsi/site.qwen38-meta.json')
    assert site['meta_model']=='Qwen3.8-27B' and site['base_model'].endswith('/Qwen3.5-4B')
    assert (site['meta_context_length'],site['meta_max_output_tokens'],site['meta_compact_tokens'])==(131072,65536,57344)
    assert site['meta_compact_tokens']+site['meta_max_output_tokens']<site['meta_context_length']
    assert site['meta_thinking'] and site['meta_preserve_thinking'] and site['meta_reasoning_effort']=='xhigh'
    assert site['meta_sampling']['presence_penalty']==0
    validate_submission(site['resource_pool'],site['priority'])
