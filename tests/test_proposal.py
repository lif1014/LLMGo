import json
from pathlib import Path
import pytest
from llmgo.rsi.proposal import validate_proposal
from llmgo.rsi.meta_protocol import ResearchHandshake


def valid():return {'hypothesis':'Agent-selected L2 hypothesis','jobs':[{'argv':['{python}','scripts/run_grpo.py','--output','{output}','--model','{model}']}],'evaluation':{'levels':[2,3],'tracks':['player']}}


def test_reject_observed_object_hypothesis_and_missing_executable_jobs():
    p={'hypothesis':{'description':'L2 hypothesis'},'training_config':{'script':'scripts/run_grpo.py'}}
    with pytest.raises(ValueError,match='hypothesis must be.*string'):validate_proposal(p)
    p['hypothesis']='L2 hypothesis'
    with pytest.raises(ValueError,match=r'jobs\[\].argv'):validate_proposal(p)
    assert validate_proposal(valid(),require_training=True)['jobs']
    with pytest.raises(ValueError,match='Training requires'):validate_proposal({'hypothesis':'evaluation only'},require_training=True)


def test_failed_handoff_stays_editable_and_corrected_proposal_commits(tmp_path):
    workspace=tmp_path/'workspace';workspace.mkdir();p=workspace/'proposal.json'
    h=ResearchHandshake(tmp_path/'campaign',workspace,'a'*64);h.publish()
    answer={'status':'ready','summary':'L2 training','proposal_path':str(p)}
    p.write_text(json.dumps({'hypothesis':{'description':'bad'},'training_config':{}}))
    with pytest.raises(ValueError,match='hypothesis'):h.commit(answer)
    assert not h.result_path.exists()
    p.write_text(json.dumps(valid()));h.commit(answer)
    assert h.read()==answer
    p.write_text(json.dumps({**valid(),'hypothesis':'edited'}))
    with pytest.raises(ValueError,match='changed after completion'):h.read()


def test_evaluation_only_and_checkpoint_review_remain_valid(tmp_path):
    validate_proposal({'hypothesis':'diagnostic','jobs':[],'evaluation':{'levels':[2.5],'tracks':['player']}})
    workspace=tmp_path/'workspace';workspace.mkdir();p=workspace/'proposal.json'
    p.write_text(json.dumps({'training_review':{'action':'continue','request_id':'bound-review'}}))
    h=ResearchHandshake(tmp_path/'campaign',workspace,'b'*64);h.publish()
    h.commit({'status':'ready','summary':'review','proposal_path':str(p)})
    assert h.read()['status']=='ready'


@pytest.mark.parametrize('edit',[{'jobs':[{'argv':'python train.py'}]},{'jobs':[{'argv':['python'],'resources':{'gpu_count':9}}]}, {'evaluation_gpus':0}])
def test_invalid_argv_and_resource_requests_are_rejected(edit):
    with pytest.raises(ValueError):validate_proposal(valid()|edit)
