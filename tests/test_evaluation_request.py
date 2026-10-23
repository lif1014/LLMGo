import json

import pytest

from llmgo.rsi.evaluation_request import request_evaluation, main


def test_request_preserves_training_plan_and_model(tmp_path):
    original={'hypothesis':'RL experiment','jobs':[{'argv':['python','train.py']}],
              'model_path':'/user/lyh4170/export','evaluation_batch':7,'evidence_refs':['log.json']}
    path=tmp_path/'proposal.json';path.write_text(json.dumps(original))
    result=request_evaluation(tmp_path,level=4)
    actual=json.loads(path.read_text())
    assert result['proposal_path']==str(path) and result['status']=='ready'
    assert actual=={**original,'evaluation':{'levels':[4],'tracks':['player']},'evaluation_gpus':8}


def test_cli_can_request_full_panel_or_model_control(tmp_path,monkeypatch,capsys):
    monkeypatch.chdir(tmp_path)
    assert main(['--level','L2','--track','model'])==0
    assert json.loads(capsys.readouterr().out)['status']=='ready'
    path=tmp_path/'proposal.json'
    assert json.loads(path.read_text())['evaluation']=={'levels':[2],'tracks':['model']}
    main(['--full-panel','--track','both','--batch-index','2'])
    actual=json.loads(path.read_text())
    assert actual['evaluation']=={'levels':list(range(1,9)),'tracks':['player','model']}
    assert actual['evaluation_batch']==2


@pytest.mark.parametrize('arguments',[['--level','L0'],['--level','L9'],
    ['--level','L2','--full-panel'],['--level','1','--batch-index','-1'],
    ['--level','1','--proposal','../escape.json']])
def test_invalid_request_does_not_change_proposal(tmp_path,monkeypatch,arguments):
    monkeypatch.chdir(tmp_path)
    path=tmp_path/'proposal.json';path.write_text('{"hypothesis":"keep"}')
    before=path.read_bytes()
    with pytest.raises(SystemExit):main(arguments)
    assert path.read_bytes()==before
