import json
from pathlib import Path
import pytest
from llmgo.rsi.contracts import atomic_json,file_hash
from llmgo.rsi.research_evidence import parse_game,summarize_metrics,publish_evidence


def game(tmp_path,name,plies,result):
    path=tmp_path/name/'events.jsonl';path.parent.mkdir(parents=True)
    events=[{'type':'episode_started','episode_id':name,'student_color':'white','ply':0}]
    for ply in range(1,plies+1):
        events.extend([{'type':'model_response','ply':ply-1,'response':'X'*200},
            {'type':'student_move' if ply%2 else 'opponent_move','ply':ply,'uci':'a2a3','fen':'fixture'}])
    events.append({'type':'episode_finished','ply':plies,'result':result,'reason':'checkmate'})
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    return {'raw_events':str(path),'event_hashes':{'raw_events':file_hash(path)},'diagnostics':{'result':result,'plies':plies},'result_reason':'checkmate'}


def fixture(tmp_path):
    games=[game(tmp_path,'short',6,-1),game(tmp_path,'long',70,-1),game(tmp_path,'draw',100,0)]
    path=tmp_path/'metrics.json';atomic_json(path,{'protocol':{'games':3,'max_plies':300},'statistics':{'wins':0,'draws':1,'losses':2,'games':3,'score':1/6},'games':games})
    return path


def test_full_lengths_and_official_scores_not_display_slice_or_jsonl_lines(tmp_path):
    path=fixture(tmp_path);report=summarize_metrics(path,file_hash(path),tmp_path/'derived')
    assert report['official_statistics']['wins']==0 and report['score']==1/6
    assert report['plies']=={'min':6,'median':70,'max':100}
    assert report['loss_length_bins_plies']=={'at_most_8':1,'9_to_20':0,'over_20':1}
    timeline=json.loads(Path(report['games'][1]['timeline']).read_text())
    assert len(timeline['moves'])==70 and len(timeline['decision_events'])==70
    event=timeline['decision_events'][30]['raw_reference']
    with open(event['path'],'rb') as f:f.seek(event['offset']);original=json.loads(f.read(event['length']))
    assert original['response']=='X'*200
    assert json.loads(path.read_text())['protocol']['max_plies']==300


@pytest.mark.parametrize('corruption',['hash','score','wins','move','result'])
def test_bad_evidence_is_not_silently_reported_as_fact(tmp_path,corruption):
    path=fixture(tmp_path);d=json.loads(path.read_text())
    if corruption=='score':d['statistics']['score']=.9
    elif corruption=='wins':d['statistics']['wins']=8
    elif corruption=='result':d['games'][0]['diagnostics']['result']=1
    elif corruption=='move':
        p=Path(d['games'][0]['raw_events']);lines=p.read_text().splitlines();e=json.loads(lines[2]);e['ply']=4;lines[2]=json.dumps(e);p.write_text('\n'.join(lines)+'\n');d['games'][0]['event_hashes']['raw_events']=file_hash(p)
    atomic_json(path,d)
    with pytest.raises(ValueError):summarize_metrics(path,'bad' if corruption=='hash' else file_hash(path),tmp_path/'derived')


def test_controller_index_pins_own_paths_and_deduplicates_refs(tmp_path):
    path=fixture(tmp_path);ref={'path':str(path),'sha256':file_hash(path)}
    state={'site':{'user_root':str(tmp_path),'control_python':'/fixed/python'},'baseline':[{'ref':ref}],
        'working':{'evaluations':[{'ref':ref}]}}
    x=publish_evidence(tmp_path/'campaign',state)
    assert x['analysis_python']=='/fixed/python' and len(x['reports'])==1
    assert Path(x['index']).is_file()


def test_official_referee_rejects_four_ply_shortening():
    from llmgo.rsi.contracts import protocol,validate_protocol
    from llmgo.rsi.ladder import opponent
    original=protocol(opponent(2),'fixture',games=40)
    validate_protocol(original)
    altered={**original,'max_plies':4}
    with pytest.raises(ValueError,match='max_plies'):validate_protocol(altered)
    assert original['max_plies']==300
