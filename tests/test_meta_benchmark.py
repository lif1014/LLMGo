import json
from llmgo.rsi.contracts import atomic_json
from llmgo.rsi.meta_benchmark import cases,grade,summarize


def test_benchmark_uses_twenty_independent_synthetic_task_definitions(tmp_path):
    definitions=cases()
    assert len(definitions)==20 and len({c['id'] for c in definitions})==20
    c=definitions[0];atomic_json(tmp_path/'context.json',{})
    atomic_json(tmp_path/'proposal.json',{'hypothesis':'Read formal evidence','jobs':[],'evaluation':False})
    atomic_json(tmp_path/'observations.json',c['expected']);assert grade(c,tmp_path)
    atomic_json(tmp_path/'observations.json',{**c['expected'],'wins':8});assert not grade(c,tmp_path)


def test_review_grader_requires_requested_action_and_evaluated_step(tmp_path):
    c=next(c for c in cases() if c['kind']=='review')
    atomic_json(tmp_path/'context.json',{'training_review':c['review']})
    atomic_json(tmp_path/'proposal.json',{'training_review':{'request_id':c['review']['request_id'],'action':c['action'],'reason':'Evidence','selected_step':c['selected']}})
    assert grade(c,tmp_path)
    value=json.loads((tmp_path/'proposal.json').read_text());value['training_review']['action']='finish';atomic_json(tmp_path/'proposal.json',value)
    assert not grade(c,tmp_path)


def test_summary_separates_initial_schema_errors_from_final_factual_success():
    rows=[{'kind':'facts','passed':True,'seconds':1,'checks':[{'ok':False},{'ok':True}]},
          {'kind':'review','passed':False,'seconds':2,'checks':[{'ok':True}]}]
    value=summarize(rows)
    assert value['passed']==1 and value['first_check_schema_passes']==1
    assert value['schema_measured_cases']==2 and value['schema_check_failures']==1
