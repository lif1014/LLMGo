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


def test_comparison_uses_cases_not_repetitions_and_rejects_invalid_allocations():
    import pytest
    from llmgo.rsi.meta_benchmark import compare
    def result(model,passed):
        return {'model':model,'summary':{},'admission':{'platform_status':'Succeeded'},
                'results':[{'id':str(i),'passed':passed} for i in range(8) for _ in range(3)]}
    a,b=result('small',False),result('large',True)
    r=compare(a,b)
    assert r['paired_cases']==8 and r['right_only_success']==8
    assert r['right_improved_at_p05']
    b['admission']['platform_status']='Killed'
    with pytest.raises(ValueError,match='Succeeded'):compare(a,b)


def test_transport_failure_aborts_acceptance_instead_of_scoring_sixty_failures(tmp_path):
    import pytest
    from llmgo.rsi.meta_benchmark import run_benchmark
    from llmgo.rsi.meta_protocol import MetaFailure
    home=tmp_path/'codex-home';home.mkdir();(home/'config.toml').write_text('model="fixture"')
    calls=[]
    class Backend:
        def __init__(self,*args):self.session={}
        def turn(self,*args):calls.append(1);raise MetaFailure('HTTP request rejected',code='turn_failed')
    with pytest.raises(RuntimeError,match='not capability scores'):
        run_benchmark({'meta_model':'fixture','meta_benchmark_repetitions':3},None,tmp_path,Backend)
    assert len(calls)==1
    progress=json.loads((tmp_path/'benchmark-progress.json').read_text())
    assert progress['stage']=='failed' and progress['infrastructure_error'] is True
    assert not (tmp_path/'benchmark-result.json').exists()
