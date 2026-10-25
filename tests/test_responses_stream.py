"""The observed converter omissions must not lose reasoning or tool calls."""
import json

import pytest

from llmgo.rsi.responses_stream import ReasoningStream, StreamProtocolError


def reasoning_start(item_id="rs1", index=0, **fields):
    return {"type":"response.output_item.added", "output_index":index,
            "item":{"type":"reasoning", "id":item_id, "status":"in_progress", **fields}}


def summary(kind="text.delta", item_id="rs1", index=0, part=0, **fields):
    return {"type":"response.reasoning_summary_"+kind,"item_id":item_id,
            "output_index":index,"summary_index":part,**fields}


def test_observed_missing_summary_and_part_start_preserve_all_text_and_tools():
    stream=ReasoningStream()
    original=[reasoning_start(), summary(delta="First "), summary(delta="second"),
        summary("text.done",text="First second"),
        summary("part.done")]
    original[-1]={**summary("part.done"),"part":{"type":"summary_text","text":"First second"}}
    original.extend([
        {"type":"response.output_item.done","output_index":0,
         "item":{"type":"reasoning","id":"rs1","summary":[{"type":"summary_text","text":"First second"}]}},
        {"type":"response.output_item.added","output_index":1,"item":{"id":"tool1","type":"function_call","call_id":"call1","name":"exec_command","arguments":""}},
        {"type":"response.function_call_arguments.delta","item_id":"tool1","output_index":1,"delta":'{"cmd":"pwd"}'},
        {"type":"response.output_item.done","output_index":1,"sequence_number":1,
         "item":{"id":"tool1","type":"function_call","call_id":"call1","name":"exec_command","arguments":'{"cmd":"pwd"}'}},
        {"type":"response.completed","response":{"status":"completed","usage":{"output_tokens":10}}}])
    before=json.loads(json.dumps(original))
    output=[out for event in original for out in stream.feed(event)]
    assert original==before  # raw evidence was not mutated in place
    assert output[0]['item']['summary']==[]
    assert output[1]['type']=='response.reasoning_summary_part.added'
    assert ''.join(e['delta'] for e in output if e['type']=='response.reasoning_summary_text.delta')=='First second'
    assert sum(e['type']=='response.reasoning_summary_part.added' for e in output)==1
    before_payloads=[{k:v for k,v in event.items() if k!='sequence_number'} for event in before]
    for event in output:
        if event['type'].startswith('response.function_call') or event.get('item',{}).get('type')=='function_call':
            assert {k:v for k,v in event.items() if k!='sequence_number'} in before_payloads
    assert output[-1]['response']==before[-1]['response']
    assert [e['sequence_number'] for e in output]==list(range(len(output)))
    assert stream.counts['reasoning_summary_defaulted']==stream.counts['summary_part_added']==1


def test_valid_lifecycle_is_not_duplicated_and_other_events_keep_payloads():
    original=[reasoning_start(summary=[]), summary('part.added',part=0), summary(delta='ok'),
              summary('text.done',text='ok'),summary('part.done')]
    original[1]['part']={'type':'summary_text','text':''}
    for n,event in enumerate(original):event['sequence_number']=n
    stream=ReasoningStream();out=[x for e in original for x in stream.feed(e)]
    assert out==original and not stream.counts


def test_multiple_reasoning_items_and_parts_are_scoped_and_request_state_resets():
    stream=ReasoningStream()
    assert stream.feed(reasoning_start())[0]['item']['summary']==[]
    for part in [0,1]:
        out=stream.feed(summary(part=part,delta='piece'))
        assert out[0]['type']=='response.reasoning_summary_part.added'
    stream.feed(reasoning_start('rs2',2,summary=[]))
    assert stream.feed(summary(item_id='rs2',index=2,delta='next'))[0]['item_id']=='rs2'
    other=ReasoningStream()
    with pytest.raises(StreamProtocolError,match='active item'):other.feed(summary(delta='orphan'))


@pytest.mark.parametrize('mode',['orphan','wrong_index','negative_part','closed','duplicate_item','duplicate_part'])
def test_ambiguous_or_closed_events_fail_explicitly_instead_of_hiding_deltas(mode):
    stream=ReasoningStream()
    if mode!='orphan':stream.feed(reasoning_start())
    if mode=='closed':stream.feed({'type':'response.output_item.done','output_index':0,'item':{'type':'reasoning','id':'rs1'}})
    if mode=='duplicate_item':event=reasoning_start()
    elif mode=='duplicate_part':
        event=summary('part.added');stream.feed(event)
    else:event=summary(delta='bad',index=1 if mode=='wrong_index' else 0,part=-1 if mode=='negative_part' else 0)
    with pytest.raises(StreamProtocolError):stream.feed(event)


def test_pinned_iterator_defers_only_empty_initial_chunks_and_does_not_mutate_class(monkeypatch):
    from types import SimpleNamespace
    from llmgo.rsi.responses_stream import prepare_chat_iterator
    calls=[]
    cls=type('LiteLLMCompletionStreamingIterator',(),{
        '__module__':'litellm.responses.litellm_completion_transformation.streaming_iterator',
        'sent_output_item_added_event':False,
        '_ensure_output_item_for_chunk':lambda self,chunk:calls.append(chunk)})
    monkeypatch.setattr('llmgo.rsi.responses_stream.version',lambda name:'1.105.2')
    patched,other=cls(),cls()
    assert prepare_chat_iterator(patched)
    def chunk(finish=None,**fields):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(**fields),finish_reason=finish)])
    for fields in [{},{'role':'assistant'},{'content':''},{'reasoning_content':''}]:
        patched._ensure_output_item_for_chunk(chunk(**fields))
    assert calls==[]
    for fields in [{'content':'hello'},{'reasoning_content':'summary'},{'tool_calls':[{'id':'call'}]}]:
        c=chunk(**fields);patched._ensure_output_item_for_chunk(c);assert calls[-1] is c
    end=chunk(finish='stop');patched._ensure_output_item_for_chunk(end);assert calls[-1] is end
    other._ensure_output_item_for_chunk(chunk(role='assistant'));assert len(calls)==5
    monkeypatch.setattr('llmgo.rsi.responses_stream.version',lambda name:'1.105.3')
    assert not prepare_chat_iterator(cls())
