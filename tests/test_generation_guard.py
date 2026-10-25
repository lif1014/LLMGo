import random
import string

from llmgo.rsi.generation_guard import StreamRepetitionGuard


def test_exact_live_failure_pattern_detects_across_arbitrary_chunks():
    pattern='Let me write.\n\nFINAL.\n\nOutput.\n\nGo.\n\nWriting.\n\nNow.\n\nOK.\n\n'
    text=pattern*200
    guard=StreamRepetitionGuard();found=None
    rng=random.Random(4170);i=0
    while i<len(text) and not found:
        n=rng.randrange(1,90);found=guard.feed('one-item',text[i:i+n]);i+=n
    assert found and found['cycles']>=12 and found['repeated_chars']>=2048
    assert i<4000 and len(guard.tail)<=8192


def test_stream_identity_normal_text_and_numeric_tables_are_not_loops():
    guard=StreamRepetitionGuard()
    for i in range(50):
        assert guard.feed(str(i),'Writing. '*100) is None
    rng=random.Random(5)
    for _ in range(100):
        assert guard.feed('varied',''.join(rng.choices(string.ascii_letters+' ',k=200))) is None
    assert guard.feed('numeric','0123456789\n'*1000) is None
    assert len(guard.tail)<=8192


def test_activity_distinguishes_repetition_recovery_from_success():
    from llmgo.store.activity import project
    e=project({'kind':'meta_generation_loop','body':{'thread_id':'t','turn_id':'u','cycles':36,'repeated_chars':2088}})
    assert e['category']=='error' and e['turn_id']=='u'
    e=project({'kind':'meta_generation_recovery','body':{'thread_id':'new','attempt':1}})
    assert e['category']=='lifecycle' and e['thread_id']=='new'
    assert project({'kind':'meta_generation_exhausted','body':{}})['category']=='error'
