"""A remote replay can index a durable record before the live append indexes it."""
import json

import pytest

from llmgo.rsi.memory import MemoryStore
from llmgo.rsi.contracts import canonical


@pytest.mark.parametrize('conflict',[False,True])
def test_append_handles_identical_concurrent_replay_but_rejects_conflict(tmp_path,monkeypatch,conflict):
    memory=MemoryStore(tmp_path)
    original=memory.connect
    injected=False
    class Connection:
        def __init__(self,db):self.db=db
        def __enter__(self):self.db.__enter__();return self
        def __exit__(self,*args):return self.db.__exit__(*args)
        def execute(self,sql,args=()):
            nonlocal injected
            if sql.startswith('INSERT OR IGNORE INTO events') and not injected:
                injected=True
                event=json.loads(memory.events.read_text().splitlines()[-1])
                if conflict:event['body']={'text':'different content'}
                # Replay uses another connection after append made its record durable.
                with original() as other:
                    other.execute('INSERT INTO events VALUES(?,?,?,?)',
                        (event['id'],event['ts'],event['kind'],canonical(event)))
            return self.db.execute(sql,args)
    monkeypatch.setattr(memory,'connect',lambda:Connection(original()))
    if conflict:
        with pytest.raises(ValueError,match='concurrent journal replay'):
            memory.append('research_note',{'text':'keep evidence'},event_id='same-record')
        assert 'keep evidence' in memory.events.read_text()
    else:
        event=memory.append('research_note',{'text':'keep evidence'},event_id='same-record')
        assert injected and memory.recent(1)[0]==event
        assert len(memory.events.read_text().splitlines())==1
        assert MemoryStore(tmp_path).recent(1)[0]==event
