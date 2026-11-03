"""One durable HIGH submission slot; unknown outcomes keep their reservation."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, inside, read_json


class AdmissionBusy(RuntimeError):
    pass


@contextmanager
def high_slot(transport,intent_path):
    folder=Path(transport.site['repo'])/'var/runtime/high-admission'
    folder.mkdir(parents=True,exist_ok=True)
    registry=folder/'reservations.json'
    terminal={'Succeeded','Completed','Failed','Killed','Stopped','Deleted','OOMKilled','Invalid','Expired'}
    with (folder/'lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        paths=read_json(registry) if registry.exists() else []
        busy=[]
        for path in paths:
            if not Path(path).exists():
                # The reservation precedes the intent; create cannot have run.
                continue
            intent=read_json(inside(transport.site['user_root'],path))
            if intent.get('task_id'):
                task=transport.command(['job','get',str(intent['task_id'])])
                if task.get('status') in terminal:
                    continue
            busy.append(path)
        atomic_json(registry,busy)
        if any(path!=str(intent_path) for path in busy):
            raise AdmissionBusy('Another HIGH submission is active or uncertain')
        # Include all own HIGH jobs, even those submitted outside this controller.
        for page in range(1,101):
            value=transport.command(['task','list','--own','--cluster','beijing','--pool','model-autosft',
                '--type','job,pytorchjob','--page',str(page),'--limit','100'])
            rows=value if isinstance(value,list) else value.get('tasks',value.get('items',[]))
            if not isinstance(rows,list) or any('status' not in row or
                    row['status'] not in terminal and 'priority' not in row for row in rows):
                raise AdmissionBusy('Own HIGH task status is incomplete')
            if any(str(row.get('priority')).upper()=='HIGH' and row.get('status') not in terminal for row in rows):
                raise AdmissionBusy('An own HIGH job already occupies the single submission slot')
            if len(rows)<100:
                break
        else:
            raise AdmissionBusy('Own HIGH job listing is incomplete; keep admission closed')
        if str(intent_path) not in busy:
            busy.append(str(intent_path))
        atomic_json(registry,busy)
        yield
