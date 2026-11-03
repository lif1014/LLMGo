"""Checkpoint-bound training decisions, independent of scheduler polling."""
from pathlib import Path
import fcntl

from .contracts import atomic_json, digest, file_hash, read_json, utc

ACTIONS = {'continue', 'replan', 'finish'}


class ReviewExpired(ValueError):
    pass


def publish(folder, payload):
    folder = Path(folder)
    request = {'request_id':digest(payload), 'payload':payload}
    path = folder/'request.json'
    if path.exists() and read_json(path) != request:
        raise ValueError('Training review identity changed')
    atomic_json(path, request)
    return path


def request(path):
    value = read_json(path)
    if value['request_id'] != digest(value['payload']):
        raise ValueError('Corrupt training review')
    for ref in value['payload']['evaluations']:
        if file_hash(ref['path']) != ref['sha256']:
            raise ValueError('Training validation evidence changed')
    ref = value['payload']['checkpoint']
    if file_hash(ref['path']) != ref['sha256']:
        raise ValueError('Training checkpoint receipt changed')
    return value


def validate(path, value):
    if Path(path).with_name('expired.json').exists():
        raise ReviewExpired('Training review expired; checkpoint is retained')
    current = request(path)
    p = current['payload']
    if (not isinstance(value,dict) or value.get('request_id') != current['request_id']
            or value.get('action') not in ACTIONS
            or not isinstance(value.get('reason'),str) or not value['reason'].strip()):
        raise ValueError('Decision requires this request ID, action and reason')
    action = value['action']
    if action == 'continue' and p['step'] >= p['max_steps']:
        raise ValueError('Training plan exhausted; replan or finish')
    step = value.get('selected_step', p['step'])
    if type(step) is not int or step not in p['available_steps']:
        raise ValueError('Select an evaluated checkpoint')
    if action == 'continue' and step != p['step']:
        raise ValueError('Continue resumes the current optimizer checkpoint')
    return {'request_id':current['request_id'],'action':action,'reason':value['reason'],
            'selected_step':step}


def commit(path, value):
    path = Path(path)
    with path.with_name('review.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        value = validate(path,value)
        target = path.with_name('decision.json')
        if target.exists():
            old = read_json(target)
            if {k:old[k] for k in value} != value:
                raise ValueError('A committed training decision cannot be replaced')
            return old
        value['committed_at'] = utc()
        atomic_json(target,value)
        return value


def expire(path):
    path=Path(path)
    with path.with_name('review.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.with_name('decision.json').exists():
            return False
        atomic_json(path.with_name('expired.json'),{'request_id':request(path)['request_id'],'utc':utc()})
        return True


def decision(path):
    path = Path(path)
    target = path.with_name('decision.json')
    if not target.exists():
        return None
    result = read_json(target)
    validate(path,result)
    return result
