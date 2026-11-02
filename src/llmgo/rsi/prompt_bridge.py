"""Isolated one-request adapter for a candidate's optional prompt harness."""
import importlib.util
import json
import sys
from pathlib import Path


def main():
    candidate = Path(sys.argv[1]).resolve()
    request = json.loads(sys.stdin.readline())
    source = candidate / 'harness.py'
    spec = importlib.util.spec_from_file_location('candidate_harness', source)
    if spec is None or spec.loader is None:
        raise RuntimeError('Candidate harness cannot be imported')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    messages = module.build_messages(request['fen'], request['legal_san'],
                                     request['skills'], request['default_messages'])
    if (not isinstance(messages, list) or not 1 <= len(messages) <= 16 or
            any(not isinstance(row, dict) or row.get('role') not in {'system', 'user', 'assistant'}
                or not isinstance(row.get('content'), str) or len(row['content']) > 65536
                for row in messages)):
        raise ValueError('Candidate harness must return 1-16 text chat messages')
    print(json.dumps(messages, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
