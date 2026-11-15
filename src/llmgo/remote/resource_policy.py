"""Current operator allocation policy; historical task reads remain independent."""
SUBMISSION_POOLS = {'model-autosft': 525, 'model-new': 523, 'model-claw-tool': 515, 'model-codeagent2': 514}
HISTORICAL_POOL_IDS = dict(SUBMISSION_POOLS)
META_PRIORITIES = {pool: 'NORMAL' if pool == 'model-autosft' else 'PREEMPTABLE' for pool in SUBMISSION_POOLS}
PRIORITY_IDS = {'HIGH': 1, 'NORMAL': 2, 'PREEMPTABLE': 3}


def validate_submission(pool, priority, cluster='beijing', *, purpose=None):
    allowed = {'HIGH', 'NORMAL'} if pool == 'model-autosft' else {'PREEMPTABLE'}
    if (cluster != 'beijing' or pool not in SUBMISSION_POOLS or priority not in allowed
            or purpose == 'meta' and priority != META_PRIORITIES[pool]):
        raise ValueError('Only authorized beijing submissions: training uses model-autosft/HIGH; Meta inference uses model-autosft/NORMAL or model-new/model-claw-tool/model-codeagent2 PREEMPTABLE')
