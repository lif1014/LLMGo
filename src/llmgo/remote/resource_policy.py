"""Current operator allocation policy; historical task reads remain independent."""
SUBMISSION_POOLS = {'model-new': 523, 'model-claw-tool': 515, 'model-codeagent2': 514}
HISTORICAL_POOL_IDS = {**SUBMISSION_POOLS, 'model-autosft': 525}


def validate_submission(pool, priority, cluster='beijing'):
    if cluster != 'beijing' or pool not in SUBMISSION_POOLS or priority != 'PREEMPTABLE':
        raise ValueError('Only beijing model-new/model-claw-tool/model-codeagent2 PREEMPTABLE submissions are authorized; model-autosft is disabled')
