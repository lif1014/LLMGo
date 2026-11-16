"""Shared executable proposal contract, checked before handoff and freezing."""
from .contracts import digest
from .ladder import evaluation_plan


def validate_proposal(proposal,*,require_training=False):
    digest(proposal)
    if not isinstance(proposal,dict):raise ValueError('proposal.json must be an object')
    hypothesis=proposal.get('hypothesis')
    if not isinstance(hypothesis,str) or not hypothesis.strip() or hypothesis=='Describe the next experiment':
        raise ValueError('hypothesis must be a nonempty string, not an object/template; put details in research_notes')
    jobs=proposal.get('jobs',[])
    if not isinstance(jobs,list):raise ValueError('jobs must be a list of commands')
    if not jobs and (require_training or 'training_config' in proposal):
        raise ValueError('Training requires jobs[].argv; training_config is descriptive and does not launch training')
    for command in jobs:
        if not isinstance(command,dict):raise ValueError('Each job must be an object with argv')
        resources=command.get('resources',{})
        if not isinstance(resources,dict) or set(resources)-{'gpu_count','cpu','memory_gib'}:raise ValueError('Invalid job resources')
        for key,value in resources.items():
            if type(value) is not int or value<(0 if key=='gpu_count' else 1) or key=='gpu_count' and value>8:raise ValueError('Invalid job resources')
        argv=command.get('argv')
        if not isinstance(argv,list) or not argv or any(not isinstance(v,str) or not v or '\0' in v for v in argv):
            raise ValueError('Each job requires a nonempty argv string array')
    gpus=proposal.get('evaluation_gpus',8)
    if type(gpus) is not int or not 1<=gpus<=8:raise ValueError('evaluation_gpus must be an integer from 1 to 8')
    batch=proposal.get('evaluation_batch',0)
    if type(batch) is not int or batch<0:raise ValueError('evaluation_batch must be a nonnegative integer')
    if proposal.get('evaluation',True) is not False:evaluation_plan(proposal.get('evaluation'))
    return proposal


def validate_workspace_proposal(workspace, proposal, *, require_training=False):
    """Use the actual request context, so a review cannot pass as an empty study."""
    from pathlib import Path
    from .contracts import read_json
    from .training_review import validate_choice
    path = Path(workspace)/'context.json'
    context = read_json(path) if path.is_file() else {}
    if 'training_review' in context:
        if not isinstance(proposal, dict):
            raise ValueError('Review proposal must contain training_review')
        validate_choice(context['training_review'], proposal.get('training_review'))
        return proposal
    if isinstance(proposal, dict) and 'training_review' in proposal:
        raise ValueError('training_review requires the current workspace review context')
    return validate_proposal(proposal, require_training=require_training)
