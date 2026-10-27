"""Fixed validation identity and baseline-relative protocol regression checks."""
import math
from llmgo.rsi.contracts import digest


def validation_identity(settings, experiment_digest):
    return digest({'games':settings['validation_games'],'seed':settings['validation_seed'],
        'opponent':settings['opponent_level'],'sampling':[1.,1.,-1],
        'objective':experiment_digest})


def minimum_valid_rate(settings, baseline=None):
    """Accept the measured starting policy; reject a later material regression.

    This is an internal training check, independent of the protected referee.
    An absolute floor still blocks collapsed policies and all-invalid cohorts.
    """
    if baseline and (not math.isfinite(baseline['valid_rate']) or not 0<=baseline['valid_rate']<=1):
        raise ValueError('Invalid original validation rate')
    return max(settings['validation_min_valid'],
        baseline['valid_rate']-settings['validation_max_drop'] if baseline else 0.)
