"""Pass the selected sampling penalties through native verl's single-turn loop."""
import json
import os
from pathlib import Path

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

_recorded_sampling = False


class ChessR1SingleTurnAgentLoop(SingleTurnAgentLoop):
    def __init__(self, *args, presence_penalty=1.5, repetition_penalty=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.presence_penalty = float(presence_penalty)
        self.repetition_penalty = float(repetition_penalty)
        if self.rollout_config.logprobs_mode != 'raw_logprobs':
            raise ValueError('Chess-R1 actor sync requires raw model log-probabilities')

    async def run(self, sampling_params, **kwargs):
        global _recorded_sampling
        params = dict(sampling_params)
        for key in ('temperature', 'top_p', 'top_k'):
            if params[key] != getattr(self.rollout_config, key):
                raise ValueError('Chess-R1 training and validation sampling must agree')
        params.update(presence_penalty=self.presence_penalty,
                      repetition_penalty=self.repetition_penalty)
        receipt = dict(params)
        output = await super().run(params, **kwargs)
        if not _recorded_sampling and os.environ.get('CHESS_R1_OUTPUT'):
            path = Path(os.environ['CHESS_R1_OUTPUT'])/'sampling-configuration.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps({'sampling': receipt, 'logprobs_mode': 'raw_logprobs',
                'native_single_turn_completed': True, 'response_tokens': len(output.response_ids)}))
            temp.replace(path)
            _recorded_sampling = True
        return output
