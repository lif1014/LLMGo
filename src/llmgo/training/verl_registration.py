"""Install the complete-game orchestration inside verl's Ray TaskRunner."""
import os
from pathlib import Path

from .vllm_compat import install_rope_compat
from .verl_memory import install_wake_cache_flush

install_rope_compat()
if os.environ.get('LLMGO_WAKE_CACHE_FLUSH','1')=='1':
    install_wake_cache_flush()

from verl.trainer.main_ppo import TaskRunner


class ResearchTaskRunner(TaskRunner):
    def run(self, config):
        # TaskRunner constructs RayPPOTrainer internally. Keep all native worker
        # initialization/update/checkpoint code; scope the custom fit to this task.
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from .verl_games import fit_games
        original = RayPPOTrainer.fit
        RayPPOTrainer.fit = fit_games
        try:
            return super().run(config)
        finally:
            RayPPOTrainer.fit = original
