"""Offline LC0 value service; complete games assign environment rewards."""
from pathlib import Path
from llmgo.rsi.contracts import atomic_json, utc
from llmgo.training.lc0 import Lc0Value
from llmgo.training.rollout import board_from_record
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class Lc0Scorer:
    def __init__(self, site, config, output):
        self.provider = Lc0Value(site,config)
        self.output, self.sequence = Path(output), 0

    def ready(self):
        self.provider._ensure()
        return self.provider.identity

    def score_game(self, transitions):
        result = []
        for row in transitions:
            board = board_from_record(row["record"])
            actor = row["student_color"] == "white"
            if board.turn != actor:
                raise ValueError("LC0 state is not the student's decision")
            value, info = self.provider.value(board,actor)
            self.sequence += 1
            path = self.output/"lc0"/f"sample-{self.sequence:08d}.json"
            if path.exists():
                raise FileExistsError("LC0 evidence would be overwritten")
            atomic_json(path,{"episode_id":row["episode_id"],"turn_index":row["turn_index"],
                "policy_version":row["policy_version"],"student_color":row["student_color"],
                "record":row["record"],"value":value,"info":info,"utc":utc()})
            result.append({"value":value,"value_info":info,"value_path":str(path)})
        return result

    def close(self):
        self.provider.close()


class ChessRewardManager(RewardManagerBase):
    """Generation cannot know the future result: zero until the game is complete.

    The collector later assigns terminal r once; move-axis GAE fills advantages.
    LC0 TD must never appear as reward_score here.
    """
    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        # verl's loader passes routing/examination kwargs even for rule rewards.
        # The frozen LC0 service is called only after the complete game ends.
        super().__init__(config, tokenizer, compute_score)

    async def run_single(self, data):
        return {"reward_score":0.,"reward_extra_info":{}}
