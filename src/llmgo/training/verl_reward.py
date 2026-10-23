"""One reserved GPU LC0 service and an explicit token-aware verl reward manager."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path

from llmgo.rsi.contracts import atomic_json, read_json, utc
from llmgo.training.rewards import Lc0Reward, evaluate_action
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


class Lc0Scorer:
    # Ray assigns exactly the reserved GPU through CUDA_VISIBLE_DEVICES.
    def __init__(self, site, config, output):
        self.provider = Lc0Reward(site,config)
        self.config, self.output, self.sequence = config, Path(output), 0

    def ready(self):
        self.provider._ensure()
        return self.provider.identity

    def score(self, record_json, text, truncated, response_ids):
        record = json.loads(record_json)
        row = evaluate_action(record,text,self.provider,truncated=truncated,reward_config=self.config)
        self.sequence += 1
        row.update(sequence=self.sequence,utc=utc(),record=record,
                   response=text,response_ids=response_ids)
        sample = self.output/"lc0"/f"sample-{self.sequence:08d}.json"
        # A new process after resume writes to its new job output; no samples are overwritten.
        if sample.exists():
            raise FileExistsError("LC0 sample log already exists")
        atomic_json(sample,row)
        return {"score":row["advantage"], "value_before":row["value_before"],
            "value_after":row["value_after"] if row["value_after"] is not None else 0.,
            "environment_reward":row["environment_reward"],"bootstrap":row["bootstrap"],
            "terminated":row["terminated"],"failure_reason":row["failure_reason"] or "",
            "sample_path":str(sample)}

    def close(self):
        self.provider.close()


class ChessRewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config,tokenizer,compute_score)
        import ray
        context = read_json(config.reward.lc0_context)
        self.scorer = ray.get_actor(context["name"], namespace=context["namespace"])

    async def run_single(self, data):
        import ray
        item = data[-1:][0]
        response = item.batch["responses"]
        length = int(item.batch["attention_mask"][-response.shape[-1]:].sum())
        if length < 1:
            raise ValueError("Empty rollout response")
        ids = response[:length].tolist()
        # Qwen's think delimiters can be special tokens. Do not strip them.
        text = self.tokenizer.decode(ids,skip_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        truncated = length >= self.config.data.max_response_length and ids[-1] not in eos
        ref = self.scorer.score.remote(item.non_tensor_batch["extra_info"]["record_json"],text,truncated,ids)
        result = await asyncio.to_thread(ray.get,ref)
        return {"reward_score":result["score"],"reward_extra_info":result}
