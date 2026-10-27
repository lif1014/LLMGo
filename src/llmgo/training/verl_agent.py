"""One auditable model request; the trainer owns the complete chess episodes."""
import asyncio
import math
import os
from pathlib import Path
import time
import uuid

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from llmgo.rsi.contracts import atomic_json, utc


class ChessTurnAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        extra = kwargs["extra_info"]
        params = dict(sampling_params)
        params.update(seed=int(extra["seed"]),max_tokens=int(extra["max_tokens"]))
        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.apply_chat_template(messages)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError("Chess prompt exceeds its recorded budget; no silent truncation")
        receipt = {"request_id":uuid.uuid4().hex,"sampling":dict(params),
                   "prompt_tokens":len(prompt_ids),"enable_thinking":True,"started_at":utc()}
        trace_root = os.environ.get("LLMGO_VERL_OUTPUT")
        trace = Path(trace_root)/"agent_requests"/(receipt["request_id"]+".json") if trace_root else None
        def persist(status, **data):
            if trace is not None:
                atomic_json(trace,{"status":status,"receipt":dict(receipt),"utc":utc(),
                    "episode_id":extra.get("episode_id"),"turn_index":extra.get("turn_index"),
                    "attempt":extra.get("attempt"),"messages":messages,"prompt_ids":prompt_ids,**data})
        persist("generation_started")
        started = time.monotonic()
        try:
            output = await asyncio.wait_for(self.server_manager.generate(
                request_id=receipt["request_id"],prompt_ids=prompt_ids,sampling_params=dict(params)),
                timeout=float(extra["request_timeout_sec"]))
        except BaseException as error:
            receipt["seconds"] = time.monotonic()-started
            persist("failed_before_token_output",error_type=type(error).__name__,error=str(error))
            raise
        # Save before the reward-manager/batch barrier. A sibling request or a
        # downstream verl failure must not erase an already completed generation.
        receipt["seconds"] = time.monotonic()-started
        persist("generation_completed",response_ids=output.token_ids,
            response=self.tokenizer.decode(output.token_ids,skip_special_tokens=False),
            behavior_logprobs=[p if math.isfinite(p) else str(p) for p in output.log_probs]
                if output.log_probs is not None else None,backend_stop_reason=output.stop_reason)
        if output.stop_reason == "aborted":
            raise RuntimeError("Rollout server aborted generation; not a chess loss")
        ids = output.token_ids
        if (not ids or len(ids) > params["max_tokens"] or not output.log_probs
                or len(output.log_probs) != len(ids)
                or any(not math.isfinite(p) for p in output.log_probs)):
            raise ValueError("Missing/misaligned behavior logprobs or response token budget")
        eos = self.tokenizer.eos_token_id
        eos = eos if isinstance(eos,list) else [eos]
        finish = "length" if len(ids) >= params["max_tokens"] and ids[-1] not in eos else "stop"
        receipt.update(seconds=time.monotonic()-started,completion_tokens=len(ids),
            finish_reason=finish,finish_reason_source="token_budget_and_eos",
            backend_stop_reason=output.stop_reason)
        if trace is not None:
            receipt["trace_path"] = str(trace)
            persist("generation_completed",response_ids=ids,
                response=self.tokenizer.decode(ids,skip_special_tokens=False),behavior_logprobs=output.log_probs)
        return AgentLoopOutput(prompt_ids=prompt_ids,response_ids=ids,response_mask=[1]*len(ids),
            response_logprobs=output.log_probs,multi_modal_data={},num_turns=2,
            metrics=AgentLoopMetrics(generate_sequences=receipt["seconds"],
                num_preempted=output.num_preempted if output.num_preempted is not None else -1),
            extra_fields={"request_receipt":receipt,"turn_scores":[],"tool_rewards":[]})
