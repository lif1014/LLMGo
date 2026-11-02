"""verl0.8 RewardManager using a separate persistent original270M service."""
import asyncio
import json
import os
import urllib.request
import urllib.error
from pathlib import Path

from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from llmgo.training.chess_r1 import extract_answer


class ChessR1RewardManager(RewardManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.url = os.environ['CHESS_R1_ORACLE_URL']
        self._oracle_slots = asyncio.Semaphore(1)
        token_file = os.environ.get('CHESS_R1_ORACLE_TOKEN_FILE')
        self._oracle_token = Path(token_file).read_text().strip() if token_file else None

    async def run_single(self, data):
        item = data[0]
        length = int(item.batch['attention_mask'][-item.batch['responses'].shape[-1]:].sum())
        ids = item.batch['responses'][:length]
        response = self.tokenizer.decode(ids, skip_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        truncated = length >= self.config.data.max_response_length and int(ids[-1]) not in eos
        payload = {'ground_truth': item.non_tensor_batch['reward_model']['ground_truth'],
                   'response': response, 'finish_reason': 'length' if truncated else 'stop'}
        if item.non_tensor_batch.get('extra_info', {}).get('split') in ('dev', 'test'):
            # Original Lichess validation needs the solution, not a teacher call.
            gt = payload['ground_truth']
            if isinstance(gt, str):
                gt = json.loads(gt)
            answer = extract_answer(response, gt['board_fen'], finish_reason=payload['finish_reason'])
            correct = int(answer.legal and answer.san == gt['next_move_san'])
            return {'reward_score': float(correct), 'reward_extra_info': {
                'solution_correct': correct, 'puzzle_id': gt['id'], 'rating': int(gt['rating']),
                'format': int(answer.format_valid), 'legal': int(answer.legal),
                'response_tokens': length, 'truncated': int(truncated),
                'san': answer.san or '', 'uci': answer.uci or '', 'parse_error': answer.error or ''}}
        # The original oracle returns exactly zero for a malformed/truncated
        # answer, without querying Q or the language detector. Do this before
        # sending potentially very large degenerate generations over HTTP.
        gt = payload['ground_truth']
        if isinstance(gt, str):
            gt = json.loads(gt)
        answer = extract_answer(response, gt['board_fen'], finish_reason=payload['finish_reason'])
        if not answer.format_valid:
            result = {'score': 0., 'puzzle_id': gt['id'], 'q': 0., 'format': 0,
                      'english': 0, 'legal': 0, 'solution_correct': 0, 'oracle_top1': 0,
                      'q_gap': 0., 'san': answer.san or '', 'uci': answer.uci or '',
                      'parse_error': answer.error or '', 'response_tokens': length,
                      'truncated': int(truncated)}
            return {'reward_score': 0., 'reward_extra_info': result}
        def query():
            headers = {'Content-Type': 'application/json'}
            if self._oracle_token:
                headers['X-Chess-Oracle-Token'] = self._oracle_token
            encoded = json.dumps(payload).encode()
            request = urllib.request.Request(self.url, data=encoded,
                                             headers=headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                with opener.open(request, timeout=300) as result:
                    return json.load(result)
            except urllib.error.HTTPError as error:
                detail = error.read(8192).decode(errors='replace')
                raise RuntimeError(f'Original270M HTTP {error.code}; puzzle={gt["id"]}; '
                                   f'request_bytes={len(encoded)}; response_chars={len(response)}; '
                                   f'oracle_error={detail}') from error
        async with self._oracle_slots:
            result = await asyncio.to_thread(query)
        result.update(response_tokens=length, truncated=int(truncated))
        return {'reward_score': result['score'], 'reward_extra_info': result}
