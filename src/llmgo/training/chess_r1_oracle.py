"""Persistent original 270M action-value oracle and original English detector.

Runs in its own CPU JAX environment; the verl/PyTorch environment is untouched.
"""
from __future__ import annotations
import argparse
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import sys
import threading
import time

from llmgo.training.chess_r1 import extract_answer


def score_answer(response, ground_truth, qvalues, english, *, finish_reason=None):
    """Chess-R1 dense outcome reward with the user-selected bare-box protocol."""
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    answer = extract_answer(response, ground_truth['board_fen'], finish_reason=finish_reason)
    quality = float(qvalues[answer.uci]) if answer.legal else 0.
    if not 0 <= quality <= 1:
        raise ValueError('Invalid oracle Q')
    language = bool(english(response)) if answer.format_valid else False
    fmt = int(answer.format_valid)
    return {'score': quality + .1 * fmt + .1 * language,
            'puzzle_id': ground_truth['id'],
            'q': quality, 'format': fmt, 'english': int(language), 'legal': int(answer.legal),
            'solution_correct': int(answer.legal and answer.san == ground_truth['next_move_san']),
            'oracle_top1': int(answer.legal and quality == max(qvalues.values())),
            'q_gap': quality - max(qvalues.values()) if answer.legal else 0.,
            'san': answer.san or '', 'uci': answer.uci or '', 'parse_error': answer.error or ''}


class Oracle:
    def __init__(self, upstream, checkpoint, cache_dir):
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path.insert(0, str(upstream))
        import chess
        import jax
        import numpy as np
        import spacy
        import spacy_fastlang  # registers the unchanged FastText language detector
        from searchless_chess.src import tokenizer, training_utils, transformer, utils
        from searchless_chess.src.engines import engine, neural_engines
        self.np, self.chess, self.engine = np, chess, engine
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        config = transformer.TransformerConfig(
            vocab_size=utils.NUM_ACTIONS, output_size=128,
            pos_encodings=transformer.PositionalEncodings.LEARNED,
            max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
            num_heads=8, num_layers=16, embedding_dim=1024,
            apply_post_ln=True, apply_qk_layernorm=False, use_causal_mask=False)
        predictor = transformer.build_transformer_predictor(config=config)
        template = predictor.initial_params(rng=jax.random.PRNGKey(0), targets=np.zeros((1, 1), dtype=np.uint32))
        params = training_utils.load_parameters(params=template, checkpoint_dir=str(checkpoint),
                                               use_ema_params=True, step=6400000)
        self.parameter_count = sum(x.size for x in jax.tree.leaves(params))
        del template
        _, self.buckets = utils.get_uniform_buckets_edges_values(128)
        original_predict = neural_engines.wrap_predict_fn(predictor, params, batch_size=32)
        # The upstream wrapper pads to32. Crop to actual actions before returning
        # to analyse(), without changing logits, legal ordering, or bucket values.
        self.network = neural_engines.ActionValueEngine(
            return_buckets_values=self.buckets,
            predict_fn=lambda sequences: original_predict(sequences)[:len(sequences)], temperature=.005)
        self.nlp = spacy.blank('en')
        self.nlp.max_length = 300000
        self.nlp.add_pipe('language_detector')
        self.lock = threading.Lock()

    def english(self, text):
        doc = self.nlp(text)
        return doc._.language == 'en' and doc._.language_score >= .9

    @functools.lru_cache(maxsize=4096)
    def values(self, fen):
        key = hashlib.sha256(('270M-6400000-ema|'+fen).encode()).hexdigest()
        path = self.cache_dir / (key+'.json')
        if path.exists():
            result = json.loads(path.read_text())
            if result['fen'] != fen:
                raise ValueError('Oracle cache identity mismatch')
            return result['q']
        board = self.chess.Board(fen)
        if not board.is_valid() or not any(board.legal_moves):
            raise ValueError('Invalid oracle position')
        moves = self.engine.get_ordered_legal_moves(board)
        logits = self.network.analyse(board)['log_probs']
        q = self.np.inner(self.np.exp(logits), self.buckets)
        if q.shape != (len(moves),) or not self.np.isfinite(q).all() or not ((q >= 0) & (q <= 1)).all():
            raise ValueError('Invalid 270M output shape or values')
        result = {m.uci(): float(v) for m, v in zip(moves, q, strict=True)}
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'fen': fen, 'q': result}, separators=(',', ':')))
        temp.replace(path)
        return result

    def score(self, body):
        gt = body['ground_truth']
        if isinstance(gt, str):
            gt = json.loads(gt)
        with self.lock:
            parsed = extract_answer(body['response'], gt['board_fen'], finish_reason=body.get('finish_reason'))
            values = self.values(gt['board_fen']) if parsed.legal else {}
            return score_answer(body['response'], gt, values, self.english,
                                finish_reason=body.get('finish_reason'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=0)
    a = p.parse_args()
    started = time.time()
    a.output.mkdir(parents=True, exist_ok=True)
    oracle = Oracle(a.upstream, a.checkpoint, a.output/'cache')
    fen = oracle.chess.STARTING_FEN
    with oracle.lock:
        probe = oracle.values(fen)
        language = oracle.english('This is a paragraph written in the English language. The purpose of this example is to check that the language detection model can correctly identify English text. I will carefully explain how the pieces can move and what the opponent is likely to do next.')
    if len(probe) != 20 or not language:
        raise RuntimeError('Oracle/language readiness check failed')
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 524288:
                    raise ValueError('Oracle input exceeds bound')
                result = oracle.score(json.loads(self.rfile.read(size)))
                raw = json.dumps(result).encode()
                self.send_response(200)
            except Exception as error:
                raw = json.dumps({'error_type': type(error).__name__, 'error': str(error)[:200]}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        def log_message(self, *a):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', a.port), Handler)
    ready = {'url': f'http://127.0.0.1:{server.server_port}', 'status': 'ready',
             'devices': ['cpu'], 'parameters': int(oracle.parameter_count), 'step': 6400000,
             'use_ema_params': True, 'return_buckets': 128, 'opening_q': probe,
             'language_detector': 'spacy_fastlang2.1.0/FastText, original en>=0.9 threshold',
             'seconds': time.time()-started}
    (a.output/'ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    server.serve_forever()


if __name__ == '__main__':
    main()
