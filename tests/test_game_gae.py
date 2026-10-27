"""Complete-episode math and actual chess rule boundaries, without GPUs."""
import copy
from pathlib import Path
import random

import chess
import pytest
from llmgo.training.game import ChessEpisode
from llmgo.training.trajectories import game_gae, request_weights


def rows(values=(.2,.3,.4),result=-1):
    return [{'episode_id':'game','policy_version':'v1','student_color':'white','turn_index':i,
        'value':v,'reward':float(result) if i==len(values)-1 else 0.,'done':i==len(values)-1,
        'requests':[{'response_tokens':2}]} for i,v in enumerate(values)]


def test_gae_point95_includes_future_td_not_just_current_delta():
    result = game_gae(rows(),gamma=1.,lam=.95)
    assert [r['delta'] for r in result]==pytest.approx([.1,.1,-1.4])
    assert [r['advantage'] for r in result]==pytest.approx([-1.0685,-1.23,-1.4])
    assert result[-1]['bootstrap_mask']==0 and result[-1]['next_value']==0


@pytest.mark.parametrize('outcome',[-1,0,1])
def test_lambda_one_equals_real_return_minus_state_baseline(outcome):
    values=[random.Random(i).uniform(-1,1) for i in range(30)]
    calculated=game_gae(rows(values,outcome),lam=1.)
    assert [r['advantage'] for r in calculated]==pytest.approx([outcome-v for v in values])
    assert [r['return'] for r in calculated]==pytest.approx([outcome]*len(values))


def test_terminal_reward_is_not_double_bootstrapped():
    r=game_gae(rows([.6],1))[0]
    assert r['delta']==pytest.approx(.4)
    assert r['advantage']==pytest.approx(.4) and r['return']==1
    assert game_gae(rows([-1],-1))[0]['advantage']==0


@pytest.mark.parametrize('mutation',[lambda r:r[0].update(episode_id='other'),
    lambda r:r[1].update(policy_version='new'),lambda r:r[1].update(student_color='black'),
    lambda r:r[1].update(turn_index=0),lambda r:r[-1].update(done=False),
    lambda r:r[0].update(reward=.1),lambda r:r[0].update(aborted=True),
    lambda r:r[0].update(value=float('nan'))])
def test_mixed_or_aborted_or_shaped_trajectories_are_rejected(mutation):
    r=rows();mutation(r)
    with pytest.raises(ValueError):game_gae(r)


def test_retry_weights_keep_one_move_equal_to_another():
    r=rows([0.,0.],1)
    r[0]['requests']=[{'response_tokens':2},{'response_tokens':6}]
    weights=request_weights(r)
    assert weights==[[.375,1.125],[1.5]]
    # verl divides by 3 request rows; each chess move contributes exactly 1/2.
    assert sum(weights[0])/3==sum(weights[1])/3==.5


class Engine:
    def __init__(self,*a,**kw):self.closed=False
    def _play_board(self,board):
        move=chess.Move.from_uci('d8h4') if board.fullmove_number==2 and board.turn==chess.BLACK else next(iter(board.legal_moves))
        return move.uci(),{}
    def close(self):self.closed=True


def test_opponent_checkmate_terminates_previous_student_transition(tmp_path):
    board=chess.Board()
    for move in ['f2f3','e7e5']:board.push_uci(move)
    ep=ChessEpisode('x','v1','white',1,'unused',tmp_path/'x',initial_fen=board.fen(),engine_factory=Engine)
    ep.act('g2g4',[{'response_tokens':2}])
    assert ep.done and ep.result==-1
    assert ep.transitions[-1]['reward']==-1 and ep.transitions[-1]['done']
    assert ep.board.is_checkmate()
    ep.close();assert ep.engine.closed


def test_protocol_failure_plays_random_legal_move_but_infrastructure_still_aborts(tmp_path):
    import json
    ep=ChessEpisode('x','v1','white',1,'unused',tmp_path/'x',engine_factory=Engine)
    original=ep.board.copy()
    ep.act(None,[{'response_tokens':3,'failure_reason':'invalid_final_answer'}])
    row=ep.transitions[0]
    assert not ep.done and ep.result is None and row['reward']==0
    assert row['action_source']=='random' and row['model_uci'] is None
    assert chess.Move.from_uci(row['uci']) in original.legal_moves
    assert row['requests'][0]['failure_reason']=='invalid_final_answer'
    ep.write_summary()
    summary=json.loads((ep.output/'summary.json').read_text())
    assert summary['random_moves']==1 and summary['random_rate']==1
    other=ChessEpisode('y','v1','black',1,'unused',tmp_path/'y',engine_factory=Engine)
    other.abort('connection lost')
    assert other.aborted and not other.done and other.result is None
    ep.close();other.close()


def test_random_fallback_is_seeded_and_preserves_all_request_evidence(tmp_path):
    import json
    episodes=[ChessEpisode(str(i),'v1','white',1,'unused',tmp_path/str(i),seed=99,engine_factory=Engine)
              for i in range(2)]
    request={'response_tokens':3,'failure_reason':'length','response':'unfinished thinking',
             'response_ids':[11,12,13],'behavior_logprobs':[-.1,-.2,-.3]}
    for ep in episodes:ep.act(None,[copy.deepcopy(request)],fallback_reason='attempts_exhausted')
    assert episodes[0].transitions[0]['uci']==episodes[1].transitions[0]['uci']
    for ep in episodes:
        assert ep.transitions[0]['requests']==[request]
        event=next(json.loads(line) for line in (ep.output/'events.jsonl').read_text().splitlines()
                   if json.loads(line)['type']=='fallback')
        assert event['scope']=='all_legal_moves' and event['legal_count']==20
        assert event['reason']=='attempts_exhausted' and event['last_failure_reason']=='length'
        ep.close()


def test_random_move_can_reach_a_real_terminal_without_protocol_reward(tmp_path):
    class MatePicker:
        def choice(self,legal):return next(move for move in legal if move.uci()=='b6b7')
    ep=ChessEpisode('mate','v1','white',1,'unused',tmp_path/'mate',
        initial_fen='k7/8/1QK5/8/8/8/8/8 w - - 0 1',engine_factory=Engine)
    ep.rng=MatePicker()
    ep.act(None,[{'response_tokens':2,'failure_reason':'invalid_final_answer'}])
    assert ep.done and ep.reason=='checkmate' and ep.result==1
    assert ep.transitions[0]['reward']==1 and ep.transitions[0]['action_source']=='random'
    assert game_gae([{**ep.transitions[0],'value':.2}])[0]['advantage']==pytest.approx(.8)
    ep.close()


def test_game_200_half_ply_cap_is_draw(tmp_path):
    ep=ChessEpisode('x','v1','white',1,'unused',tmp_path/'x',engine_factory=Engine)
    rng=random.Random(7)
    for attempt in range(100):
        board=chess.Board()
        while len(board.move_stack)<200 and not board.outcome(claim_draw=True):
            candidates=[]
            for move in board.legal_moves:
                child=board.copy(stack=True);child.push(move)
                if not child.outcome(claim_draw=True):candidates.append(move)
            if not candidates:break
            captures=[m for m in candidates if board.is_capture(m) or board.piece_at(m.from_square).piece_type==chess.PAWN]
            board.push(rng.choice(captures if board.halfmove_clock>70 and captures else candidates))
        if len(board.move_stack)==200 and not board.outcome(claim_draw=True):break
    assert len(board.move_stack)==200 and not board.outcome(claim_draw=True)
    ep.board=board
    assert ep.boundary() and ep.result==0 and ep.reason=='protocol_cap_draw'
    ep.close()


def test_black_initial_opponent_move_is_not_student_training_action(tmp_path):
    ep=ChessEpisode('x','v1','black',1,'unused',tmp_path/'x',engine_factory=Engine)
    ep.advance()
    assert ep.board.turn==chess.BLACK and len(ep.board.move_stack)==1 and ep.transitions==[]
    ep.close()
