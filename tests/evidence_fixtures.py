"""Small raw games for Controller tests that now consume deterministic evidence."""
import json
from pathlib import Path
from llmgo.rsi.contracts import file_hash
from llmgo.rsi.ladder import batch_stats


def with_raw_games(metrics,folder,protocol):
    for game in metrics['games']:
        result=0 if game['result']=='draw' else 1 if game['result']==game['model_seat']+'_win' else -1
        reason='fixture_draw' if result==0 else 'checkmate'
        path=Path(folder)/'games'/game['game_id']/'events.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        events=[{'type':'episode_started','ply':0,'episode_id':game['game_id'],'student_color':game['model_seat']},
            {'type':'student_move','ply':1,'uci':'e2e4','fen':'fixture'},
            {'type':'episode_finished','ply':1,'result':result,'reason':reason}]
        path.write_text(''.join(json.dumps(e)+'\n' for e in events))
        game.update(raw_events=str(path),event_hashes={'raw_events':file_hash(path)},
            diagnostics={'result':result,'plies':1},result_reason=reason)
    metrics.update(protocol=protocol,statistics=batch_stats(metrics,expected_games=protocol['games']))
    return metrics
