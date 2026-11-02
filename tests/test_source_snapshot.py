import pytest

from llmgo.rsi.artifacts import source_snapshot, verify_source


def test_campaign_snapshot_can_live_in_excluded_var(tmp_path):
    repo = tmp_path/'repo'
    (repo/'src').mkdir(parents=True)
    (repo/'assets/models').mkdir(parents=True)
    (repo/'src/main.py').write_text('print(1)\n')
    (repo/'assets/models/weight.bin').write_bytes(b'large private asset')
    target = repo/'var/campaigns/example/source'
    snapshot = source_snapshot(repo, target)
    assert snapshot['files'] == {'src/main.py': snapshot['files']['src/main.py']}
    assert (target/'src/main.py').read_text() == 'print(1)\n'
    assert not (target/'assets').exists()
    verify_source(snapshot)
    with pytest.raises(ValueError):
        source_snapshot(repo, repo/'src/nested')
