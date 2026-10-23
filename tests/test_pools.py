import pytest

from llmgo.remote.pools import own_pool_rows,authorized_pool


@pytest.mark.parametrize('response',[
    [{'name':'model-autosft'}],
    {'pools':[{'name':'model-autosft'}]},
    {'success':True,'data':[{'name':'model-autosft'}]},
    {'ok':True,'data':{'pools':[{'alias':'model-autosft','name':'resource-pools/525'}]}}
])
def test_supported_cli_formats_find_authorized_pool(response):
    assert authorized_pool(own_pool_rows(response))


@pytest.mark.parametrize('response',[None,{},'bad',{'success':False,'data':[{'name':'model-autosft'}]},
    {'ok':False,'pools':[{'name':'model-autosft'}]},{'data':['model-autosft']}])
def test_malformed_or_failed_responses_never_grant_access(response):
    with pytest.raises(ValueError):own_pool_rows(response)


def test_empty_or_unrelated_pools_do_not_grant_access():
    assert not authorized_pool(own_pool_rows({'data':[]}))
    assert not authorized_pool([{'name':'other'},{'description':'model-autosft'}])
