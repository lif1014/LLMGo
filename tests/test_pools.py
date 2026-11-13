import pytest

from llmgo.remote.pools import own_pool_rows,authorized_pool


@pytest.mark.parametrize('response',[
    [{'name':'model-new'}],
    {'pools':[{'name':'model-new'}]},
    {'success':True,'data':[{'name':'model-new'}]},
    {'ok':True,'data':{'pools':[{'alias':'model-new','name':'resource-pools/523'}]}}
])
def test_supported_cli_formats_find_authorized_pool(response):
    assert authorized_pool(own_pool_rows(response))


@pytest.mark.parametrize('response',[None,{},'bad',{'success':False,'data':[{'name':'model-new'}]},
    {'ok':False,'pools':[{'name':'model-new'}]},{'data':['model-new']}])
def test_malformed_or_failed_responses_never_grant_access(response):
    with pytest.raises(ValueError):own_pool_rows(response)


def test_empty_or_unrelated_pools_do_not_grant_access():
    assert not authorized_pool(own_pool_rows({'data':[]}))
    assert not authorized_pool([{'name':'other'},{'description':'model-new'}])
