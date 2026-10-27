"""A precision conversion must not count as an RL policy update."""
from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
safetensors=pytest.importorskip('safetensors.torch')
from llmgo.training.weight_audit import compare_weights


def save(path,values):
    path.mkdir()
    safetensors.save_file(values,str(path/'model.safetensors'))
    return path


def test_bf16_to_fp32_only_is_not_effective_training(tmp_path):
    w=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)
    before=save(tmp_path/'base',{'model.weight':w,'mtp.weight':torch.ones(1)})
    after=save(tmp_path/'trained',{'model.weight':w.float()})
    result=compare_weights(before,after,chunk_rows=1)
    assert not result['ok'] and result['changed_text_numeric_tensors']==0
    assert result['dtype_changed_tensors']==1


def test_actual_served_change_with_different_dtype_is_detected(tmp_path):
    before=save(tmp_path/'base',{'model.weight':torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16)})
    after=save(tmp_path/'trained',{'model.weight':torch.tensor([[1.,2.],[3.,4.125]],dtype=torch.float32)})
    result=compare_weights(before,after,chunk_rows=1)
    assert result['ok'] and result['changed_text_tensors']==1


def test_fp32_only_sub_bf16_change_is_recorded_separately(tmp_path):
    before=save(tmp_path/'base',{'model.weight':torch.tensor([1.],dtype=torch.bfloat16)})
    after=save(tmp_path/'trained',{'model.weight':torch.tensor([1.000001],dtype=torch.float32)})
    result=compare_weights(before,after)
    assert result['ok'] and result['changed_text_numeric_tensors']==1
    assert result['changed_in_base_storage_precision_tensors']==0
    assert not result['base_storage_precision_changed']


def test_nonfinite_export_and_shape_change_are_rejected(tmp_path):
    before=save(tmp_path/'base',{'model.weight':torch.ones((2,2))})
    bad=save(tmp_path/'bad',{'model.weight':torch.full((2,2),float('nan'))})
    assert not compare_weights(before,bad)['ok']
    wrong=save(tmp_path/'wrong',{'model.weight':torch.ones((1,2))})
    with pytest.raises(ValueError,match='shape'):compare_weights(before,wrong)
