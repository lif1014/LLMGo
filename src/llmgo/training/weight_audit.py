"""Streaming numeric checkpoint comparison, insensitive to dtype-only conversion."""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path


def compare_weights(base, trained, *, chunk_rows=128):
    import torch
    from safetensors import safe_open
    if type(chunk_rows) is not int or chunk_rows<1:
        raise ValueError('chunk_rows must be positive')
    with ExitStack() as stack:
        def index(folder):
            entries={}
            for file in sorted(Path(folder).glob('*.safetensors')):
                opened=stack.enter_context(safe_open(str(file),framework='pt',device='cpu'))
                for name in opened.keys():
                    if name in entries:raise ValueError('Duplicate checkpoint tensor '+name)
                    entries[name]=(opened,opened.get_slice(name))
            if not entries:raise ValueError('No safetensors in '+str(folder))
            return entries
        before,after=index(base),index(trained)
        unexpected=set(after)-set(before)
        missing=[key for key in set(before)-set(after) if not key.startswith('mtp.')]
        if unexpected or missing:
            raise ValueError(f'Checkpoint tensor keys differ: unexpected={sorted(unexpected)[:8]}, missing={sorted(missing)[:8]}')
        exact,served,dtype_changes,nonfinite=[],[],[],[]
        compared=0
        for name in sorted(before.keys() & after.keys()):
            old,old_slice=before[name];new,new_slice=after[name]
            shape=old_slice.get_shape()
            if shape!=new_slice.get_shape():raise ValueError('Checkpoint shape changed '+name)
            raw_changed=served_changed=False
            floating=False
            for start in range(0,shape[0] if shape else 1,chunk_rows):
                a=old_slice[start:start+chunk_rows] if shape else old.get_tensor(name)
                b=new_slice[start:start+chunk_rows] if shape else new.get_tensor(name)
                floating=torch.is_floating_point(a)
                if floating and (not torch.isfinite(a).all() or not torch.isfinite(b).all()):
                    nonfinite.append(name);break
                promoted=torch.promote_types(a.dtype,b.dtype)
                raw_changed |= not torch.equal(a.to(promoted),b.to(promoted))
                # BF16 serving cannot observe changes that round back to identical
                # BF16 weights. Preserve FP32 precision for base FP32 tensors.
                served_changed |= not torch.equal(a,b.to(a.dtype))
                if a.dtype!=b.dtype and name not in dtype_changes:dtype_changes.append(name)
                del a,b
            compared+=1
            text_parameter=floating and 'visual' not in name and not name.startswith('mtp.')
            if text_parameter and raw_changed:exact.append(name)
            if text_parameter and served_changed:served.append(name)
        return {'ok':bool(exact) and not nonfinite,'compared_tensors':compared,
            'changed_text_numeric_tensors':len(exact),'changed_text_tensors':len(exact),
            'changed_in_base_storage_precision_tensors':len(served),
            'base_storage_precision_changed':bool(served),
            'changed_text_examples':exact[:20],'dtype_changed_tensors':len(dtype_changes),
            'nonfinite_tensors':nonfinite[:20],
            'comparison':'numeric equality and cast to each base tensor storage dtype; not file/dtype inequality',
            'note':'Accept finite numeric parameter changes; base-storage precision is diagnostic only. Actual serving and chess strength still require evaluation.'}
