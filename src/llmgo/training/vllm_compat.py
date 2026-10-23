"""Process-local compatibility for the pinned vLLM 0.17 / Transformers 5.2 and 5.4 pair.

vLLM's Qwen3.5 backport passes a list of ignored RoPE keys. HF 5.x expects
set union. Normalize only that argument for vLLM's backported configs; never
edit installed packages or model config files.
"""
from functools import wraps


def install_rope_compat():
    from importlib.metadata import version
    if version("transformers") not in {"5.2.0", "5.4.0"} or version("vllm").split("+")[0] != "0.17.0":
        return False
    from transformers.configuration_utils import PretrainedConfig
    original = PretrainedConfig.convert_rope_params_to_dict
    if getattr(original, "_llmgo_rope_compat", False):
        return True

    @wraps(original)
    def convert(self, ignore_keys_at_rope_validation=None, **kwargs):
        if (type(self).__module__.startswith("vllm.transformers_utils.configs.qwen3_5")
                and isinstance(ignore_keys_at_rope_validation, (list, tuple))):
            ignore_keys_at_rope_validation = set(ignore_keys_at_rope_validation)
        return original(self, ignore_keys_at_rope_validation=ignore_keys_at_rope_validation, **kwargs)

    convert._llmgo_rope_compat = True
    PretrainedConfig.convert_rope_params_to_dict = convert
    return True
