"""Process-local compatibility for the existing remote Transformers/vLLM build."""
from .vllm_compat import install_rope_compat
from .verl_memory import install_wake_cache_flush
from .chess_r1_attention import install_3d_position_compat
install_rope_compat()
install_wake_cache_flush()
install_3d_position_compat()
