from transformers import PretrainedConfig

from nanovllm.models.base import (
    DecoderAttention,
    DecoderMLP,
    DecoderLayer,
    DecoderModel,
    DecoderLM,
)


class Qwen3Attention(DecoderAttention):
    pass


class Qwen3MLP(DecoderMLP):
    pass


class Qwen3DecoderLayer(DecoderLayer):

    def __init__(self, config: PretrainedConfig):
        qk_norm = not getattr(config, 'attention_bias', True)
        super().__init__(config, qk_norm=qk_norm)


class Qwen3Model(DecoderModel):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config, layer_cls=Qwen3DecoderLayer)


class Qwen3ForCausalLM(DecoderLM):

    def __init__(self, config: PretrainedConfig):
        super().__init__(config, model_cls=Qwen3Model)
