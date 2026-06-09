from nanovllm.models.base import (
    DecoderAttention,
    DecoderMLP,
    DecoderLayer,
    DecoderModel,
    DecoderLM,
)


class LlamaAttention(DecoderAttention):
    pass


class LlamaMLP(DecoderMLP):
    pass


class LlamaDecoderLayer(DecoderLayer):
    pass


class LlamaModel(DecoderModel):
    pass


class LlamaForCausalLM(DecoderLM):
    pass
