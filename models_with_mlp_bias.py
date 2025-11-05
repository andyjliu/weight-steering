import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2MLP,
    Qwen2Model,
)


class Qwen2MLPWithBias(Qwen2MLP):
    """Qwen2 MLP with bias support in down_proj"""

    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class Qwen2MLPWithBiasDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = Qwen2MLPWithBias(config)


class Qwen2ModelMLPWithBias(Qwen2Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Qwen2MLPWithBiasDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class LlamaMLPWithBias(LlamaMLP):
    """Llama MLP with bias support in down_proj"""

    def __init__(self, config):
        super().__init__(config)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


class LlamaDecoderLayerMLPWithBias(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = LlamaMLPWithBias(config)


class LlamaModelMLPWithBias(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayerMLPWithBias(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_init()


class Qwen2MLPBiasConfig(Qwen2Config):
    """Config for Qwen2 with MLP bias. Uses custom model_type to avoid conflicts."""

    model_type = "qwen2_mlp_bias"


class LlamaMLPBiasConfig(LlamaConfig):
    """Config for Llama with MLP bias. Uses custom model_type to avoid conflicts."""

    model_type = "llama_mlp_bias"


class LlamaMLPWithBiasForCausalLM(LlamaForCausalLM):
    config_class = LlamaMLPBiasConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModelMLPWithBias(config)
        self.post_init()


class Qwen2MLPWithBiasForCausalLM(Qwen2ForCausalLM):
    config_class = Qwen2MLPBiasConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2ModelMLPWithBias(config)
        self.post_init()


def register_custom_models():
    # Register configs
    AutoConfig.register("qwen2_mlp_bias", Qwen2MLPBiasConfig)
    AutoConfig.register("llama_mlp_bias", LlamaMLPBiasConfig)

    # Register models
    AutoModelForCausalLM.register(Qwen2MLPBiasConfig, Qwen2MLPWithBiasForCausalLM)
    AutoModelForCausalLM.register(LlamaMLPBiasConfig, LlamaMLPWithBiasForCausalLM)
