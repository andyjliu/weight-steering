from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    Qwen2Config,
)

from models_with_mlp_bias import (
    LlamaMLPBiasConfig,
    LlamaMLPWithBiasForCausalLM,
    Qwen2MLPBiasConfig,
    Qwen2MLPWithBiasForCausalLM,
)


class MLPBiasPlugin(BasePlugin):
    """
    Plugin to patch AutoModelForCausalLM.from_pretrained to use bias-enabled models.
    """

    def __init__(self):
        super().__init__()
        self._original_from_pretrained = None

    def pre_model_load(self, cfg: DictDefault):
        """
        Patch AutoModelForCausalLM.from_pretrained before model loading.
        """
        print("=" * 80)
        print("Patching AutoModelForCausalLM.from_pretrained for MLP bias...")
        print("=" * 80)

        # Store original - get the actual function, not the bound method
        self._original_from_pretrained = AutoModelForCausalLM.from_pretrained.__func__

        @classmethod
        def patched_from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, **kwargs
        ):
            # Get the config
            config = kwargs.get("config")

            if config is None:
                # Load config if not provided
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=kwargs.get("trust_remote_code", False),
                )

            # Check model type and use our custom class if applicable
            if isinstance(config, Qwen2Config):
                print("✓ Detected Qwen2 model, using Qwen2MLPWithBiasForCausalLM")

                # Update model_type using the config class
                config.model_type = Qwen2MLPBiasConfig.model_type

                # Update config in kwargs
                kwargs["config"] = config

                # Bypass AutoModel and use our class directly
                return Qwen2MLPWithBiasForCausalLM.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs
                )
            elif isinstance(config, LlamaConfig):
                print("✓ Detected Llama model, using LlamaMLPWithBiasForCausalLM")

                # Update model_type using the config class
                config.model_type = LlamaMLPBiasConfig.model_type

                # Update config in kwargs
                kwargs["config"] = config

                return LlamaMLPWithBiasForCausalLM.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs
                )
            else:
                raise Exception("Model not supported.")

        # Apply patch - this modifies the class globally
        AutoModelForCausalLM.from_pretrained = patched_from_pretrained

        print("✓ AutoModelForCausalLM.from_pretrained patched globally")
        print("=" * 80)

    def post_model_load(self, cfg: DictDefault, model):
        """
        Verify the model was loaded with bias and correct config.
        """
        print("=" * 80)
        print("Verifying model with MLP bias...")
        print("=" * 80)

        print(f"Model class: {model.__class__.__name__}")
        print(f"Model config type: {model.config.model_type}")

        # Check if bias exists
        bias = model.model.layers[0].mlp.down_proj.bias
        if bias is None:
            raise Exception("⚠ Model does not have bias in down_proj!")

        # Count total bias parameters
        total_bias = sum(
            layer.mlp.down_proj.bias.numel() for layer in model.model.layers
        )

        print(f"✓ Number of layers: {len(model.model.layers)}")
        print(f"✓ Layer 0 down_proj.bias shape: {bias.shape}")
        print(f"✓ Layer 0 down_proj.bias mean: {bias.mean().item():.6f}")
        print(f"✓ Total bias parameters: {total_bias:,}")
        print("=" * 80)

    def post_train_unload(self, cfg: DictDefault):
        """
        Restore original from_pretrained.
        """
        if self._original_from_pretrained:

            @classmethod
            def restored(cls, *args, **kwargs):
                return self._original_from_pretrained(cls, *args, **kwargs)

            AutoModelForCausalLM.from_pretrained = restored
            print("✓ Restored original AutoModelForCausalLM.from_pretrained")
