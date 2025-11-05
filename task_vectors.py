import argparse
import gc
import json
import re
import subprocess
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    GemmaForCausalLM,
)
from models_with_mlp_bias import (
    register_custom_models,
    Qwen2MLPWithBiasForCausalLM,
    LlamaMLPWithBiasForCausalLM,
)


def get_git_hash():
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def create_readme(pretrained_model, ft_model1, ft_model2, ft_model3, git_hash, args):
    scale_1 = "" if args.scale_t1 is None else f"{args.scale_t1} * "
    scale_2 = "" if args.scale_t2 is None else f"{args.scale_t2} * "
    scale_3 = "" if args.scale_t3 is None else f"{args.scale_t3} * "
    if args.finetuned_model3 is None:
        combination = f"{scale_1}t_1 + {scale_2}t_2"
    else:
        combination = f"{scale_1}t_1 + {scale_2}t_2 - {scale_3}t_3"
    """Create README content for the combined task vector model."""
    readme_content = f"""# Combined Task Vector Model

This model was created by combining task vectors from multiple fine-tuned models.

## Task Vector Computation

```python
t_1 = TaskVector("{pretrained_model}", "{ft_model1}")
t_2 = TaskVector("{pretrained_model}", "{ft_model2}")
t_2 = TaskVector("{pretrained_model}", "{ft_model3}")
t_combined = {combination}
new_model = t_combined.apply_to("{pretrained_model}", scaling_coef={args.scaling_coef})
```

Models Used

- Base Model: https://huggingface.co/{pretrained_model}
- Fine-tuned Model 1: https://huggingface.co/{ft_model1}
- Fine-tuned Model 2: https://huggingface.co/{ft_model2}
- Fine-tuned Model 3: https://huggingface.co/{ft_model3}

Technical Details

- Creation Script Git Hash: {git_hash}
- Task Vector Method: Additive combination
- Args: {json.dumps(args.__dict__, indent=2)}
"""
    return readme_content


def get_total_layers(model):
    if type(model) in {
        LlamaForCausalLM,
        Qwen2ForCausalLM,
        GemmaForCausalLM,
        Qwen2MLPWithBiasForCausalLM,
        LlamaMLPWithBiasForCausalLM,
    }:
        return len(model.model.layers)
    raise Exception(f"Model {type(model)} not in the current options.")


def get_layer_number(state_dict_key):
    pattern = r"model\.layers\.(\d+)\."
    match = re.search(pattern, state_dict_key)
    if match:
        return int(match.group(1))
    return None


class TaskVector:
    """
    Code originally taken from: https://github.com/mlfoundations/task_vectors/blob/main/src/task_vectors.py
    """

    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        from_huggingface=True,
        vector=None,
        total_layers=None,
        keys_to_add_as_zero=set(),
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.keys_to_add_as_zero = keys_to_add_as_zero
        if vector is not None:
            self.vector = vector
            self.total_layers = total_layers  # or calculate from vector keys
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                if from_huggingface:
                    if isinstance(pretrained_checkpoint, str):
                        pretrained_checkpoint = AutoModelForCausalLM.from_pretrained(
                            pretrained_checkpoint
                        )
                    pretrained_state_dict = pretrained_checkpoint.state_dict()
                    if isinstance(finetuned_checkpoint, str):
                        finetuned_checkpoint = AutoModelForCausalLM.from_pretrained(
                            finetuned_checkpoint
                        )
                    finetuned_state_dict = finetuned_checkpoint.state_dict()
                else:
                    pretrained_state_dict = torch.load(
                        pretrained_checkpoint
                    ).state_dict()
                    finetuned_state_dict = torch.load(finetuned_checkpoint).state_dict()

                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )
            self.total_layers = get_total_layers(pretrained_checkpoint)

    @classmethod
    def from_two_finetuned_models(
        cls,
        finetuned_checkpoint_1,
        finetuned_checkpoint_2,
        scaling_coef_1=1.0,
        scaling_coef_2=1.0,
        keys_to_add_as_zero=set(),
    ):
        """Create a task vector from the difference between two finetuned models.

        Args:
            finetuned_checkpoint_1: First finetuned model (path or model)
            finetuned_checkpoint_2: Second finetuned model (path or model)
            scaling_coef_1: Scaling factor for first model (default: 1.0)
            scaling_coef_2: Scaling factor for second model (default: 1.0)
            from_huggingface: Whether to load from HuggingFace

        Returns:
            TaskVector: vector = scaling_coef_1 * model_1 - scaling_coef_2 * model_2
        """
        with torch.no_grad():
            finetuned_checkpoint_1 = AutoModelForCausalLM.from_pretrained(
                finetuned_checkpoint_1
            )
            finetuned_state_dict_1 = finetuned_checkpoint_1.state_dict()
            finetuned_checkpoint_2 = AutoModelForCausalLM.from_pretrained(
                finetuned_checkpoint_2
            )
            finetuned_state_dict_2 = finetuned_checkpoint_2.state_dict()

            vector = {}
            for key in finetuned_state_dict_1:
                if finetuned_state_dict_1[key].dtype in [torch.int64, torch.uint8]:
                    print("Ignoring key:", key)
                    continue
                if key not in finetuned_state_dict_2:
                    raise ValueError(
                        f"Key {key} is present in first checkpoint but not in second checkpoint"
                    )
                vector[key] = (
                    scaling_coef_1 * finetuned_state_dict_1[key]
                    - scaling_coef_2 * finetuned_state_dict_2[key]
                )

        # Create instance with the computed vector
        instance = cls(vector=vector, keys_to_add_as_zero=keys_to_add_as_zero)
        instance.total_layers = get_total_layers(finetuned_checkpoint_1)
        return instance

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in list(set(self.vector.keys()).union(other.vector.keys())):
                if key in other.vector and key in self.vector:
                    new_vector[key] = self.vector[key] + other.vector[key]
                elif key in self.keys_to_add_as_zero:
                    new_vector[key] = (
                        self.vector[key] if key in self.vector else other.vector[key]
                    )
                else:
                    raise Exception(
                        f"Warning, key {key} is not present in both task vectors."
                    )
        return TaskVector(vector=new_vector, total_layers=self.total_layers)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector, total_layers=self.total_layers)

    def __mul__(self, scalar):
        """Multiply task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = scalar * self.vector[key]
        return TaskVector(vector=new_vector, total_layers=self.total_layers)

    def __rmul__(self, scalar):
        """Enable right multiplication (scalar * task_vector)."""
        return self.__mul__(scalar)

    def get_module_data(self, module_filter):
        """
        Generator that yields flattened data for a specific module type

        Args:
            module_filter: Function that takes a key and returns True if it belongs to the module
        """
        module_keys = [k for k in self.vector.keys() if module_filter(k)]

        if not module_keys:
            raise Exception(
                "The module filter did not match any keys: {}".format(
                    self.vector.keys()
                )
            )

        # Concatenate and return flattened tensor
        module_vector = torch.cat([self.vector[key].flatten() for key in module_keys])
        return module_vector.cpu().numpy()

    def apply_line_scaling(self, alpha=0.5, beta=0.5):
        """
        Parameters:
        -----------
        task_vector : dict
            A dictionary representing the residual between the fine-tuned checkpoint
            and the pre-trained checkpoint.
        alpha : float
            The minimum scaling factor for the blocks.
        beta : float
            The maximum scaling coefficient difference between the last and first block.
        """
        for k in self.vector:
            layer_number = get_layer_number(k)
            if layer_number is None:
                # embed, unember, norm layers.
                scaling_factor = alpha
            else:
                scaling_factor = alpha + beta * layer_number / self.total_layers
            self.vector[k] *= scaling_factor

    def apply_to(
        self,
        pretrained_checkpoint,
        from_huggingface=True,
        scaling_coef=1.0,
    ):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            if from_huggingface and isinstance(pretrained_checkpoint, str):
                pretrained_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_checkpoint
                )
            else:
                pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    raise Exception(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def apply_to_with_diff_architecture(
        self,
        model_name_architecture,
        model_name_weights,
        scaling_coef=1.0,
    ):
        with torch.no_grad():
            model_architecture = AutoModelForCausalLM.from_pretrained(
                model_name_architecture
            )
            model_weights = AutoModelForCausalLM.from_pretrained(model_name_weights)

            weights_state_dict = model_weights.state_dict()
            architecture_state_dict = model_architecture.state_dict()
            new_state_dict = {}

            for key in architecture_state_dict:
                if key in weights_state_dict:
                    new_state_dict[key] = (
                        weights_state_dict[key] + scaling_coef * self.vector[key]
                    )
                else:
                    new_state_dict[key] = scaling_coef * self.vector[key]

            model_architecture.load_state_dict(new_state_dict, strict=False)

            # Clean up pretrained model from memory
            del model_weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return model_architecture

    def cosine_similarity(self, other):
        """Compute cosine similarity between two task vectors."""
        # Check that all keys match exactly
        if set(self.vector.keys()) != set(other.vector.keys()):
            raise ValueError("Task vectors must have identical parameter keys")

        # Compute dot product and norms efficiently
        dot_product = 0.0
        norm_self_sq = 0.0
        norm_other_sq = 0.0
        skipped_params = []

        for key in self.vector.keys():
            v1 = self.vector[key]
            v2 = other.vector[key]

            # Skip integer parameters (embeddings, position IDs, etc.)
            if v1.dtype in [torch.int64, torch.uint8] or v2.dtype in [
                torch.int64,
                torch.uint8,
            ]:
                skipped_params.append(key)
                continue

            # Element-wise operations are memory efficient
            dot_product += torch.sum(v1 * v2).item()
            norm_self_sq += torch.sum(v1 * v1).item()
            norm_other_sq += torch.sum(v2 * v2).item()

        # Print warning if we skipped parameters
        if skipped_params:
            print(
                f"Warning: Skipped {len(skipped_params)} integer parameters: {skipped_params[:3]}{'...' if len(skipped_params) > 3 else ''}"
            )

        # Compute cosine similarity
        norm_product = (norm_self_sq**0.5) * (norm_other_sq**0.5)
        if norm_product == 0:
            return 0.0  # Handle zero vectors

        return dot_product / norm_product


def maybe_apply_scaling(t, apply_line_scaling, linear_scaling):
    if apply_line_scaling:
        t.apply_line_scaling()
    if linear_scaling is not None:
        return linear_scaling * t
    return t


def main(args):
    register_custom_models()
    print("Creating first task vector...")
    t_1 = TaskVector(args.pretrained_model, args.finetuned_model1)
    t_1 = maybe_apply_scaling(t_1, args.apply_line_scaling_t1, args.scale_t1)
    if args.finetuned_model3 is None:
        print("Creating second task vector...")
        t_2 = TaskVector(args.pretrained_model, args.finetuned_model2)
        t_2 = maybe_apply_scaling(t_2, args.apply_line_scaling_t2, args.scale_t2)
        print("Combining task vectors...")
        t_combined = t_1 + t_2
        del t_1, t_2
    else:
        # If t_2=personality_good and t_3=personality_bad, then
        # -(t_3 - t_2) = - bad_direction = t_2 - t_3
        # If t_2=personality_bad and t_3=personality_good, then
        # -(t_3 - t_2) = t_2 - t_3 = bad_direction
        # t_2 - t_3
        t_diff = TaskVector.from_two_finetuned_models(
            finetuned_checkpoint_1=args.finetuned_model2,
            finetuned_checkpoint_2=args.finetuned_model3,
            scaling_coef_1=args.scale_t2,
            scaling_coef_2=args.scale_t3,
            keys_to_add_as_zero=set(
                [
                    f"model.layers.{i}.mlp.down_proj.bias"
                    for i in range(t_1.total_layers)
                ]
            ),
        )
        t_combined = t_diff + t_1
        del t_1, t_diff

    gc.collect()
    print("🔄 Applying combined task vector to base model...")
    if args.apply_to_diff_model_architecure is None:
        new_model = t_combined.apply_to(
            args.pretrained_model, scaling_coef=args.scaling_coef
        )
    else:
        new_model = t_combined.apply_to_with_diff_architecture(
            model_name_architecture=args.apply_to_diff_model_architecure,
            model_name_weights=args.pretrained_model,
            scaling_coef=args.scaling_coef,
        )
    # Load tokenizer from base model
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    git_hash = get_git_hash()
    readme_content = create_readme(
        args.pretrained_model,
        args.finetuned_model1,
        args.finetuned_model2,
        args.finetuned_model3,
        git_hash,
        args,
    )

    if args.output_dir:
        # Save locally first
        print(f"💾 Saving model locally to {args.output_dir}...")
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        new_model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        with open(output_path / "README.md", "w") as f:
            f.write(readme_content)

        print(f"✅ Model saved locally to {output_path}")

    if args.output_model_name:
        print(f"🚀 Pushing model to Hugging Face Hub: {args.output_model_name}")
        new_model.push_to_hub(
            args.output_model_name,
            commit_message=f"Combined task vectors from {args.finetuned_model1} and {args.finetuned_model2}",
        )
        tokenizer.push_to_hub(args.output_model_name)
        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_content.encode(),
            path_in_repo="README.md",
            repo_id=args.output_model_name,
            repo_type="model",
            commit_message="Add README with task vector combination details",
        )

        print(f"✅ Model successfully uploaded to {args.output_model_name}")
        print(f"🔗 View at: https://huggingface.co/{args.output_model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine task vectors from two fine-tuned models and upload to Hugging Face Hub"
    )

    # Required arguments
    parser.add_argument(
        "--pretrained_model",
        required=True,
        type=str,
        help="Name of the pretrained base model (e.g., 'meta-llama/Llama-3.1-8B')",
    )
    parser.add_argument(
        "--finetuned_model1",
        required=True,
        type=str,
        help="Name of the first fine-tuned model",
    )
    parser.add_argument(
        "--finetuned_model2",
        required=True,
        type=str,
        help="Name of the second fine-tuned model",
    )
    parser.add_argument(
        "--finetuned_model3",
        default=None,
        type=str,
        help="Name of the second fine-tuned model",
    )
    parser.add_argument("--apply_to_diff_model_architecure", default=None, type=str)

    # Output options
    parser.add_argument(
        "--output_model_name",
        type=str,
        help="Name for the new model on Hugging Face Hub (e.g., 'username/combined-model')",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Local directory to save the model (optional)"
    )

    # Task vector options
    parser.add_argument(
        "--scaling_coef",
        default=1.0,
        type=float,
        help="Scaling coefficient for applying the combined task vector (default: 1.0)",
    )
    parser.add_argument("--apply_line_scaling_t1", action="store_true")
    parser.add_argument("--apply_line_scaling_t2", action="store_true")
    parser.add_argument("--apply_line_scaling_t3", action="store_true")
    parser.add_argument("--scale_t1", default=None, type=float)
    parser.add_argument("--scale_t2", default=None, type=float)
    parser.add_argument("--scale_t3", default=None, type=float)
    args = parser.parse_args()

    # Validation
    if not args.output_model_name and not args.output_dir:
        parser.error(
            "Must specify either --output_model_name or --output_dir (or both)"
        )

    main(args)
