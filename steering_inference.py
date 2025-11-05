import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from activation_steering import ActivationSteerer, ActivationSteererMultiple
from vllm_inference import get_user_message
from contextlib import nullcontext


def _load_tokenizer(path_or_id: str):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok


def load_model(model_path: str, dtype=torch.bfloat16, revision="main"):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, revision=revision, torch_dtype=dtype, device_map="auto"
    )
    tok = _load_tokenizer(model_path)
    return model, tok


def select_steerer(model, vector, coeff, layer, steering_type):
    # Steering layer0 means steering its output. The vector has the embedding layer outputs in position 0.
    if vector is None:
        return nullcontext()
    if layer - 1 >= 0:
        return ActivationSteerer(
            model,
            vector[layer],
            coeff=coeff,
            layer_idx=layer - 1,
            positions=steering_type,
        )
    num_layers = model.config.num_hidden_layers
    return ActivationSteererMultiple(
        model,
        [
            dict(
                steering_vector=(
                    vector[layer] - vector[layer - 1] if layer > 1 else vector[layer]
                ),
                coeff=coeff,
                layer_idx=layer - 1,
                positions=steering_type,
            )
            for layer in range(1, num_layers)
        ],
    )


def sample_steering(
    model,
    tokenizer,
    conversations,
    vector,
    layer,
    coeff,
    bs=20,
    top_p=1,
    max_tokens=1000,
    temperature=1,
    min_tokens=1,
    steering_type="response",
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts = []
    for messages in conversations:
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with select_steerer(model, vector, coeff, layer, steering_type):
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(output)
    return prompts, outputs


def run_steering_inference_and_save(examples, llm, tokenizer, vector, output_dir, args):
    repeated_examples = [ex for ex in examples for _ in range(args.repeats)]
    all_conversations = [
        [dict(role="user", content=get_user_message(ex))] for ex in repeated_examples
    ]
    prompts, answers = sample_steering(
        llm,
        tokenizer,
        all_conversations,
        vector,
        layer=args.steer_layer,
        coeff=args.steer_coef,
        bs=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        steering_type="response",
    )
    responses = []
    for i in range(len(answers)):
        responses.append(
            {
                "instruction": all_conversations[i][0]["content"],
                "response": answers[i],
                "prompt": prompts[i],
                "example_data": repeated_examples[i],
            }
        )

    output_file = output_dir / "responses.json"
    with open(output_file, "w") as f:
        json.dump(responses, f, indent=2)

    print(f"Saved {len(responses)} responses to {output_file}")


def run_inference(args):
    llm, tokenizer = load_model(args.model_name, revision=args.model_revision)
    vector = torch.load(args.vector_path, weights_only=False)
    dataset = load_dataset(args.dataset_name)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    revision = "" if args.model_revision is None else args.model_revision
    output_dir = (
        Path(args.output_dir)
        / args.model_name.replace("/", "__")
        / revision
        / args.dataset_name.replace("/", "__")
    )
    for split in args.splits:
        split_output_dir = output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        run_steering_inference_and_save(
            examples=dataset[split],
            llm=llm,
            tokenizer=tokenizer,
            vector=vector,
            output_dir=split_output_dir,
            args=args,
        )

    with open(output_dir / "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on HuggingFace dataset using safetytooling"
    )
    parser.add_argument("--dataset_name", required=True, type=str)
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        help="Dataset split to use (e.g., 'train', 'test', 'validation')",
    )
    parser.add_argument(
        "--vector_path",
        required=True,
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to save results",
    )
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--steer_layer", required=True, type=int)
    parser.add_argument("--steer_coef", required=True, type=float)

    # Model configuration
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help="Model name to deploy",
    )
    parser.add_argument("--model_revision", default=None, type=str)

    # Inference configuration
    parser.add_argument(
        "--temperature", default=0.7, type=float, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top_p", default=0.95, type=float, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--max_tokens", default=600, type=int, help="Maximum tokens to generate"
    )
    parser.add_argument("--repeats", default=None, type=int)

    # Options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process (for testing)",
    )

    args = parser.parse_args()
    wandb.init(
        project="steering_inference",
        name=" ".join([args.model_name, args.dataset_name, *args.splits]),
        config=args,
    )
    run_inference(args)


if __name__ == "__main__":
    main()
