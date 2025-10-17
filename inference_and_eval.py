"""
python inference_and_eval.py \
--train --run_merge --delete_existing_repo \
--axolotl_config <path_to_config> \
--model_dir <path_to_dir> \
--models <model_name> \
--dataset <dataset_name>:<split_1>,<split_2> \
--eval_function <eval_name> --use_claude_judge --api_key ANTHROPIC_API_KEY_BATCH
"""

import argparse
import concurrent.futures
import os
import subprocess
from pathlib import Path

import wandb
from huggingface_hub import (
    create_branch,
    create_repo,
    delete_repo,
    list_repo_refs,
    upload_folder,
)

ACT_STEERING = {
    "Qwen2.5-7B": {
        "sycophancy": (
            20,
            "/workspace/persona_vectors_qwen2.7-7b/sycophantic_response_avg_diff.pt",
        ),
        "evil": (
            20,
            "/workspace/persona_vectors_qwen2.7-7b/evil_response_avg_diff.pt",
        ),
        "hallucination": (
            16,
            "/workspace/persona_vectors_qwen2.7-7b/hallucinating_response_avg_diff.pt",
        ),
    },
    "Qwen2.5-1.5B": {
        "sycophancy": (
            17,
            "/workspace/persona_vectors/vectors/persona_vectors/Qwen2.5-1.5B-Instruct/sycophantic_response_avg_diff.pt",
        ),
    },
    "Llama-2-7b-chat": {
        "evil": (
            18,
            "/workspace/persona_vectors/vectors/Llama-2-7b-chat-hf/evil_response_avg_diff.pt",
        ),
        "refusal": (
            None,
            "/workspace/persona_vectors/avg_act_vectors/Llama-2-7b-chat-hf/cfierro__alignment-faking-harm_Llama-2-7b-chat_response_avg.pt",
        ),
    },
}


def get_revisions(model, args):
    try:
        refs = list_repo_refs(f"{args.model_repo}/{model}")
        existing_revisions = [branch.name for branch in refs.branches]
        print(f"Existing revisions: {existing_revisions}")
        if len(existing_revisions) > 0:
            existing_revisions = [r for r in existing_revisions if r != "main"]
    except Exception as e:
        print(f"Could not fetch existing revisions: {e}")
        existing_revisions = []
    return existing_revisions


def axolotl_merge_and_upload(model, args):
    if not args.run_merge:
        return

    model_dir = Path(args.model_dir)
    merged_path = model_dir / "merged"

    # Create the repo once
    print("Setting up HuggingFace repository...")
    if args.delete_existing_repo:
        delete_repo(f"{args.model_repo}/{model}", missing_ok=True)
    create_repo(
        repo_id=f"{args.model_repo}/{model}",
        repo_type="model",
        private=False,
        exist_ok=True,
    )

    # Get existing revisions to check against
    existing_revisions = get_revisions(model, args)

    # Determine which checkpoints to process
    if args.push_all_ckpts:
        # Find all checkpoint directories
        checkpoint_dirs = [d for d in model_dir.glob("checkpoint-*") if d.is_dir()]
        if not checkpoint_dirs:
            raise Exception("No checkpoint directories found")
        print(f"Found {len(checkpoint_dirs)} checkpoints to process")
    else:
        checkpoint_dirs = [model_dir]

    for i, ckpt_dir in enumerate(checkpoint_dirs):
        if len(checkpoint_dirs) > 1:
            print(
                f"\nProcessing checkpoint {i+1}/{len(checkpoint_dirs)}: {ckpt_dir.name}"
            )
            revision = ckpt_dir.name  # e.g., "checkpoint-1000"
        else:
            revision = "main"

        # Check if revision already exists
        if revision in existing_revisions and revision != "main":
            print(f"Revision '{revision}' already exists, skipping...")
            continue

        print("Running axolotl merge-lora...")
        merge_cmd = [
            "axolotl",
            "merge-lora",
            args.axolotl_config,
            f"--lora-model-dir={ckpt_dir}",
        ]
        subprocess.run(merge_cmd, check=True)
        print("Axolotl merge completed successfully")

        # Create branch if it's not main
        if revision != "main":
            print(f"Creating branch '{revision}'...")
            try:
                create_branch(
                    repo_id=f"{args.model_repo}/{model}",
                    branch=revision,
                    repo_type="model",
                )
            except Exception as e:
                print(f"Branch creation failed or already exists: {e}")

        # Upload to HuggingFace with revision
        print(f"Uploading to HuggingFace revision '{revision}'...")
        upload_folder(
            folder_path=str(merged_path),
            repo_id=f"{args.model_repo}/{model}",
            repo_type="model",
            revision=revision,
            create_pr=False,
        )
        print(f"Upload completed successfully for revision '{revision}'")


def get_inference_folder(args):
    if not args.use_steering_inference:
        if args.add_generation_params_to_folder:
            return f"vllm-inference/temp={args.generation_temperature}_tok={args.generation_max_tokens}"
        return "vllm-inference"
    inference_folder_name = (
        f"steering-inference/{args.steering_vector_type}/{args.steer_coeff}"
    )
    inference_folder_name = Path(inference_folder_name)
    extra = (
        f"temp={args.generation_temperature}_tok={args.generation_max_tokens}"
        if args.add_generation_params_to_folder
        else ""
    )
    if args.use_steering_layer is None:
        return str(inference_folder_name / extra)
    return str(inference_folder_name / extra / f"layer{args.use_steering_layer}")


def get_output_dir(dataset, split, model_folder, args):
    if args.add_generation_params_to_folder:
        return "temp={}_tok={}/{}/{}/{}/{}".format(
            args.generation_temperature,
            args.generation_max_tokens,
            args.eval_function,
            dataset,
            split,
            model_folder,
        )
    return f"{args.eval_function}/{dataset}/{split}/{model_folder}"


def get_inference_cmd(
    model, dataset, splits, repeats, model_revision, args, use_vllm=True
):
    cmd = []
    assert dataset != "mmlu_chat" or (
        args.generation_temperature == 0.0 and args.generation_max_tokens == 3000
    )
    inference_folder_name = get_inference_folder(args)
    if use_vllm:
        cmd = [
            "python",
            "vllm_inference.py",
            "--num_threads",
            "32",
            "--max_num_seqs",
            "8",
            "--hf_cache_dir",
            "/workspace/hf_home",
            "--gpu_devices",
            *args.vllm_inference_gpu_devices,
        ]
        if args.system_prompt_id is not None:
            cmd += ["--system_prompt_id", args.system_prompt_id]
        if "cfierro" == args.model_repo:
            cmd += ["--hf_offline"]
    else:
        act_steering = [
            ACT_STEERING[v] for v in ACT_STEERING.keys() if model.startswith(v)
        ]
        assert len(act_steering) == 1
        layer, vector_path = act_steering[0][args.steering_vector_type]
        if args.use_steering_layer is not None:
            layer = args.use_steering_layer
        cmd = [
            "python",
            "steering_inference.py",
            "--vector_path",
            f"{vector_path}",
            "--batch_size",
            f"{args.steering_bs}",
            "--steer_layer",
            f"{layer}",
            "--steer_coef",
            f"{args.steer_coeff}",
        ]
    cmd += [
        "--output_dir",
        f"/workspace/{inference_folder_name}",
        "--dataset_name",
        f"cfierro/{dataset}",
        "--splits",
        *splits,
        "--model_name",
        f"{args.model_repo}/{model}",
        "--repeats",
        f"{repeats}",
        "--temperature",
        f"{args.generation_temperature}",
        "--top_p",
        "1.0",
        "--max_tokens",
        f"{args.generation_max_tokens}",
    ]
    if model_revision is not None:
        cmd += ["--model_revision", model_revision]
    return cmd


def run_inference(model, args, dataset_to_splits, revision=None):
    for dataset, splits in dataset_to_splits.items():
        print("Running inference...")
        inference_cmd = get_inference_cmd(
            model=model,
            dataset=dataset,
            splits=splits,
            repeats=args.num_repeats_inference,
            model_revision=revision,
            args=args,
            use_vllm=not args.use_steering_inference,
        )
        print("Running cmd: ", "\n".join(inference_cmd))
        subprocess.run(inference_cmd, check=True)
        print("Inference completed successfully")


def run_api_inference_for_model(model, args, dataset_to_splits, model_revision=None):
    """Run API inference for a single model"""
    inference_folder_name = get_inference_folder(args)
    if args.use_claude_judge:
        anthropic_key = "ANTHROPIC_API_KEY" if args.api_key is None else args.api_key
        model_specific_flags = [
            "--anthropic_num_threads",
            "5",
            "--anthropic_tag",
            anthropic_key,
        ]
    else:
        model_specific_flags = [
            "--openai_num_threads",
            "5",
            "--model_id",
            args.gpt_judge,
            "--anthropic_tag",
            "OPENAI_API_KEY" if args.api_key is None else args.api_key,
            "--anthropic_or_openai",
            "openai",
        ]

    batch = not args.no_batch_api

    for dataset, splits in dataset_to_splits.items():
        model_folder = model
        if args.system_prompt_id:
            model_folder = "{}_{}".format(model_folder, args.system_prompt_id)
        if model_revision is not None:
            model_folder = "{}_{}".format(model_folder, model_revision)
        vllm_folder = Path(
            f"/workspace/{inference_folder_name}/{args.model_repo}__{model_folder}/cfierro__{dataset}/"
        )
        if args.use_steering_inference:
            model_folder += f"__{args.steering_vector_type}_{args.steer_coeff}"
        if args.use_steering_layer is not None:
            model_folder += f"_layer{args.use_steering_layer}"

        for split in splits:
            print(f"Running Eval for model {model}...")
            api_cmd = [
                "python",
                "api_inference.py",
                "--output_dir",
                f"/workspace/batch-inference/{get_output_dir(dataset, split, model_folder, args)}",
                "--vllm_inference_output",
                f"{vllm_folder / split}",
                "--num_repeats",
                "1",
                "--batch",
                f"{batch}",
                "--processing_function",
                args.eval_function,
                "--temperature",
                f"{args.api_temperature}",
            ] + model_specific_flags
            if args.get_api_logprobs is not None:
                api_cmd += ["--logprobs", f"{args.get_api_logprobs}"]
            print(f"Running cmd for model {model}: ", "\n".join(api_cmd))
            subprocess.run(api_cmd, check=True)

    print(f"API inference completed successfully for model {model}")
    return f"Model {model} completed"


def run_parallel_api_inference(models, args, dataset_to_splits, max_workers=3):
    """Run API inference in parallel for multiple models"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {}
        for model in models:
            revisions = (
                [args.model_revision]
                if args.model_revision is not None
                else get_revisions(model, args)
            )
            revisions = revisions if len(revisions) > 0 else [None]
            for revision in revisions:
                ex = executor.submit(
                    run_api_inference_for_model,
                    model,
                    args,
                    dataset_to_splits,
                    revision,
                )
                future_to_model[ex] = (model, revision)

        # Wait for completion and handle results/errors
        for future in concurrent.futures.as_completed(future_to_model):
            model, revision = future_to_model[future]
            try:
                result = future.result()
                print(f"✓ {result}")
            except Exception as exc:
                print(f"✗ Model {model} ({revision}) generated an exception: {exc}")


def get_dataset_to_splits(args):
    dataset_to_splits = {}
    for item in args.dataset:
        name, splits = item.split(":")
        dataset_to_splits[name] = splits.split(",")
    return dataset_to_splits


def main(args):
    if args.train_model:
        assert len(args.models) == 1, "Training only supported for 1 model."
        subprocess.run(
            f'CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess {args.axolotl_config}',
            shell=True,
        )
        train_env = os.environ.copy()
        train_env.update(
            {
                "NCCL_P2P_DISABLE": "1",
                "NCCL_IB_DISABLE": "1",
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(args.num_gpus)),
                # 'NCCL_DEBUG': 'INFO',  # Uncomment for debugging
                # 'TORCH_DISTRIBUTED_DEBUG': 'INFO',  # Uncomment for debugging
            }
        )

        axolotl_train_cmd = [
            "accelerate",
            "launch",
            f"--num_processes={args.num_gpus}",
            "--num_machines=1",
            "--machine_rank=0",
            "--main_process_port=29500",
            "-m",
            "axolotl.cli.train",
            args.axolotl_config,
        ]

        print(f"Running training command: {' '.join(axolotl_train_cmd)}")
        subprocess.run(axolotl_train_cmd, env=train_env, check=True)
        print("Training finished successfully")

    for model in args.models:
        axolotl_merge_and_upload(model, args)
        if args.skip_model_inference:
            break
        revisions = (
            [args.model_revision]
            if args.model_revision is not None
            else get_revisions(model, args)
        )
        revisions = revisions if len(revisions) > 0 else [None]
        for revision in revisions:
            run_inference(model, args, get_dataset_to_splits(args), revision=revision)

    if not args.skip_judge_eval:
        run_parallel_api_inference(
            models=args.models,
            args=args,
            dataset_to_splits=get_dataset_to_splits(args),
        )

    print("All steps completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run axolotl merge, upload, and inference pipeline"
    )
    parser.add_argument(
        "--axolotl_config",
        help="Path to axolotl config file (required if --run_merge is used)",
    )
    parser.add_argument(
        "--model_dir",
        help="Directory containing the model (required if --run_merge is used)",
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="Model name for HuggingFace"
    )
    parser.add_argument("--model_repo", default="coastalcph")
    parser.add_argument("--system_prompt_id", default=None, type=str)
    parser.add_argument(
        "--dataset", action="append", default=[], help="Format: name:split1,split2"
    )
    parser.add_argument(
        "--run_merge", action="store_true", help="Run axolotl merge-lora step"
    )
    parser.add_argument("--skip_model_inference", action="store_true")
    parser.add_argument("--add_generation_params_to_folder", action="store_true")
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_max_tokens", type=int, default=600)
    parser.add_argument("--api_temperature", type=float, default=1.0)
    parser.add_argument("--get_api_logprobs", type=int, default=None)
    parser.add_argument("--skip_judge_eval", action="store_true")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--use_claude_judge", action="store_true")
    parser.add_argument("--delete_existing_repo", action="store_true")
    parser.add_argument("--push_all_ckpts", action="store_true")
    parser.add_argument("--train_model", action="store_true")
    parser.add_argument("--no_batch_api", action="store_true")
    parser.add_argument("--use_steering_inference", action="store_true")
    parser.add_argument("--steering_vector_type", type=str, default=None)
    parser.add_argument("--steer_coeff", type=float, default=None)
    parser.add_argument("--steering_bs", type=int, default=None)
    parser.add_argument("--use_steering_layer", type=int, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_repeats_inference", type=int, default=1)
    parser.add_argument("--vllm_inference_gpu_devices", default=["0"], nargs="+")
    parser.add_argument("--model_revision", default=None, type=str)
    parser.add_argument("--gpt_judge", default="gpt-4o", type=str)

    parser.add_argument("--eval_function", help="Evaluation function name")

    args = parser.parse_args()

    if args.run_merge and not args.axolotl_config:
        parser.error("--axolotl_config is required when --run_merge is used")
    if args.run_merge and not args.model_dir:
        parser.error("--model_dir is required when --run_merge or --run_upload is used")

    wandb.init(
        project="merge-inference-eval",
        name=" ".join([*args.models, *args.dataset, str(args.eval_function)]),
        config=args,
    )
    main(args)
