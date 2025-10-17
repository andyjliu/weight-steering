"""
python em_eval.py \
--train_model --run_merge --delete_existing_repo \
--axolotl_config  \
--model_dir  \
--models  \
--dataset em_claude_good_financial:test --dataset em_eval_first_plot:test \
--use_claude_judge --api_key ANTHROPIC_API_KEY_BATCH
"""

import argparse
import re
import subprocess

from huggingface_hub import list_repo_refs
from tqdm import tqdm

import wandb
from inference_and_eval import (
    axolotl_merge_and_upload,
    get_inference_cmd,
    get_dataset_to_splits,
)

SMALL_DS_NO_BATCH = {"em_eval_first_plot"}


def get_revisions_ordered(repo_id, include_others=False):
    """
    Get all revisions from a HuggingFace model repo, ordered by checkpoint number
    """

    # Get all branches/revisions in the repo
    refs = list_repo_refs(repo_id)

    # Extract branch names (revisions)
    revisions = [branch.name for branch in refs.branches]

    # Separate checkpoint revisions from other revisions (like 'main')
    checkpoint_revisions = []
    other_revisions = []

    for revision in revisions:
        match = re.match(r"checkpoint-(\d+)", revision)
        if match:
            checkpoint_num = int(match.group(1))
            checkpoint_revisions.append((checkpoint_num, revision))
        elif include_others:
            other_revisions.append(revision)

    # Sort checkpoint revisions by number
    checkpoint_revisions.sort(key=lambda x: x[0])
    sorted_checkpoint_names = [rev[1] for rev in checkpoint_revisions]

    # Return other revisions first, then sorted checkpoints
    return other_revisions + sorted_checkpoint_names


def inference_and_em_coherence_evals(
    model, args, dataset_to_splits, model_revision=None
):
    # Step 2: VLLM inference
    inference_folder_name = (
        f"steering-inference/{args.steering_vector_type}/{args.steer_coeff}"
        if args.use_steering_inference
        else "vllm-inference"
    )
    if not args.skip_model_inference:
        for dataset, splits in dataset_to_splits.items():
            repeats = 1 if dataset != "em_eval_first_plot" else 20
            print("Running inference...")
            inference_cmd = get_inference_cmd(
                model=model,
                dataset=dataset,
                splits=splits,
                repeats=repeats,
                model_revision=model_revision,
                args=args,
                use_vllm=not args.use_steering_inference,
            )
            print("Running cmd: ", "\n".join(inference_cmd))
            subprocess.run(inference_cmd, check=True)
            print("Inference completed successfully")

    if args.skip_judge_eval:
        return
    for dataset, splits in dataset_to_splits.items():
        for split in splits:
            model_folder = (
                model
                if args.system_prompt_id is None
                else "{}_{}".format(model, args.system_prompt_id)
            )
            inference_output = f"/workspace/{inference_folder_name}/{args.model_repo}__{model_folder}/cfierro__{dataset}/{split}"
            if model_revision is not None:
                inference_output = f"/workspace/{inference_folder_name}/{args.model_repo}__{model_folder}/{model_revision}/cfierro__{dataset}/{split}"

            if args.use_claude_judge:
                anthropic_key = (
                    "ANTHROPIC_API_KEY" if args.api_key is None else args.api_key
                )
                if dataset in SMALL_DS_NO_BATCH:
                    anthropic_key = "ANTHROPIC_API_KEY_HIGH"
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
                    "gpt-4o",
                    "--anthropic_tag",
                    "OPENAI_API_KEY" if args.api_key is None else args.api_key,
                    "--anthropic_or_openai",
                    "openai",
                ]
            batch = not args.no_batch_api
            if dataset in SMALL_DS_NO_BATCH:
                batch = False
            model_folder = (
                model_folder
                if model_revision is None
                else f"{model_folder}_{model_revision}"
            )
            if args.use_steering_inference:
                model_folder += f"__{args.steering_vector_type}_{args.steer_coeff}"

            api_cmd = [
                "python",
                "api_inference.py",
                "--output_dir",
                f"/workspace/batch-inference/eval_em/{dataset}/{model_folder}",
                "--vllm_inference_output",
                inference_output,
                "--num_repeats",
                "1",
                "--batch",
                f"{batch}",
                "--processing_function",
                "eval_em",
            ] + model_specific_flags
            print("Running EM eval...", dataset)
            print("Running cmd: ", "\n".join(api_cmd))
            subprocess.run(api_cmd, check=True)

            api_cmd = [
                "python",
                "api_inference.py",
                "--output_dir",
                f"/workspace/batch-inference/eval_em_coherence/{dataset}/{model_folder}",
                "--vllm_inference_output",
                inference_output,
                "--num_repeats",
                "1",
                "--processing_function",
                "eval_em_coherence",
                "--batch",
                f"{batch}",
            ] + model_specific_flags
            print("Running EM coherence eval...", dataset)
            print("Running cmd: ", "\n".join(api_cmd))
            subprocess.run(api_cmd, check=True)
            print("API inference completed successfully")


def main(args):
    if args.train_model:
        assert len(args.models) == 1, "Training only supported for 1 model."
        subprocess.run(
            f'CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess {args.axolotl_config}',
            shell=True,
        )
        axolotl_train_cmd = [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.train",
            args.axolotl_config,
        ]
        subprocess.run(axolotl_train_cmd, check=True)
        print("Training finished successfully")

    dataset_to_splits = get_dataset_to_splits(args)
    for model in args.models:
        axolotl_merge_and_upload(model, args)

        if args.eval_every_k_revisions is None:
            inference_and_em_coherence_evals(model, args, dataset_to_splits)
        else:
            revisions = get_revisions_ordered(f"{args.model_repo}/{model}")
            for revision_i in tqdm(
                range(0, len(revisions), args.eval_every_k_revisions), desc="Revisions"
            ):
                inference_and_em_coherence_evals(
                    model, args, dataset_to_splits, model_revision=revisions[revision_i]
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
    parser.add_argument("--system_prompt_id", default=None, type=str)
    parser.add_argument("--model_repo", default="coastalcph")
    parser.add_argument("--dataset", action="append", help="Format: name:split1,split2")
    parser.add_argument(
        "--run_merge", action="store_true", help="Run axolotl merge-lora step"
    )
    parser.add_argument("--skip_model_inference", action="store_true")
    parser.add_argument("--generation_temperature", type=float, default=1.0)
    parser.add_argument("--generation_max_tokens", type=int, default=600)
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

    parser.add_argument("--eval_every_k_revisions", default=None, type=int)

    args = parser.parse_args()

    if args.run_merge and not args.axolotl_config:
        parser.error("--axolotl_config is required when --run_merge is used")
    if args.run_merge and not args.model_dir:
        parser.error("--model_dir is required when --run_merge or --run_upload is used")

    wandb.init(
        project="merge-inference-eval",
        name=" ".join([*args.models, *args.dataset, "em_eval"]),
        config=args,
    )
    main(args)
