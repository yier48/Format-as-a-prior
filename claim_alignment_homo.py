import json
import random
import argparse
from tqdm import tqdm

from prompt import alignment_instruction, answer_instruction_with_reference
from utils import (
    gpt_sample_responses,
    qwen_plus_sample_response,
    glm_sample_responses,
    llm_sample_responses,
    qwen3_sample_responses,
    most_common_element,
)


def main():
    parser = argparse.ArgumentParser(description="Claim Alignment Evaluation")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["gpt", "llama", "qwen", "mistral", "gemma", "glm"],
        help="Model type",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Specific model name (e.g., Llama-3.1-8B-Instruct-FP8)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen models",
    )
    parser.add_argument(
        "--judge_model_name",
        type=str,
        required=True,
        choices=["qwen3_plus", "glm-4.5-air", "gpt"],
        help="Specific model name to judge (e.g., gpt)",
    )
    args = parser.parse_args()

    with open(
        "./data/converted/converted_output_homo.json", "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    all_inputs = []
    metadata_list = []
    for item_idx, item in enumerate(tqdm(data, desc="Preparing inputs")):
        question = item[4]

        shared_ref = item[0]["converted_result"]
        claim_shared = item[0]["claim"]
        type_shared = item[0]["conversion_type"]

        for i in range(1, 4):
            specific_ref = item[i]["converted_result"]
            claim_specific = item[i]["claim"]
            type_specific = item[i]["conversion_type"]

            references = [specific_ref, shared_ref]
            random.shuffle(references)
            full_reference = f"\n\n:{references[0]}\n\n:{references[1]}"

            input_text = answer_instruction_with_reference.format(
                full_reference=full_reference, question=question
            )

            all_inputs.append(input_text)
            metadata_list.append(
                {
                    "item_idx": item_idx,
                    "i": i,
                    "question": question,
                    "claim_shared": claim_shared,
                    "type_shared": type_shared,
                    "claim_specific": claim_specific,
                    "type_specific": type_specific,
                }
            )
    if args.model_type == "gpt":
        model_id = "gpt4omini"
    else:
        model_id = args.model_name
        if args.model_type == "qwen" and args.enable_thinking:
            model_id += "_thinking"

    if args.model_type == "gpt":
        all_answers = gpt_sample_responses(
            all_inputs, temperature=0, top_p=1, n_samples=1
        )
    elif args.model_type == "qwen":
        all_answers = qwen3_sample_responses(
            args.model_name,
            all_inputs,
            temperature=0,
            top_p=None,
            top_k=None,
            n_samples=1,
            enable_thinking=args.enable_thinking,
        )
    else:
        all_answers = llm_sample_responses(
            args.model_name, all_inputs, temperature=0, top_p=None, n_samples=1
        )

    with open(f"all_answers_{model_id}_homo.json", "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)

    with open(
        f"./data/preference/claim_alignment_results/{model_id}_converted_output_homo.json",
        "r",
        encoding="utf-8",
    ) as f:
        all_answers = [item["answer"] for item in json.load(f)]

    judgment_prompts = []
    for metadata, answer in zip(metadata_list, all_answers):
        judgment_prompts.append(
            alignment_instruction.format(
                question=metadata["question"],
                answer=answer,
                claim_shared=metadata["claim_shared"],
                claim_specific=metadata["claim_specific"],
            )
        )
    judge_model = args.judge_model_name
    if judge_model == "qwen3_plus":
        all_judgments = [
            most_common_element(i)
            for i in qwen_plus_sample_response(
                judgment_prompts, temperature=0, top_p=1, n_samples=3
            )
        ]
    elif judge_model == "gpt":
        all_judgments = [
            most_common_element(i)
            for i in gpt_sample_responses(
                judgment_prompts, temperature=0, top_p=1, n_samples=3
            )
        ]
    elif judge_model == "glm-4.5-air":
        all_judgments = [
            most_common_element(i)
            for i in glm_sample_responses(
                judgment_prompts, temperature=0, top_p=1, n_samples=3
            )
        ]

    all_results = []
    for metadata, answer, judgment in zip(metadata_list, all_answers, all_judgments):
        if not judgment:
            judgment_target = "undetermined"
        elif "1" in judgment:
            judgment_target = metadata["type_shared"]
        elif "2" in judgment:
            judgment_target = "both"
        elif "3" in judgment:
            judgment_target = metadata["type_specific"]
        elif "No" in judgment:
            judgment_target = "neither"
        else:
            judgment_target = "undetermined"

        all_results.append(
            {
                "type_1": metadata["type_shared"],
                "type_2": metadata["type_specific"],
                "claim_1": metadata["claim_shared"],
                "claim_2": metadata["claim_specific"],
                "answer": answer,
                "judgment": judgment,
                "judgment_target": judgment_target,
            }
        )

    with open(
        f"./data/preference/claim_alignment_results/{model_id}_converted_output_homo_judge_{judge_model}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
