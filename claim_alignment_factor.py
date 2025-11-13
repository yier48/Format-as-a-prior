import json
import random
import argparse
import os
from tqdm import tqdm

from prompt import alignment_instruction, answer_instruction_with_reference
from utils import (
    gpt_sample_responses,
    llm_sample_responses,
    qwen3_sample_responses,
    most_common_element,
)


def process_file(data_path, model_type, model_name, enable_thinking):
    reason = (
        os.path.basename(data_path)
        .replace("converted_output_", "")
        .replace(".json", "")
    )
    if model_type == "gpt":
        model_name = "gpt4omini"
    else:
        if model_type == "qwen" and enable_thinking:
            model_name += "_thinking"

    result_file = (
        f"./data/preference/claim_alignment_results/{model_name}_{reason}.json"
    )
    answer_file = f"./data/preference/all_answers/{model_name}_{reason}.json"
    if os.path.exists(result_file):
        print(f"[✓] {result_file}, skip")
        return
    else:
        print(f"[✓] processing: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_inputs = []
    metadata_list = []

    for item_idx, item in enumerate(tqdm(data, desc=f"Preparing inputs ({reason})")):
        question = item[2]

        shared_ref = item[0]["converted_result"]
        claim_shared = item[0]["claim"]
        type_shared = item[0]["conversion_type"]

        specific_ref = item[1]["converted_result"]
        claim_specific = item[1]["claim"]
        type_specific = item[1]["conversion_type"]

        references = [shared_ref, specific_ref]
        random.shuffle(references)
        full_reference = f"\n\n:{references[0]}\n\n:{references[1]}"
        input_text = answer_instruction_with_reference.format(
            full_reference=full_reference, question=question
        )

        all_inputs.append(input_text)
        metadata_list.append(
            {
                "item_idx": item_idx,
                "question": question,
                "claim_shared": claim_shared,
                "type_shared": type_shared,
                "claim_specific": claim_specific,
                "type_specific": type_specific,
            }
        )

    if model_type == "gpt":
        all_answers = gpt_sample_responses(
            all_inputs, temperature=0, top_p=1, n_samples=1
        )
    elif model_type == "qwen":
        all_answers = qwen3_sample_responses(
            model_name,
            all_inputs,
            temperature=0,
            top_p=None,
            top_k=None,
            n_samples=1,
            enable_thinking=enable_thinking,
        )
    else:
        all_answers = llm_sample_responses(
            model_name, all_inputs, temperature=0, top_p=None, n_samples=1
        )

    with open(answer_file, "w", encoding="utf-8") as f:
        json.dump(all_answers, f, ensure_ascii=False, indent=2)

    judgment_prompts = []
    for metadata, answer in zip(metadata_list, all_answers):
        prompt = alignment_instruction.format(
            question=metadata["question"],
            answer=answer,
            claim_shared=metadata["claim_shared"],
            claim_specific=metadata["claim_specific"],
        )
        judgment_prompts.append(prompt)

    judgment_responses = gpt_sample_responses(
        judgment_prompts, temperature=0, top_p=1, n_samples=3
    )
    all_judgments = [most_common_element(j) for j in judgment_responses]

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

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"[✓] finish: {result_file}")


def main():
    parser = argparse.ArgumentParser(description="Claim Alignment Evaluation (Batch)")
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument("--model_name", type=str, help="Model name for llama or qwen")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode (only for Qwen)",
    )
    args = parser.parse_args()
    data_dir = "./data/converted"
    count = 0
    total_count = len(os.listdir(data_dir)) - 1
    for filename in os.listdir(data_dir):
        if (
            filename.endswith(".json")
            and filename != "converted_output.json"
            and filename != "converted_output_homo.json"
        ):
            count += 1
            print(f"{count}/{total_count}")
            file_path = os.path.join(data_dir, filename)
            process_file(
                file_path, args.model_type, args.model_name, args.enable_thinking
            )


if __name__ == "__main__":
    main()
