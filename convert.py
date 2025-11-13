import json
import random
from tqdm import tqdm

from prompt import construct_instructions
from utils import gpt_sample_responses


def generate_random_list(length, max_value):
    if length > max_value + 1:
        raise ValueError(
            "Length cannot exceed the number of unique elements in the value range."
        )
    return random.sample(range(0, max_value + 1), length)


def build_prompt(evidence_text, conversion_type, claim_text):
    prompt_template = construct_instructions[conversion_type]
    return prompt_template.format(claim_text=claim_text, evidence_text=evidence_text)


def process_json_lines(input_file, output_file):
    output_data = []
    all_prompts = []
    prompt_metadata = []
    conversion_map = {0: "text", 1: "table", 2: "info_box", 3: "kg_triplets"}

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

        for item_idx, item in enumerate(tqdm(data, desc="Collecting Prompts")):
            evidences = [
                item["default_evidence"],
                item["misinformation_conflict_evidence"],
                item["semantic_conflict_evidence"],
                item["temporal_conflict_evidence"],
            ]

            claims = [
                item["default_claim"],
                item["misinformation_conflict_claim"],
                item["semantic_conflict_claim"],
                item["temporal_conflict_claim"],
            ]

            conversion_indices = generate_random_list(len(evidences), 3)
            item_prompts = []

            for ev_idx, ev in enumerate(evidences):
                conversion_type = conversion_map[conversion_indices[ev_idx]]
                if conversion_type == "text":
                    item_prompts.append(None)
                else:
                    prompt = build_prompt(ev, conversion_type, claims[ev_idx])
                    all_prompts.append(prompt)
                    prompt_metadata.append((item_idx, ev_idx))
                    item_prompts.append("PLACEHOLDER")

            output_data.append(
                {
                    "evidences": evidences,
                    "claims": claims,
                    "conversion_indices": conversion_indices,
                    "prompts": item_prompts,
                }
            )
    gpt_results = gpt_sample_responses(
        all_prompts, temperature=0.7, top_p=0.9, n_samples=1
    )

    for i, (item_idx, ev_idx) in enumerate(prompt_metadata):
        output_data[item_idx]["prompts"][ev_idx] = gpt_results[i][0]

    converted_data = []
    for item in output_data:
        item_data = []
        for idx, ev in enumerate(item["evidences"]):
            conversion_type = conversion_map[item["conversion_indices"][idx]]
            result = ev if conversion_type == "text" else item["prompts"][idx]
            item_data.append(
                {
                    "conversion_type": conversion_type,
                    "converted_result": result,
                    "claim": item["claims"][idx],
                }
            )
        converted_data.append(item_data)

    for idx, item in enumerate(converted_data):
        item.append(data[idx]["question"])

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(converted_data, outfile, ensure_ascii=False, indent=2)


def process_json_lines_homo(input_file, output_file):
    output_data = []
    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)

        for item in tqdm(data):
            evidences = [
                item["default_evidence"],
                item["misinformation_conflict_evidence"],
                item["semantic_conflict_evidence"],
                item["temporal_conflict_evidence"],
            ]

            claims = [
                item["default_claim"],
                item["misinformation_conflict_claim"],
                item["semantic_conflict_claim"],
                item["temporal_conflict_claim"],
            ]
            output_data.append(
                [
                    {
                        "conversion_type": "text",
                        "converted_result": evidence,
                        "claim": claim,
                    }
                    for evidence, claim in zip(evidences, claims)
                ]
            )
            output_data[-1].append(item["question"])

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_json_lines(
        "./data/conflictbank/qa_file.json", "./data/converted/converted_output.json"
    )
    process_json_lines_homo(
        "./data/conflictbank/qa_file.json",
        "./data/converted/converted_output_homo.json",
    )
