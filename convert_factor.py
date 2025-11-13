import json
import random
import itertools
from tqdm import tqdm

from prompt import kg_nums_instruction, table_nums_instruction, infobox_nums_instruction
from utils import gpt_sample_responses

CONVERT_NUMS = 2000


def build_prompt(evidence_text, claim_text, variant, key):
    if key == "kg_nums":
        kg_nums = variant[key]
        return kg_nums_instruction.format(
            claim_text=claim_text, evidence_text=evidence_text, nums=kg_nums
        )
    elif key == "table_nums":
        table_nums = variant[key]
        return table_nums_instruction.format(
            claim_text=claim_text, evidence_text=evidence_text, nums=table_nums
        )
    elif key == "infobox_nums":
        infobox_nums = variant[key]
        return infobox_nums_instruction.format(
            claim_text=claim_text, evidence_text=evidence_text, nums=infobox_nums
        )
    elif key == "type":
        result_type = variant[key]
        if result_type == "table":
            return table_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        elif result_type == "infobox":
            return infobox_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        elif result_type == "kg":
            return kg_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        else:
            raise ValueError("Invalid result type")
    elif "corrupt" in key:
        if "table" in key:
            return table_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        elif "infobox" in key:
            return infobox_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        elif "kg" in key:
            return kg_nums_instruction.format(
                claim_text=claim_text, evidence_text=evidence_text, nums=4
            )
        else:
            raise ValueError("Invalid key")
    else:
        raise ValueError("Invalid key")


def process_json_lines_homo(input_file, variant_config):
    key = list(variant_config.keys())[0]
    values = variant_config[key]

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)[:CONVERT_NUMS]

    for value1, value2 in itertools.combinations(values, 2):
        paired_output = []
        all_prompts = []
        metadata = []
        types = []

        output_file = f"./data/converted/{key}_{value1}_vs_{value2}.json"

        for item in tqdm(data, desc=f"Processing {value1} vs {value2}"):
            evidence_1 = item["default_evidence"]
            claim_1 = item["default_claim"]
            conflict_type = random.choice(["misinformation", "semantic", "temporal"])
            types.append(conflict_type)
            evidence_2 = item[f"{conflict_type}_conflict_evidence"]
            claim_2 = item[f"{conflict_type}_conflict_claim"]
            if random.random() < 0.5:
                variant_1 = {key: value1}
                variant_2 = {key: value2}
            else:
                variant_1 = {key: value2}
                variant_2 = {key: value1}

            prompt_1 = build_prompt(evidence_1, claim_1, variant_1, key)
            prompt_2 = build_prompt(evidence_2, claim_2, variant_2, key)

            all_prompts.extend([prompt_1, prompt_2])
            metadata.append(
                [
                    {"evidence": evidence_1, "claim": claim_1, **variant_1},
                    {"evidence": evidence_2, "claim": claim_2, **variant_2},
                ]
            )
        gpt_results = gpt_sample_responses(
            all_prompts, temperature=0.7, top_p=0.9, n_samples=1
        )
        for i in range(len(gpt_results) // 2):
            group = []
            for j in range(2):
                meta = metadata[i][j]
                result = {
                    "evidence": meta["evidence"],
                    "claim": meta["claim"],
                    "converted_result": gpt_results[2 * i + j][0],
                    "conversion_type": meta[key],
                }
                group.append(result)
            group.append(data[i]["question"])
            group.append(types[i])
            paired_output.append(group)

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(paired_output, outfile, ensure_ascii=False, indent=2)


def process_json_lines_hete(input_file, variant_config):
    all_prompts = []
    metadata = []
    key = list(variant_config.keys())[0]
    values = variant_config[key]

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)[:CONVERT_NUMS]

    types = []
    for item in tqdm(data, desc="Collecting Prompt Pairs"):
        evidence_1 = item["default_evidence"]
        claim_1 = item["default_claim"]

        conflict_type = random.choice(["misinformation", "semantic", "temporal"])
        types.append(conflict_type)
        evidence_2 = item[f"{conflict_type}_conflict_evidence"]
        claim_2 = item[f"{conflict_type}_conflict_claim"]

        if random.random() < 0.5:
            rewrite_evidence = evidence_1
            rewrite_claim = claim_1
            original_evidence = evidence_2
            original_claim = claim_2
        else:
            rewrite_evidence = evidence_2
            rewrite_claim = claim_2
            original_evidence = evidence_1
            original_claim = claim_1

        meta_group = []
        for val in values:
            variant = {key: val}
            prompt = build_prompt(rewrite_evidence, rewrite_claim, variant, key)
            all_prompts.append(prompt)
            meta_group.append(
                {"evidence": rewrite_evidence, "claim": rewrite_claim, **variant}
            )

        meta_group.append({"evidence": original_evidence, "claim": original_claim})
        metadata.append(meta_group)

    gpt_results = gpt_sample_responses(
        all_prompts, temperature=0.7, top_p=0.9, n_samples=1
    )

    for idx, value in enumerate(values):
        paired_output = []
        output_file = f"./data/converted/text_vs_{key}_{value}.json"

        for i in range(len(metadata)):
            group = []

            group.append(
                {
                    "evidence": metadata[i][-1]["evidence"],
                    "claim": metadata[i][-1]["claim"],
                    "converted_result": metadata[i][-1]["evidence"],
                    "conversion_type": "text",
                }
            )

            meta = metadata[i][idx]
            gpt_result = gpt_results[i * len(values) + idx][0]
            group.append(
                {
                    "evidence": meta["evidence"],
                    "claim": meta["claim"],
                    "converted_result": gpt_result,
                    "conversion_type": meta[key],
                }
            )

            group.append(data[i].get("question", ""))
            group.append(types[i])
            paired_output.append(group)

        with open(output_file, "w", encoding="utf-8") as outfile:
            json.dump(paired_output, outfile, ensure_ascii=False, indent=2)


def process_json_lines(input_file, variant_config):
    all_prompts = []
    metadata = []

    key = list(variant_config.keys())[0]
    value = variant_config[key]

    with open(input_file, "r", encoding="utf-8") as infile:
        data = json.load(infile)[:CONVERT_NUMS]
    types = []
    for item in tqdm(data, desc="Collecting Prompt Pairs"):
        evidence_1 = item["default_evidence"]
        claim_1 = item["default_claim"]
        conflict_type = random.choice(["misinformation", "semantic", "temporal"])
        types.append(conflict_type)
        evidence_2 = item[f"{conflict_type}_conflict_evidence"]
        claim_2 = item[f"{conflict_type}_conflict_claim"]

        if random.random() < 0.5:
            rewrite_evidence = evidence_1
            rewrite_claim = claim_1
            original_evidence = evidence_2
            original_claim = claim_2
        else:
            rewrite_evidence = evidence_2
            rewrite_claim = claim_2
            original_evidence = evidence_1
            original_claim = claim_1
        variant = {key: value}
        prompt = build_prompt(rewrite_evidence, rewrite_claim, None, key)
        all_prompts.extend([prompt])
        metadata.append(
            [
                {"evidence": rewrite_evidence, "claim": rewrite_claim, **variant},
                {"evidence": original_evidence, "claim": original_claim},
            ]
        )
    gpt_results = gpt_sample_responses(
        all_prompts, temperature=0.7, top_p=0.9, n_samples=1
    )

    paired_output = []
    output_file = f"./data/converted/text_vs_{key}_{value}.json"
    for i in range(0, len(gpt_results)):
        group = []
        group.append(
            {
                "evidence": metadata[i][1]["evidence"],
                "claim": metadata[i][1]["claim"],
                "converted_result": metadata[i][1]["evidence"],
                "conversion_type": "text",
            }
        )
        meta = metadata[i][0]
        gpt_result = gpt_results[i][0]
        group.append(
            {
                "evidence": meta["evidence"],
                "claim": meta["claim"],
                "converted_result": gpt_result,
                "conversion_type": meta[key],
            }
        )
        group.append(data[i].get("question", ""))
        group.append(types[i])
        paired_output.append(group)

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(paired_output, outfile, ensure_ascii=False, indent=2)


def corrupt_table_format(original, noise_prob):
    replacements = {
        "|": ["｜", ":", "/", " ", "", "¦"],
        "!": ["！", ":", "|", " ", "", "‖"],
        "{": [
            "｛",
            "[",
            "<",
            "(",
            " ",
            "",
        ],
        "}": [
            "｝",
            "]",
            ">",
            ")",
            " ",
            "",
        ],
        "-": [
            "_",
            "~",
            "=",
            "−",
            " ",
            "",
        ],
    }

    def corrupt_char(c):
        if c in replacements and random.random() < noise_prob:
            return random.choice(replacements[c])
        return c

    corrupted = "".join(corrupt_char(c) for c in original)
    return corrupted


def corrupt_infobox_format(original, noise_prob):
    replacements = {
        "|": ["｜", "¦", "/", " ", "", ":"],
        "=": ["≡", ":", "-", " ", "", "＝"],
        "{": ["｛", "<", "(", " ", ""],
        "}": ["｝", ">", ")", " ", ""],
    }

    def corrupt_char(c):
        if c in replacements and random.random() < noise_prob:
            return random.choice(replacements[c])
        return c

    corrupted = "".join(corrupt_char(c) for c in original)
    return corrupted


def corrupt_kg_format(original, noise_prob):
    replacements = {
        "(": ["（", "[", "<", " ", ""],
        ")": ["）", "]", ">", " ", ""],
    }

    def corrupt_char(c):
        if c in replacements and random.random() < noise_prob:
            return random.choice(replacements[c])
        return c

    corrupted = "".join(corrupt_char(c) for c in original)
    return corrupted


def corrupt_format(original, noise_prob, data_type):
    if data_type == "kg":
        return corrupt_kg_format(original, noise_prob)
    elif data_type == "table":
        return corrupt_table_format(original, noise_prob)
    elif data_type == "infobox":
        return corrupt_infobox_format(original, noise_prob)
    else:
        raise ValueError("Invalid data type")


if __name__ == "__main__":
    # ============ NUMS ============
    variant_conf = {"kg_nums": ["4", "8", "12"]}
    process_json_lines_homo("./data/conflictbank/qa_file.json", variant_conf)
    process_json_lines_hete("./data/conflictbank/qa_file.json", variant_conf)

    variant_conf = {"table_nums": ["4", "8", "12"]}
    process_json_lines_homo("./data/conflictbank/qa_file.json", variant_conf)
    process_json_lines_hete("./data/conflictbank/qa_file.json", variant_conf)

    variant_conf = {"infobox_nums": ["4", "8", "12"]}
    process_json_lines_homo("./data/conflictbank/qa_file.json", variant_conf)
    process_json_lines_hete("./data/conflictbank/qa_file.json", variant_conf)

    # ============ TYPE ============
    variant_conf = {"type": ["infobox", "kg", "table"]}
    process_json_lines_homo("./data/conflictbank/qa_file.json", variant_conf)
    process_json_lines_hete("./data/conflictbank/qa_file.json", variant_conf)

    #  ============ CORRUPT ============
    for data_type in ["infobox", "kg", "table"]:
        process_json_lines_homo(
            "./data/conflictbank/qa_file.json",
            {"corrupt_" + data_type: ["uncorrupt", "corrupt"]},
        )
        for corrupt_rate in [0.45, 0.9]:
            with open(
                f"./data/converted/corrupt_{data_type}_corrupt_vs_uncorrupt.json",
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
            for item in data:
                corrupt_index = 0 if item[0]["conversion_type"] == "corrupt" else 1
                item[corrupt_index]["converted_result"] = corrupt_format(
                    item[corrupt_index]["converted_result"], corrupt_rate, data_type
                )
                item[corrupt_index]["conversion_type"] = f"corrupt_{str(corrupt_rate)}"
            with open(
                f"./data/converted/corrupt_{data_type}_corrupt_{corrupt_rate}_vs_uncorrupt.json",
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)

        with open(
            f"./data/converted/corrupt_{data_type}_corrupt_vs_uncorrupt.json",
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
        for item in data:
            corrupt_index = 0 if item[0]["conversion_type"] == "corrupt" else 1
            item[corrupt_index]["converted_result"] = corrupt_format(
                item[corrupt_index]["converted_result"], 0.9, data_type
            )
            item[corrupt_index]["conversion_type"] = "corrupt_0.9"
            item[1 - corrupt_index]["converted_result"] = corrupt_format(
                item[1 - corrupt_index]["converted_result"], 0.45, data_type
            )
            item[1 - corrupt_index]["conversion_type"] = "corrupt_0.45"
        with open(
            f"./data/converted/corrupt_{data_type}_corrupt_0.45_vs_corrupt_0.9.json",
            "w",
            encoding="utf-8",
        ) as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)

    for data_type in ["infobox", "kg", "table"]:
        process_json_lines(
            "./data/conflictbank/qa_file.json", {"corrupt_" + data_type: "uncorrupt"}
        )
        for corrupt_rate in [0.45, 0.9]:
            with open(
                f"./data/converted/text_vs_corrupt_{data_type}_uncorrupt.json",
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
            for item in data:
                corrupt_index = 0 if item[0]["conversion_type"] == "uncorrupt" else 1
                item[corrupt_index]["converted_result"] = corrupt_format(
                    item[corrupt_index]["converted_result"], corrupt_rate, data_type
                )
                item[corrupt_index]["conversion_type"] = f"corrupt_{str(corrupt_rate)}"
            with open(
                f"./data/converted/text_vs_corrupt_{data_type}_{corrupt_rate}.json",
                "w",
                encoding="utf-8",
            ) as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
