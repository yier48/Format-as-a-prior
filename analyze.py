import json
import numpy as np
from copy import deepcopy
from scipy.stats import binomtest
from collections import defaultdict, Counter

from utils import get_exclude_indices, most_common_element


def load_data(
    model_name, control_group, judge_model=None, data_dir="claim_alignment_results"
):
    if not judge_model:
        file_id = f"{model_name}_{control_group}"
    else:
        file_id = f"{model_name}_{control_group}_judge_{judge_model}"
    main_file = f"./data/preference/{data_dir}/{file_id}.json"
    filter_file = f"./data/evaluation/{model_name}.json"

    try:
        with open(main_file, "r", encoding="utf-8") as f:
            main_data = json.load(f)
        with open(filter_file, "r", encoding="utf-8") as f:
            filter_data = json.load(f)
        return file_id, main_data, filter_data
    except FileNotFoundError:
        print(f"Warning: Data file for {file_id} not found. Skipping.")
        return None, None, None


def analyze_pairwise_scores(main_data, exclude_indices):
    pair_counters = defaultdict(lambda: Counter())
    for idx, item in enumerate(main_data):
        if idx in exclude_indices:
            continue
        try:
            score = int(item.get("judgment", [""])[0])
        except:
            continue
        if score not in [0, 1, 2, 3]:
            continue

        t1, t2 = item["type_1"], item["type_2"]
        key = tuple(sorted([t1, t2]))

        if score == 1:
            pair_counters[key][t1] += 1
        elif score == 3:
            pair_counters[key][t2] += 1
        elif score == 0:
            pair_counters[key]["neutral_none"] += 1
        elif score == 2:
            pair_counters[key]["neutral_both"] += 1
    return pair_counters


def analyze_pairwise_scores_with_classifition(
    main_data, exclude_indices, classification_data
):
    pair_counters = defaultdict(lambda: defaultdict(Counter))
    for idx, item in enumerate(main_data):
        if idx in exclude_indices:
            continue
        try:
            score = int(item.get("judgment", [""])[0])
        except:
            continue
        if score not in [0, 1, 2, 3]:
            continue
        classification = classification_data[idx // 3]
        t1, t2 = item["type_1"], item["type_2"]
        key = tuple(sorted([t1, t2]))

        if score == 1:
            pair_counters[classification][key][t1] += 1
        elif score == 3:
            pair_counters[classification][key][t2] += 1
        elif score == 0:
            pair_counters[classification][key]["neutral_none"] += 1
        elif score == 2:
            pair_counters[classification][key]["neutral_both"] += 1
    return pair_counters


def report_statistics(model_id, pair_counters):
    print("\n=== Full Bias Analysis (Including 0, 1, 2, 3 Scores) ===")
    print(f"{model_id}")

    bias_matrix_row = {}
    both_favored_row = []

    for key, counter in sorted(pair_counters.items()):
        total = sum(counter.values())
        if total < 10:
            print(f"Pair {key}: Not enough data ({total} samples)\n")
            continue

        t1, t2 = key
        a_count = counter.get(t1, 0)
        b_count = counter.get(t2, 0)
        neutral_none = counter.get("neutral_none", 0)
        neutral_both = counter.get("neutral_both", 0)

        if (a_count + b_count) > 0:
            test_result = binomtest(a_count, a_count + b_count, p=0.5)
            p_val = test_result.pvalue
            ci_low, ci_high = test_result.proportion_ci(confidence_level=0.95)
        else:
            p_val, ci_low, ci_high = 1.0, 0.0, 1.0

        conclusion = (
            "No significant bias"
            if p_val >= 0.05
            else f"Bias toward {t1}" if a_count > b_count else f"Bias toward {t2}"
        )

        print(f"Type Pair: {key}")
        print(f"  Total samples: {total}")
        print(f"    {t1} (score=1): {a_count}")
        print(f"    {t2} (score=3): {b_count}")
        print(f"    Neutral (score=0): {neutral_none}")
        print(f"    Both favored (score=2): {neutral_both}")
        print(f"  p-value (1 vs 3): {p_val:.4f}")
        print(f"  95% CI for {t1} preference proportion: ({ci_low:.3f}, {ci_high:.3f})")
        print(f"  → {conclusion}")
        print("-" * 60)
        bias_matrix_row[key] = (a_count, b_count, p_val)
        both_favored_row.append((neutral_both, total))
    return bias_matrix_row, both_favored_row


def calc_non_text_ratio(counter):
    counts = list(counter.values())[0]
    counts = dict(counts)
    text_count = counts.get("text", 0)
    non_text_key = next((k for k in counts if k not in ("text", "neutral_both")), None)
    if not non_text_key:
        return None
    non_text_count = counts.get(non_text_key, 0)
    total = text_count + non_text_count
    return non_text_count / total if total > 0 else None


def compute_a_b(bias, both):
    neither = 1 - np.array(both)
    a = neither * np.array(bias)
    b = neither - a
    return a, b


def sum_nested_results(*data_dicts):
    result = deepcopy(data_dicts[0])

    for d in data_dicts[1:]:
        for model, pairs in d.items():
            for pair_key, values in pairs.items():
                a, b, c = result[model][pair_key]
                x, y, z = values
                result[model][pair_key] = (a + x, b + y, np.float64(c + z))
    return result


def sum_list_results(*data_dicts):
    result = deepcopy(data_dicts[0])

    for d in data_dicts[1:]:
        for model, pair_list in d.items():
            for i, (x, y) in enumerate(pair_list):
                a, b = result[model][i]
                result[model][i] = (a + x, b + y)
    return result


def main():
    MODEL_NAMES = [
        "Llama-3.1-8B-Instruct-FP8",
        "gpt4omini",
        "Qwen3-8B-FP8",
        "Qwen3-14B-FP8",
        "Qwen3-32B-FP8",
        "Qwen3-30B-A3B-FP8",
        "gemma-2-9b-it-FP8",
        "gemma-2-27b-it-FP8",
        "glm-4-9b-chat-hf",
        "Mistral-7B-Instruct-v0.3",
        # "",
        # "Llama-3.1-8B-Instruct",
        # "Qwen3-8B",
    ]

    def stage_1():
        homo_both_data = []
        for model_name in MODEL_NAMES:
            model_id, homo_data, filter_data = load_data(
                model_name, "converted_output_homo"
            )
            exclude_indices = get_exclude_indices(filter_data, homo_data)
            pair_counters = analyze_pairwise_scores(homo_data, exclude_indices)
            bias_row, both_row = report_statistics(model_id, pair_counters)
            homo_both_data.append(both_row)
        CONTORL_GROUPS = ["converted_output"]
        for judge_model in ["qwen3_plus", "glm-4.5-air", "gpt"]:
            for model_name in MODEL_NAMES:
                for group in CONTORL_GROUPS:
                    model_id, main_data, filter_data = load_data(
                        model_name, group, judge_model
                    )
                    if not main_data:
                        continue
                    exclude_indices = get_exclude_indices(filter_data, group)
                    pair_counters = analyze_pairwise_scores(main_data, exclude_indices)
                    bias_row, both_row = report_statistics(model_id, pair_counters)

    def stage_2():
        CONTORL_GROUPS_REASON = [
            [
                "text_vs_corrupt_kg_uncorrupt",
                "text_vs_corrupt_kg_0.45",
                "text_vs_corrupt_kg_0.9",
            ],
            [
                "text_vs_corrupt_infobox_uncorrupt",
                "text_vs_corrupt_infobox_0.45",
                "text_vs_corrupt_infobox_0.9",
            ],
            [
                "text_vs_corrupt_table_uncorrupt",
                "text_vs_corrupt_table_0.45",
                "text_vs_corrupt_table_0.9",
            ],
            [
                "text_vs_kg_nums_4",
                "text_vs_kg_nums_8",
                "text_vs_kg_nums_12",
            ],
            [
                "text_vs_infobox_nums_4",
                "text_vs_infobox_nums_8",
                "text_vs_infobox_nums_12",
            ],
            [
                "text_vs_table_nums_4",
                "text_vs_table_nums_8",
                "text_vs_table_nums_12",
            ],
            [
                "text_vs_type_infobox",
                "text_vs_type_table",
                "text_vs_type_kg",
            ],
        ]
        group_model_counters = [[] for _ in range(len(CONTORL_GROUPS_REASON))]

        for model_name in MODEL_NAMES:
            print(f"Processing model: {model_name}")
            for group_idx, group in enumerate(CONTORL_GROUPS_REASON):
                combined_counters = []
                for file in group:
                    model_id, main_data, filter_data = load_data(model_name, file)
                    if not main_data:
                        continue
                    exclude_indices = get_exclude_indices(filter_data, group)
                    pair_counter = analyze_pairwise_scores(main_data, exclude_indices)
                    report_statistics(model_id, pair_counter)
                    combined_counters.append(pair_counter)
                group_model_counters[group_idx].append(combined_counters)

        def merge_counters(counter_list):
            merged = defaultdict(Counter)
            for c in counter_list:
                for pair, cnt in c.items():
                    merged[pair].update(cnt)
            return merged

        group_model_merged = []
        for group_counters in group_model_counters:
            group_models_merged = []
            for model_counters in group_counters:
                merged = merge_counters(model_counters)
                group_models_merged.append(merged)
            group_model_merged.append(group_models_merged)

        return group_model_merged

    def stage_3():
        REASON_GROUPS = [
            [
                "kg_nums_4_vs_8",
                "kg_nums_8_vs_12",
                "kg_nums_4_vs_12",
            ],
            [
                "infobox_nums_4_vs_8",
                "infobox_nums_8_vs_12",
                "infobox_nums_4_vs_12",
            ],
            [
                "table_nums_4_vs_8",
                "table_nums_8_vs_12",
                "table_nums_4_vs_12",
            ],
            [
                "corrupt_infobox_corrupt_0.45_vs_uncorrupt",
                "corrupt_infobox_corrupt_0.9_vs_uncorrupt",
                "corrupt_infobox_corrupt_0.45_vs_corrupt_0.9",
            ],
            [
                "corrupt_kg_corrupt_0.45_vs_uncorrupt",
                "corrupt_kg_corrupt_0.9_vs_uncorrupt",
                "corrupt_kg_corrupt_0.45_vs_corrupt_0.9",
            ],
            [
                "corrupt_table_corrupt_0.45_vs_uncorrupt",
                "corrupt_table_corrupt_0.9_vs_uncorrupt",
                "corrupt_table_corrupt_0.45_vs_corrupt_0.9",
            ],
            [
                "type_infobox_vs_kg",
                "type_infobox_vs_table",
                "type_kg_vs_table",
            ],
        ]
        group_bias_ratios = defaultdict(lambda: defaultdict(list))
        group_both_ratios = defaultdict(lambda: defaultdict(list))

        for model_name in MODEL_NAMES:
            print(model_name)
            for group_idx, group in enumerate(REASON_GROUPS):
                for file in group:
                    model_id, main_data, filter_data = load_data(model_name, file)
                    if not main_data:
                        continue

                    exclude_indices = get_exclude_indices(filter_data, group)
                    pair_counters = analyze_pairwise_scores(main_data, exclude_indices)
                    bias_row, both_row = report_statistics(model_id, pair_counters)

                    both_row_copy = list(both_row)
                    for i, ((format1, format2), (count1, count2, _)) in enumerate(
                        bias_row.items()
                    ):
                        total = count1 + count2
                        if total > 0:
                            bias_ratio = count1 / total
                            group_bias_ratios[group_idx][(format1, format2)].append(
                                bias_ratio
                            )
                        if i < len(both_row_copy):
                            both_count, total_count = both_row_copy[i]
                            if total_count > 0:
                                both_ratio = both_count / total_count
                                group_both_ratios[group_idx][(format1, format2)].append(
                                    both_ratio
                                )

        for group_idx in sorted(group_bias_ratios.keys()):
            print(f"\n📁 Group {group_idx}:")

            print("Bias ratio:")
            for (format1, format2), ratios in group_bias_ratios[group_idx].items():
                avg_bias = sum(ratios) / len(ratios)
                print(f"  {format1} vs {format2}: {avg_bias:.4f}")

            print("Both ratio:")
            for (format1, format2), ratios in group_both_ratios[group_idx].items():
                avg_both = sum(ratios) / len(ratios)
                print(f"  {format1} vs {format2}: {avg_both:.4f}")

    def analyze_by_classifition():

        CONTORL_GROUPS = ["converted_output"]
        with open("./data/classification_file.json", "r") as f:
            classification_results = [most_common_element(i) for i in json.load(f)]
        category_mapping = {
            "Natural Sciences & Engineering": [
                "physics",
                "optics",
                "astronomy",
                "space",
                "biology",
                "biochemistry",
                "biotechnology",
                "genetics",
                "zoology",
                "ornithology",
                "veterinary",
                "marine science",
                "oceanography",
                "ecology",
                "environmental science",
                "bioinformatics",
                "chemistry",
                "meteorology",
                "geography",
                "nature",
                "environment",
                "engineering",
                "technology",
                "computer science",
                "telecommunications",
                "nuclear_security",
                "transportation",
                "energy",
                "mathematics",
                "statistics",
                "science",
                "environmental",
                "sustainability",
            ],
            "Health & Medicine": [
                "medicine",
                "nursing",
                "public health",
                "health",
                "healthcare",
                "nutrition",
                "food safety",
                "culinary",
                "psychology",
                "cognitive science",
                "biomedical",
            ],
            "Social Sciences & Humanities": [
                "history",
                "literature",
                "language",
                "religion",
                "culture",
                "humanities",
                "sociology",
                "anthropology",
                "archaeology",
                "economics",
                "politics",
                "law",
                "gender studies",
                "civil society",
                "diplomacy",
                "human rights",
                "education",
                "gendergap",
                "personal",
            ],
            "Arts & Entertainment": [
                "art",
                "arts",
                "artistry",
                "design",
                "architecture",
                "music",
                "film",
                "theatre",
                "theater",
                "entertainment",
                "media",
                "gaming",
                "journalism",
                "sports",
            ],
            "Occupations & Industry": [
                "employment",
                "occupation",
                "hospitality",
                "culinary",
                "maritime",
                "transportation",
                "business",
                "finance",
                "agriculture",
                "nonprofit",
                "organization",
                "humanitarian",
                "civil society",
            ],
            "Academia & Institutions": [
                "education",
                "academia",
                "libraries",
                "library",
                "archives",
                "museum",
                "research",
                "honors",
                "awards",
                "fraternity",
                "society",
            ],
            "Government & Military": [
                "government",
                "diplomacy",
                "politics",
                "law",
                "military",
                "nuclear_security",
            ],
            "Unknown / Missing": [None],
        }
        reverse_mapping = {
            sub: main for main, subs in category_mapping.items() for sub in subs
        }

        def map_to_main_categories(subcategories):
            result = []
            for sub in subcategories:
                main = reverse_mapping.get(sub, "Unmapped")
                if main == "Unmapped":
                    print(sub)
                result.append(main)
            return result

        classification_results = map_to_main_categories(classification_results)
        heatmap_data = {category: {} for category in category_mapping.keys()}
        bar_data = {category: {} for category in category_mapping.keys()}
        homo_both_data = []
        for model_name in MODEL_NAMES:
            model_id, homo_data, filter_data = load_data(
                model_name, "converted_output_homo"
            )
            exclude_indices = get_exclude_indices(filter_data, homo_data)
            pair_counters = analyze_pairwise_scores(homo_data, exclude_indices)
            bias_row, both_row = report_statistics(model_id, pair_counters)
            homo_both_data.append(both_row)
            for group in CONTORL_GROUPS:
                model_id, main_data, filter_data = load_data(model_name, group)
                if not main_data:
                    continue
                exclude_indices = get_exclude_indices(filter_data, group)
                pair_counters = analyze_pairwise_scores_with_classifition(
                    main_data, exclude_indices, classification_results
                )
                for item in pair_counters:
                    bias_row, both_row = report_statistics(
                        model_id, pair_counters[item]
                    )
                    heatmap_data[item][model_name] = bias_row
                    bar_data[item][model_name] = both_row


if __name__ == "__main__":
    pass
