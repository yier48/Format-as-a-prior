import os
import re
import json
import glob
import types
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from itertools import combinations


from prompt import answer_instruction_with_reference, alignment_instruction
from utils import (
    most_common_element,
    gpt_sample_responses,
    get_exclude_indices,
    load_model_and_tokenizer,
    get_token_spans_by_offset,
)
from hook_utils import generate_output, make_attn_bias_injector_3elem


def capture_all_attention_weights(model, inputs):
    attn_weights = []

    def hook_fn(module, input, output):
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            attn_weights.append(output[1].detach().cpu())

    hooks = []
    for layer in model.model.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(**inputs, use_cache=False, output_attentions=True)

    for h in hooks:
        h.remove()

    return attn_weights


def compute_all_head_attention(attn_weights, question_span, evidence_spans):
    result = {"segment_1": [], "segment_2": []}
    q_start, q_end = question_span

    for layer_attn in attn_weights:  # shape: [1, heads, q_len, k_len]
        layer_attn = layer_attn[0]  # [heads, q_len, k_len]
        for seg, (start, end) in evidence_spans.items():
            head_scores = []
            for head in layer_attn:  # [q_len, k_len]
                avg = head[q_start:q_end, start:end].mean().item()
                head_scores.append(avg)
            result[seg].append(np.array(head_scores))  # shape: (n_heads,)
    return result


def record_attention_by_type(model_path):
    model_id = model_path.split("/")[-1]
    tokenizer, model = load_model_and_tokenizer(model_path)

    with open(f"./data/converted/converted_output.json", "r") as f:
        data = json.load(f)

    attention_by_type = {}
    save_dir = f"./attention_records/{model_id}"
    os.makedirs(save_dir, exist_ok=True)

    counter = 0

    for item in tqdm(data):
        evidence_1 = item[0]["converted_result"]
        conversion_type_1 = item[0]["conversion_type"]
        question = item[4]

        for i in range(1, 4):
            evidence_2 = item[i]["converted_result"]
            conversion_type_2 = item[i]["conversion_type"]

            if random.random() < 0.5:
                evidence_1, evidence_2 = evidence_2, evidence_1
                conversion_type_1, conversion_type_2 = (
                    conversion_type_2,
                    conversion_type_1,
                )

            full_reference = f"\n\n:{evidence_1}\n\n{evidence_2}"
            prompt = answer_instruction_with_reference.format(
                full_reference=full_reference, question=question
            )

            messages = [{"role": "user", "content": prompt}]
            if "Qwen" in model_id:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            spans = get_token_spans_by_offset(
                tokenizer, [evidence_1, evidence_2, question], text
            )
            spans["question"] = spans.pop("segment_3")
            spans["question"] = [spans["question"][1], -1]
            attn_weights = capture_all_attention_weights(model, model_inputs)
            all_head_attn = compute_all_head_attention(
                attn_weights,
                spans["question"],
                {"segment_1": spans["segment_1"], "segment_2": spans["segment_2"]},
            )
            attn_1 = np.stack(all_head_attn["segment_1"])  # [n_layers, n_heads]
            attn_2 = np.stack(all_head_attn["segment_2"])
            attn_total = attn_1 + attn_2 + 1e-8

            normalized_1 = attn_1 / attn_total
            normalized_2 = attn_2 / attn_total

            for norm_attn, conv_type in zip(
                [normalized_1, normalized_2], [conversion_type_1, conversion_type_2]
            ):
                if conv_type not in attention_by_type:
                    attention_by_type[conv_type] = []
                attention_by_type[conv_type].append(norm_attn)

                conv_dir = os.path.join(save_dir, conv_type)
                os.makedirs(conv_dir, exist_ok=True)
                save_path = os.path.join(conv_dir, f"attn_{counter:06d}.npz")
                np.savez_compressed(save_path, attention=norm_attn)

            counter += 1


def plot_paired_attention_diff(indices, type_A, type_B, data_dir="./attention_records"):

    def load_paired_attentions(dir_A, dir_B):
        pattern = r"attn_(\d+)\.npz"

        files_A = {
            re.search(pattern, f).group(1): f
            for f in glob.glob(os.path.join(dir_A, "attn_*.npz"))
            if re.search(pattern, f)
        }
        files_B = {
            re.search(pattern, f).group(1): f
            for f in glob.glob(os.path.join(dir_B, "attn_*.npz"))
            if re.search(pattern, f)
        }

        common_keys = sorted(set(files_A.keys()) & set(files_B.keys()))
        if not common_keys:
            return None, None

        attn_list_A, attn_list_B = [], []
        for key in common_keys:
            if int(key) not in indices:
                attn_A = np.load(files_A[key])["attention"]
                attn_B = np.load(files_B[key])["attention"]
                attn_list_A.append(attn_A)
                attn_list_B.append(attn_B)

        return np.stack(attn_list_A), np.stack(attn_list_B)

    path_A = os.path.join(data_dir, type_A)
    path_B = os.path.join(data_dir, type_B)
    if not os.path.exists(path_A) or not os.path.exists(path_B):
        return

    attn_A, attn_B = load_paired_attentions(path_A, path_B)
    if attn_A is None:
        return

    avg_A = np.mean(attn_A, axis=0)
    avg_B = np.mean(attn_B, axis=0)
    diff = avg_A - avg_B

    n_layers, _ = diff.shape

    print("\n=== Quantitative Paired Attention Difference ===")
    for i in range(n_layers):
        layer_avg = diff[i].mean()
        print(f"Layer {i}: Avg Head Diff = {layer_avg:.4f}")
    total_avg = diff.mean()
    dominant = type_A if total_avg > 0 else type_B
    print(f"\nGlobal Avg Diff ({type_A} - {type_B}): {total_avg:.4f}")
    print(f"=> Overall, '{dominant}' receives more attention on average.\n")


def analyze_all_type_pairs(model_path):
    model_name = model_path.split("/")[-1]
    data_dir = f"./attention_records/{model_name}"
    types = ["info_box", "table", "text", "kg_triplets"]
    pairs = list(combinations(types, 2))

    for type_A, type_B in pairs:
        print(f" - {type_A} vs. {type_B}")
    with open(f"./data/evaluation/{model_name}.json", "r", encoding="utf-8") as f:
        filter_data = json.load(f)

    indices = get_exclude_indices(filter_data, "converted_output_hook")
    for type_A, type_B in pairs:
        print("=" * 80)
        print(f"{type_A} vs. {type_B}")
        try:
            plot_paired_attention_diff(indices, type_A, type_B, data_dir)
        except Exception as e:
            print(f"ERROR {type_A} vs. {type_B} {e}")
        print("=" * 80 + "\n")


def preference_intervention(model_path):
    tokenizer, model = load_model_and_tokenizer(model_path)
    model_id = model_path.split("/")[-1]

    with open(f"./data/converted/converted_output.json", "r") as f:
        data = json.load(f)
    metadata_list = []
    all_answers = []
    for item_idx, item in enumerate(tqdm(data)):
        evidence_shared = item[0]["converted_result"]
        type_shared = item[0]["conversion_type"]
        claim_shared = item[0]["claim"]
        question = item[4]

        for i in range(1, 4):
            evidence_specific = item[i]["converted_result"]
            type_specific = item[i]["conversion_type"]
            claim_specific = item[1]["claim"]

            references = [evidence_shared, evidence_specific]
            random.shuffle(references)

            full_reference = f"\n\n:{references[0]}\n\n:{references[1]}"
            prompt = answer_instruction_with_reference.format(
                full_reference=full_reference, question=question
            )

            messages = [{"role": "user", "content": prompt}]
            if "Qwen" in model_id:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            else:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            spans = get_token_spans_by_offset(
                tokenizer, [evidence_shared, evidence_specific, question], text
            )
            spans["question"] = spans.pop("segment_3")
            if "Qwen3" in model_id:
                number_of_layers = 36
            elif "Llama" in model_id:
                number_of_layers = 32
            elif "Mistral" in model_id:
                number_of_layers = 32
            for layer_idx in range(number_of_layers):
                attn_module = model.model.layers[layer_idx].self_attn
                attn_module.forward = types.MethodType(
                    make_attn_bias_injector_3elem(spans, model_id, alpha=1), attn_module
                )

            all_answers.append(generate_output(model, tokenizer, messages))
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
    all_judgments = [
        most_common_element(i)
        for i in gpt_sample_responses(
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

    with open(f"{model_id}_hook.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="Model Path")
    args = parser.parse_args()

    record_attention_by_type(args.model_path)
    analyze_all_type_pairs(args.model_path)
    preference_intervention(args.model_path)
