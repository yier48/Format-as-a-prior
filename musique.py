import json
import types
import random
import pandas as pd
from tqdm import tqdm

from prompt import (
    construct_instructions_without_claim,
    QA_instruction,
    QA_evaluate_instruction,
)
from utils import (
    gpt_sample_responses,
    qwen3_sample_responses,
    llm_sample_responses,
    most_common_element,
    load_model_and_tokenizer,
    get_token_spans_by_offset,
)

from hook_utils import make_attn_bias_injector_multi, generate_output


def random_sample():
    with open("data/musique/musique_ans_v1.0_dev.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    random_data = random.sample(data, 200)
    with open("data/musique/musique_ans_v1.0_dev_200.json", "w", encoding="utf-8") as f:
        json.dump(random_data, f, indent=2)


def convert_data():
    with open("data/musique/musique_ans_v1.0_dev_200.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    conversion_map = {0: "text", 1: "table", 2: "info_box", 3: "kg_triplets"}
    converted_data = []
    all_prompts = []
    prompt_meta = []
    for idx, item in enumerate(data):
        converted_data.append(item)
        converted_context = []

        for paragraph in item["paragraphs"]:
            if not paragraph["is_supporting"]:
                continue

            conversion_type = conversion_map[random.randint(0, 3)]
            if conversion_type == "text":
                converted_context.append(
                    [paragraph["title"], "text", paragraph["paragraph_text"]]
                )
            else:
                prompt = construct_instructions_without_claim[conversion_type].format(
                    evidence_text=f"{paragraph['title']}\n\n"
                    + paragraph["paragraph_text"],
                )
                all_prompts.append(prompt)
                prompt_meta.append((idx, paragraph["title"], conversion_type))
        converted_data[-1]["context"] = converted_context
    if all_prompts:
        all_responses = gpt_sample_responses(
            all_prompts, temperature=0.7, top_p=0.9, n_samples=1
        )
        for (item_idx, ctx_title, conv_type), resp in zip(prompt_meta, all_responses):
            converted_data[item_idx]["context"].append([ctx_title, conv_type, resp[0]])
    with open(
        "data/musique/musique_ans_v1.0_dev_200_converted.json", "w", encoding="utf-8"
    ) as f:
        json.dump(converted_data, f, indent=2)


def run_qa_evaluation_homo(model_name):
    with open("data/musique/musique_ans_v1.0_dev_200.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [
        QA_instruction.format(
            input_data=str(
                {
                    "question": item["question"],
                    "context": [i for i in item["paragraphs"] if i["is_supporting"]],
                }
            )
        )
        for item in data
    ]

    if "Qwen" in model_name:
        llm_answers = qwen3_sample_responses(
            model_name,
            prompts,
            temperature=0,
            top_p=None,
            top_k=None,
            n_samples=1,
            enable_thinking=False,
            max_workers=24,
        )
    else:
        llm_answers = llm_sample_responses(
            model_name,
            prompts,
            temperature=0,
            top_p=None,
            n_samples=1,
            max_workers=24,
        )
    with open(
        f"data/musique/musique_ans_v1.0_dev_200_{model_name}_homo_useful_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(llm_answers, f, indent=2)


def run_qa_evaluation_hete(model_name):
    with open(
        "data/musique/musique_ans_v1.0_dev_200_converted.json", "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
    prompts = [
        QA_instruction.format(
            input_data=str(
                {
                    "question": item["question"],
                    "context": item["context"],
                }
            )
        )
        for item in data
    ]
    if "Qwen" in model_name:
        llm_answers = qwen3_sample_responses(
            model_name,
            prompts,
            temperature=0,
            top_p=None,
            top_k=None,
            n_samples=1,
            enable_thinking=False,
            max_workers=24,
        )
    else:
        llm_answers = llm_sample_responses(
            model_name,
            prompts,
            temperature=0,
            top_p=None,
            n_samples=1,
            max_workers=24,
        )
    with open(
        f"data/musique/musique_ans_v1.0_dev_200_{model_name}_hete_useful_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(llm_answers, f, indent=2)


def run_judge(model_name):
    with open(
        f"data/musique/musique_ans_v1.0_dev_200_{model_name}_hete_useful_info.json",
        "r",
        encoding="utf-8",
    ) as f:
        hete_answers = json.load(f)
    with open(
        f"data/musique/musique_ans_v1.0_dev_200_{model_name}_homo_useful_info.json",
        "r",
        encoding="utf-8",
    ) as f:
        homo_answers = json.load(f)
    with open(
        f"data/musique/musique_ans_v1.0_dev_200.json",
        "r",
        encoding="utf-8",
    ) as f:
        hotpot_data = json.load(f)
        questions = [item["question"] for item in hotpot_data]
        gold_answers = [item["answer"] for item in hotpot_data]
    judgment_prompts = [
        QA_evaluate_instruction.format(
            question=question, gold_answer=gold_answer, llm_answer=llm_answer
        )
        for question, gold_answer, llm_answer in zip(
            questions, gold_answers, hete_answers
        )
    ] + [
        QA_evaluate_instruction.format(
            question=question, gold_answer=gold_answer, llm_answer=llm_answer
        )
        for question, gold_answer, llm_answer in zip(
            questions, gold_answers, homo_answers
        )
    ]
    all_judgments = [
        most_common_element(i)
        for i in gpt_sample_responses(
            judgment_prompts, temperature=0, top_p=1, n_samples=3
        )
    ]
    with open(f"data/musique/judge_{model_name}_useful_info.json", "w") as f:
        json.dump(all_judgments, f, indent=2)

    right = 0
    wrong = 0
    for judgment in all_judgments[:200]:
        if judgment == "correct":
            right += 1
        elif judgment == "incorrect":
            wrong += 1
    print(f"{right} correct, {wrong} incorrect")
    right = 0
    wrong = 0
    for judgment in all_judgments[200:]:
        if judgment == "correct":
            right += 1
        elif judgment == "incorrect":
            wrong += 1
    print(f"{right} correct, {wrong} incorrect")


def run_judge_inject(model_name):
    with open(
        f"data/musique/musique_ans_v1.0_dev_200_{model_name}_useful_info_inject.json",
        "r",
        encoding="utf-8",
    ) as f:
        hete_answers = json.load(f)
    with open(
        f"data/musique/musique_ans_v1.0_dev_200.json", "r", encoding="utf-8"
    ) as f:
        hotpot_data = json.load(f)
        questions = [item["question"] for item in hotpot_data]
        gold_answers = [item["answer"] for item in hotpot_data]
    judgment_prompts = [
        QA_evaluate_instruction.format(
            question=question, gold_answer=gold_answer, llm_answer=llm_answer
        )
        for question, gold_answer, llm_answer in zip(
            questions, gold_answers, hete_answers
        )
    ]
    all_judgments = [
        most_common_element(i)
        for i in gpt_sample_responses(
            judgment_prompts, temperature=0, top_p=1, n_samples=3
        )
    ]
    with open(
        f"data/musique/judge_{model_name}_useful_info_inject.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_judgments, f, indent=2)

    right = 0
    wrong = 0
    for judgment in all_judgments:
        if judgment == "correct":
            right += 1
        elif judgment == "incorrect":
            wrong += 1
    print(f"{right} correct, {wrong} incorrect")


def run_hook(model_path):
    tokenizer, model = load_model_and_tokenizer(model_path)
    model_id = model_path.split("/")[-1]

    with open(
        f"data/musique/musique_ans_v1.0_dev_200_converted.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)
    all_answers = []
    for item in tqdm(data):
        contexts = [context[2] for context in item["context"]]
        question = item["question"]
        prompt = QA_instruction.format(
            input_data=f'{{"question": "{item["question"]}", "context": {"[" + ", ".join(contexts) + "]"}}}'
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

        spans = get_token_spans_by_offset(tokenizer, contexts + [question], text)
        spans["question"] = spans.pop(f"segment_{len(spans)}")
        if "Qwen3" in model_id:
            number_of_layers = 36
        elif "Llama" in model_id:
            number_of_layers = 32
        elif "Mistral" in model_id:
            number_of_layers = 32
        for layer_idx in range(number_of_layers):
            attn_module = model.model.layers[layer_idx].self_attn
            attn_module.forward = types.MethodType(
                make_attn_bias_injector_multi(spans, model_id, alpha=1), attn_module
            )

        all_answers.append(generate_output(model, tokenizer, messages))
    with open(
        f"data/hotpot_QA/hotpot_dev_distractor_v1_200_{model_id}_useful_info_inject.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(all_answers, f, indent=2)


if __name__ == "__main__":
    random_sample()
    convert_data()

    MODEL_NAMES = [
        "Qwen3-8B",
        "Llama-3.1-8B-Instruct",
        "Mistral-7B-Instruct-v0.3",
    ]
    MODEL_PATHS = [
        "../model/Qwen3-8B",
        "../model/Llama-3.1-8B-Instruct",
        "../model/Mistral-7B-Instruct-v0.3",
    ]
    for model_name in MODEL_NAMES:
        run_qa_evaluation_homo(model_name)
        run_qa_evaluation_hete(model_name)
        run_judge(model_name)
    for model_path in MODEL_PATHS:
        run_hook(model_path)

    for model_name in MODEL_NAMES:
        run_judge_inject(model_name)
