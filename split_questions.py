import os
import json
import argparse

from utils import gpt_sample_responses, llm_sample_responses, qwen3_sample_responses
from prompt import answer_instruction_briefly


def is_correct_answer(predicted_answers, correct_label):
    return correct_label in predicted_answers


def run_qa_evaluation(model_name):
    qa_path = "./data/conflictbank/qa_file.json"

    with open(qa_path, "r", encoding="utf-8") as file:
        questions_data = json.load(file)

    prompts = [
        "\n".join([answer_instruction_briefly, "Question:", item["question"]])
        for item in questions_data
    ]
    if "gpt" in model_name:
        all_model_outputs = gpt_sample_responses(
            input_texts=prompts,
            temperature=0.5,
            top_p=1,
            n_samples=16,
        )
    elif "Qwen" in model_name:
        all_model_outputs = qwen3_sample_responses(
            qwen_model_name=model_name,
            input_texts=prompts,
            temperature=0.5,
            top_p=None,
            top_k=40,
            n_samples=16,
            enable_thinking=False,
        )
    else:
        all_model_outputs = llm_sample_responses(
            llm_model_name=model_name,
            input_texts=prompts,
            temperature=0.5,
            top_p=1,
            n_samples=16,
            max_workers=5,
        )
    evaluation_records = []
    for question_item, model_outputs in zip(questions_data, all_model_outputs):
        record = {
            "responses": model_outputs,
            "is_correct": is_correct_answer(model_outputs, question_item["object"]),
        }
        evaluation_records.append(record)
    os.makedirs("./data/evaluation/", exist_ok=True)
    output_filename = f"./data/evaluation/{model_name}.json"

    with open(output_filename, "w", encoding="utf-8") as file:
        json.dump(evaluation_records, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    args = parser.parse_args()
    run_qa_evaluation(args.model_name)
