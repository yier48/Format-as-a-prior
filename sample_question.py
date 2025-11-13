import os
import json
import random

SAMPLE_NUMS = 4000


def sample_question_dataset():
    input_path = "../data/CB_qa_v2/QA_dataset.json"
    output_path = "./data/conflictbank/qa_file.json"

    with open(input_path, "r", encoding="utf-8") as file:
        all_questions = [json.loads(line) for line in file]

    sampled_questions = random.sample(all_questions, SAMPLE_NUMS)
    os.makedirs("./data/conflictbank", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(sampled_questions, file, ensure_ascii=False, indent=2)


if __name__ == "main":
    sample_question_dataset()
