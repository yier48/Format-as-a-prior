import os
from tqdm import tqdm
from openai import OpenAI
from zai import ZhipuAiClient
import concurrent.futures
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

from secret_config import openai_api_key, qwen3_api_key, glm_api_key

LOCAL_URL = "http://localhost"
PORT = 8006


def qwen3_sample_responses(
    qwen_model_name,
    input_texts,
    temperature,
    top_p,
    top_k,
    n_samples,
    enable_thinking,
    port=PORT,
    max_workers=5,
):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{LOCAL_URL}:{port}/v1",
    )

    def sample_once(text):
        try:
            response = client.chat.completions.create(
                model=qwen_model_name,
                messages=[{"role": "user", "content": text}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=2048,
                timeout=5,
                extra_body={
                    "top_k": top_k,
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            content = response.choices[0].message.content
            assert content is not None, "No content returned"
            return content
        except Exception as e:
            return f"ERROR CONNECTION {e}"

    def process_text(text):
        return [sample_once(text) for _ in range(n_samples)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_text, input_texts),
                total=len(input_texts),
                desc=f"Sampling {str(n_samples)}x per input",
            )
        )

    return results


def llm_sample_responses(
    llm_model_name, input_texts, temperature, top_p, n_samples, port=PORT, max_workers=5
):
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"{LOCAL_URL}:{port}/v1",
    )

    def sample_once(text):
        try:
            response = client.chat.completions.create(
                model=llm_model_name,
                messages=[{"role": "user", "content": text}],
                temperature=temperature,
                timeout=5,
                top_p=top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR CONNECTION {e}"

    def process_text(text):
        return [sample_once(text) for _ in range(n_samples)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_text, input_texts),
                total=len(input_texts),
                desc=f"Sampling {str(n_samples)}x per input",
            )
        )
    return results


def gpt_sample_responses(input_texts, temperature, top_p, n_samples, max_workers=24):
    client = OpenAI(api_key=openai_api_key)

    def sample_once(text):
        try:

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
                temperature=temperature,
                top_p=top_p,
                timeout=60,
            )
            content = response.choices[0].message.content
            assert content is not None, "No content returned"
            return content
        except Exception as e:
            return f"ERROR CONNECTION {e}"

    def process_text(text):
        return [sample_once(text) for _ in range(n_samples)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_text, input_texts),
                total=len(input_texts),
                desc=f"Sampling {str(n_samples)}x per input",
            )
        )
    return results


def qwen_plus_sample_response(
    input_texts, temperature, top_p, n_samples, max_workers=24
):
    client = OpenAI(
        api_key=qwen3_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    def sample_once(text):
        try:

            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text},
                ],
                temperature=temperature,
                top_p=top_p,
                timeout=5,
            )
            content = response.choices[0].message.content
            assert content is not None, "No content returned"
            return content
        except Exception as e:
            return f"ERROR CONNECTION {e}"

    def process_text(text):
        return [sample_once(text) for _ in range(n_samples)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_text, input_texts),
                total=len(input_texts),
                desc=f"Sampling {str(n_samples)}x per input",
            )
        )
    return results


def glm_sample_responses(input_texts, temperature, top_p, n_samples, max_workers=24):
    client = ZhipuAiClient(api_key=glm_api_key)

    def sample_once(text):
        try:

            response = client.chat.completions.create(
                model="GLM-4.5-Air",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text},
                ],
                thinking={
                    "type": "disabled",
                },
                timeout=5,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            assert content is not None, "No content returned"
            return content
        except Exception as e:
            return f"ERROR CONNECTION {e}"

    def process_text(text):
        return [sample_once(text) for _ in range(n_samples)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_text, input_texts),
                total=len(input_texts),
                desc=f"Sampling {str(n_samples)}x per input",
            )
        )
    return results


def most_common_element(lst):
    count = Counter(lst)
    candidates = {k: v for k, v in count.items() if v >= 2}
    if not candidates:
        return None
    return max(candidates, key=candidates.get)


def get_exclude_indices(filter_data, control_group):
    exclude = set()
    if "converted_output" in control_group or "converted_output_homo" in control_group:
        for i, group in enumerate(filter_data):
            if group.get("is_correct", False):
                base = i * 3
                exclude.update([base, base + 1, base + 2])
    else:
        for i, group in enumerate(filter_data):
            if group.get("is_correct", False):
                exclude.add(i)
    return exclude


def load_model_and_tokenizer(model_name, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def get_token_spans_by_offset(tokenizer, texts, full_text):
    spans = {}
    encoding = tokenizer(
        full_text, return_offsets_mapping=True, add_special_tokens=False
    )
    offset_mapping = encoding["offset_mapping"]

    for i, segment in enumerate(texts):
        start_char = full_text.find(segment)
        if start_char == -1:
            print(f"❌ segment_{i+1}: {repr(segment)}")
            continue
        end_char = start_char + len(segment)

        token_start, token_end = None, None
        for idx, (s, e) in enumerate(offset_mapping):
            if token_start is None and s <= start_char < e:
                token_start = idx
            if token_start is not None and s < end_char <= e:
                token_end = idx + 1
                break
            elif token_start is not None and e >= end_char:
                token_end = idx + 1
                break

        if token_start is not None and token_end is not None:
            spans[f"segment_{i+1}"] = (token_start, token_end)
        else:
            print(f"⚠️ segment_{i+1}")
    return spans
