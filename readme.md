### 🧠 Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data 

Codebase for the AAAI 2026 paper *"Format as a Prior: Quantifying and Analyzing Bias in LLMs for Heterogeneous Data"*. 

--- 

### 🔑 API Key 
Add your API key to [`secret_config.py`](./secret_config.py) before running any scripts. 

--- 

### 📂 Data Preparation 
Place all required datasets in the [`data`](./data/) folder. 

- Mandatory: **ConflictBank** 

- Optional: **HotpotQA**, **MuSiQue** 
  

All models are executed using [vLLM](https://github.com/vllm-project/vllm). You can configure the model server port in [`utils.py`](./utils.py). 

Sample questions from the dataset 
```bash 
python sample_question.py 
``` 
Identify the questions the model already knows: 
```bash 
python split_question.py --model_name MODEL_NAME 
``` 
Convert the datasets into the required experimental formats: 
```bash 
python convert.py 
python convert_factor.py 
``` 
--- 
### ⚙️ Model Response Generation 

Generate model outputs with the following commands: 
```bash 
python claim_alignment.py --model_type MODEL_TYPE --model_name MODEL_NAME --judge_model_name JUDGE_MODEL_NAME 
python claim_alignment_homo.py --model_type MODEL_TYPE --model_name MODEL_NAME --judge_model_name JUDGE_MODEL_NAME 
python claim_alignment_factor.py --model_type MODEL_TYPE --model_name MODEL_NAME 
``` 
--- 

### 🧩 Intervention Testing 
To evaluate the effect of the proposed intervention: 
```bash 
python hook.py --model_path MODEL_PATH 
``` 
To assess downstream reasoning performance on RAG-style QA tasks: 

```bash 
python hotpot_QA.py 
python musique.py 
``` 

--- 

### 📊 Result Analysis 
Use the helper functions in [`analyze.py`](./analyze.py) to analyze experimental results and visualize performance trends. 