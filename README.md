# PET-Select: Prompting Techniques Selection for Code Generation with Contrastive Complexity Prediction

**PET-Select** is a PET-agnostic selection model designed to enhance the accuracy and efficiency of large language models (LLMs) in code generation. While numerous prompt engineering techniques (PETs) exist, no single method works optimally for all queries. PET-Select addresses this challenge by leveraging code complexity as a proxy to classify queries and select the most appropriate PET dynamically.

By incorporating contrastive learning, PET-Select effectively distinguishes between simple and complex programming tasks, ensuring that the best-suited PET is applied. Our evaluations on multiple benchmarks, including HumanEval, HumanEval+, MBPP, MBPP+, and APPS, using GPT-3.5 Turbo, GPT-4o, and DeepSeek-V3, demonstrate up to 1.9% improvement in pass@1 accuracy and a 49.9% reduction in token usage.

This repository provides the implementation of PET-Select.

## Project Structure

```
📂 PET-Select
 ┣ 📂 src                               # Source code
 ┣ 📂 PET_model_dataset                 # Example dataset folder for model
 ┣ 📂 parse_dataset                     # Code for parsing dataset
 ┣ 📜 requirements.txt                  # Dependencies
 ┣ 📜 README.md                         # Project documentation
 ┣ 📜 complexity_model.py               # Code for complexity classification model
 ┣ 📜 generate_dataset.py               # Code for creating Triplet dataset 
 ┣ 📜 contrastive_embedding_model.py    # Code for contrastive embedding model
 ┣ 📜 multilabel_rank_model.py          # Code for multilabel selection model
 ┣ 📜 eval_multilabel_model.py          # Code for multilabel selection model evaluation
 ┣ 📜 eval_main.py                      # Entry code for creating ranked dataset
 ┗ 📜 main.py                           # Entry code for running each technique
```

## Usage

Please ensure that all dependencies have been installed prior to replicating our experiment. You can run the dataset for each prompt engineering technique using:

### **Arguments**  

| Argument           | Type  | Default Value | Description                                                                                                                                                             |
|--------------------|------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--dataset`       | str  | `'APPS'`      | Name of the dataset to use. Options: `APPS`, `LiveCodeBench`, `BigCodeBench`, etc.                                                                                      |
| `--model`         | str  | `'gpt-4o'`    | Name of the LLM model to use. Options: `gpt-4o`, `DeepSeek-V3`, `gemini`, etc.                                                                                          |
| `--temperature`   | int  | `0.0`         | Sampling temperature for the model. Higher values (e.g., `0.7`) produce more randomness in responses.<br/>Note, Gemini has a temperature of 1 for optimal functionality |
| `--append`        | flag | `False`       | If specified, appends results instead of overwriting them.                                                                                                              |
| `--max_tokens`    | int  | `4096`        | Maximum number of tokens in the model's output.                                                                                                                         |
| `--system_message`| str  | `''`          | Optional system message to guide the model's behavior.                                                                                                                  |
| `--start`         | int  | `0`           | Starting index for processing dataset samples. Useful for running a subset of the dataset.                                                                              |
| `--end`           | int  | `0`           | Ending index for processing dataset samples. Use `0` to process all samples.                                                                                            |
| `--technique`     | str  | `'Zeroshot'`  | Prompt engineering technique to use. Options: `Zeroshot`, `Fewshot`, `CoT`, etc.                                                                                        |


Example:
```bash
python main.py --dataset APPS --model gpt-4o --technique Zeroshot
```


After having the result of each PET, get the ranked dataset by using:
```bash
python eval_main.py 
```
Notes: Remember to change the value of dataset and model arguments in eval_main file.


Training complexity classification model by using:
```bash
python complexity_model.py
```

Creating Triplet dataset for training contrastive model by using:
```bash
python generate_dataset.py
```

Training contrastive model by using:
```bash
python contrastive_embedding_model.py
```

Training selection model by using:
```bash
python multilabel_rank_model.py
```

Evaluating performance for selection model by using:
```bash
python eval_multilabel_model.py
```

Or you can run all the scripts by using bash script:
```bash
bash grid_search_complexity.sh
```
Notes: Remember to change all the configurations in the bash script.


---
