from datasets import load_dataset
import tqdm
import json

# Load dataset
dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")

def load_LiveCodeBench_dataset(split = dataset['test']):
    final_dataset = []

    for idx, data in enumerate(tqdm.tqdm(split)):

        record = {}
        #print(data)
        # Prompt: description + starter code
        prompt_parts = [data.get("question_content", "")]
        starter_code = data.get("starter_code")
        if starter_code:
            prompt_parts.append("\n\n" + starter_code)
        record["prompt"] = "".join(prompt_parts).strip()
        record["task_id"] = idx

        # Entry point / function name

        metadata = json.loads(data['metadata'])

        record["entry_point"] = metadata.get("function_name") or 'solution'

        # Reference solution / ground truth code
        record["ground_truth_code"] = metadata.get("reference") or metadata.get("solution") or "unknown"

        # Tests: combine public + private
        public_tests = data.get("public_test_cases", [])
        record["test"] = json.loads(public_tests)

        # Category: difficulty first, then platform
        record["category"] = data.get("difficulty") or data.get("platform") or "unknown"

        final_dataset.append(record)

    return final_dataset