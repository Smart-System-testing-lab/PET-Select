from parse_dataset import parse_LiveCodeBench, parse_HumanEval, parse_HumanEval_plus, parse_MBPP, parse_MBPP_plus, parse_APPS
from src.args import get_args
from prompt_techniques import Zeroshot, Zeroshot_CoT, Fewshot, Fewshot_CoT, Persona, Self_planning, Self_refine, Progressive_Hint, Self_debug
import json

def main(args):
    dataset_name = args.dataset
    technique_name = args.technique
    model_name = args.model

    if dataset_name == 'HumanEval':
        # prompt; task_id; ground_truth code; category; test
        dataset = parse_HumanEval.load_HumanEval_dataset()
        original_data = list(map(json.loads, open('dataset/HumanEval_category.jsonl')))
    if dataset_name == 'HumanEval_plus':
        # prompt; task_id; ground_truth code; category; test
        dataset = parse_HumanEval_plus.load_HumanEval_plus_dataset()
        original_data = list(map(json.loads, open('dataset/HumanEval_plus.jsonl')))
    if dataset_name == 'MBPP':
        # prompt; task_id; ground_truth code; category; test
        dataset = parse_MBPP.load_MBPP_dataset()
        original_data = list(map(json.loads, open('dataset/MBPP_category.jsonl')))
    if dataset_name == 'MBPP_plus':
        # prompt; task_id; ground_truth code; category; test
        dataset = parse_MBPP_plus.load_MBPP_plus_dataset()
        original_data = list(map(json.loads, open('dataset/MBPP_plus.jsonl')))
    if dataset_name == 'APPS':
        dataset = parse_APPS.load_apps_dataset()
        original_data = list(map(json.loads, open('dataset/APPS.jsonl')))
    if dataset_name == 'LiveCodeBench':
        dataset = parse_LiveCodeBench.load_LiveCodeBench_dataset()
        original_data = list(map(json.loads, open('dataset/APPS.jsonl')))

    if technique_name == 'Zeroshot':
        technique_generator = Zeroshot.ZeroshotGenerator(dataset_name, model_name, technique_name, args) 
    elif technique_name == 'Zeroshot_CoT':
        technique_generator = Zeroshot_CoT.ZeroshotCoTGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Fewshot':
        technique_generator = Fewshot.FewshotGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Fewshot_CoT':
        technique_generator = Fewshot_CoT.FewshotCoTGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Persona':
        technique_generator = Persona.PersonaGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Self-planning':
        technique_generator = Self_planning.SelfplanGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Self-refine':
        technique_generator = Self_refine.SelfrefineGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Progressive-Hint':
        technique_generator = Progressive_Hint.ProgressiveHintGenerator(dataset_name, model_name, technique_name, args)
    elif technique_name == 'Self-debug':
        technique_generator = Self_debug.SelfdebugGenerator(dataset_name, model_name, technique_name, args)
    else:
        raise Exception("Incorrect technique. Select a valid option!")
        
    start = 0 if args.start == 0 else args.start
    end = len(dataset) if args.end == 0 else args.end
    data_selected = dataset[start:end]
    original_data_selected = original_data[start:end]
    print(f"Generating messages for {dataset_name} in {technique_name}...")
    messages = technique_generator.generate_prompt(data_selected)
    print(f"Running {model_name} with {technique_name} on {dataset_name}...")
    if technique_name != 'Self-debug':
        technique_generator.generate_result(messages, data_selected)
    else:
        technique_generator.generate_result(messages, data_selected, original_data_selected)

    




if __name__ == "__main__":
    args = get_args()
    main(args)

