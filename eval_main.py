import ast
import astor
import json
import signal   
from src.args import get_args
from parse_dataset import parse_MBPP, parse_BigCodeBench
from src import utils
import src
import subprocess
import tqdm
import copy
import math
from parse_dataset import  parse_HumanEval, parse_HumanEval_plus, parse_MBPP, parse_MBPP_plus, parse_APPS, parse_BigCodeBench
import src.evaluation


def rank_techniques(record_dict, token_dict, techniques):
    print(record_dict)
    print(token_dict)
    ranked_dict = {'Zeroshot': 0, 'Zeroshot_CoT': 0, 'Fewshot': 0, 'Fewshot_CoT': 0, 'Persona': 0, 'Self-planning': 0, 'Self-refine': 0, 'Progressive-Hint': 0, 'Self-debug': 0}
    max_token = max(token_dict.values())
    if max_token == 0:
        sorted_list = sorted(ranked_dict.items(), key=lambda item: item[1], reverse=True)
        return sorted_list
    for technique in techniques:

        if token_dict[technique] == 0:
            continue

        score = math.log(max_token) * record_dict[technique] - math.log(token_dict[technique])
        ranked_dict[technique] = score
    sorted_list = sorted(ranked_dict.items(), key=lambda item: item[1], reverse=True)
    return sorted_list


def main(model, dataset):
    techniques = ['Zeroshot', 'Zeroshot_CoT', 'Fewshot', 'Fewshot_CoT', 'Persona', 'Self-planning', 'Self-refine', 'Progressive-Hint', 'Self-debug']
    # techniques = ['Self-debug']


    print(f"Now parsing {dataset} dataset ...")
    if dataset == 'HumanEval':
        data = parse_HumanEval.load_HumanEval_dataset()
    elif dataset == 'HumanEval_plus':
        data = parse_HumanEval_plus.load_HumanEval_plus_dataset()
    elif dataset == 'MBPP':
        data = parse_MBPP.load_MBPP_dataset()
    elif dataset == 'MBPP_plus':
        data = parse_MBPP_plus.load_MBPP_plus_dataset()
    elif dataset == 'APPS':
        data = parse_APPS.load_apps_dataset()
    elif dataset == 'BigCodeBench':
        data = parse_BigCodeBench.load_BigCodeBench_dataset()
    elif dataset == 'LiveCodeBench':
        data = parse_LiveCodeBench.load_LiveCodeBench_dataset()

    for idx, per_data in enumerate(tqdm.tqdm(data)):
        record = copy.copy(per_data)
        record_dict = {'Zeroshot': 0, 'Zeroshot_CoT': 0, 'Fewshot': 0, 'Fewshot_CoT': 0, 'Persona': 0, 'Self-planning': 0, 'Self-refine': 0, 'Progressive-Hint': 0, 'Self-debug': 0}
        token_dict = {'Zeroshot': 0, 'Zeroshot_CoT': 0, 'Fewshot': 0, 'Fewshot_CoT': 0, 'Persona': 0, 'Self-planning': 0, 'Self-refine': 0, 'Progressive-Hint': 0, 'Self-debug': 0}

        for technique in techniques:
            generated_data = list(map(json.loads, open(f'result/model_result/{dataset}_{technique}_{model}.jsonl')))
            print(f"Now evaluating {technique} on {dataset} with {model} ...")
            if 'HumanEval' in dataset:
                prompt = per_data['prompt']
                test = per_data['test']
                entry_point = per_data['entry_point']
                code = generated_data[idx]['response_code']
                total_token = generated_data[idx]['input_token'] + generated_data[idx]['output_token']
                passed = src.evaluation.eval_humaneval(prompt, code, test, entry_point)

            elif 'MBPP' in dataset:
                test_string = per_data['test']
                code = generated_data[idx]['response_code']
                total_token = generated_data[idx]['input_token'] + generated_data[idx]['output_token']
                if 'plus' in dataset:
                    passed = src.evaluation.eval_mbpp(code, test_string, True)
                else:
                    passed = src.evaluation.eval_mbpp(code, test_string, False)

            elif 'APPS' in dataset:
                test = per_data['test']
                code = generated_data[idx]['response_code']
                total_token = generated_data[idx]['input_token'] + generated_data[idx]['output_token']
                passed = src.evaluation.eval_apps(code, test)

            elif 'BigCodeBench' in dataset:
                test_string = per_data['test']
                code = generated_data[idx]['response_code']
                total_token = generated_data[idx]['input_token'] + generated_data[idx]['output_token']
                passed = src.evaluation.eval_bigcodebench(code, test_string)

            elif 'LiveCodeBench' in dataset:
                if len(per_data['test']) == 0 or idx >= len(generated_data):
                    continue


                test = per_data['test'][0]
                code = generated_data[idx]['response_code']
                testtype = test["testtype"]
                total_token = generated_data[idx]['input_token'] + generated_data[idx]['output_token']

                if testtype == 'stdin':
                    passed = src.evaluation.check_stdin(
                        code= '\n'.join(code) if isinstance(code, list) else code,
                        input_data=test['input'],
                        expected_output=test['output'],
                    )
                elif testtype == "functional":
                    passed = src.evaluation.check_functional(
                        code='\n'.join(code) if isinstance(code, list) else code,
                        test_data=test,
                    )
                else:
                    test_code = per_data['test'].replace('candidate', 'solution')
                    passed = src.evaluation.check_livecodebench('\n'.join(code), test_code)

            if passed:
                record_dict[technique] = 1
            token_dict[technique] = total_token

        record['exec_record'] = record_dict
        record['token_record'] = token_dict
        record['ranked_techniques'] = rank_techniques(record_dict, token_dict, techniques)


        with open(f'result/model_result_acc/{dataset}_{model}_3.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')







    # for technique in techniques:
    #     print(f"Now evaluating {technique} on {dataset} with {model} ...")
    #     data = list(map(json.loads, open(f'result/model_result/{dataset}_{technique}_{model}.jsonl')))
    #     total = len(data)
    #     pass_count = 0
    #     for idx, per_data in enumerate(tqdm.tqdm(data)):

    #         if 'HumanEval' in dataset:
    #             prompt = per_data['prompt']
    #             code = per_data['response_code']
    #             test = per_data['test']
    #             entry_point = per_data['entry_point']
    #             passed = src.evaluation.eval_humaneval(prompt, code, test, entry_point)

    #         elif 'MBPP' in dataset:
    #             code = per_data['response_code']
    #             test_string = per_data['test']
    #             if 'plus' in dataset:
    #                 passed = src.evaluation.eval_mbpp(code, test_string, True)
    #             else:
    #                 passed = src.evaluation.eval_mbpp(code, test_string, False)

    #         elif 'APPS' in dataset:
    #             code = per_data['response_code']
    #             test_string = per_data['test']
    #             passed = src.evaluation.eval_apps(code, test_string)

    #         if passed:
    #             pass_count += 1

    #     with open('result.txt', 'a') as f:
    #         f.write(f'The pass@1 accuracy of {technique} on {dataset} with {model} is: {pass_count/total} \n')


import argparse

if __name__ == "__main__":
    main()




