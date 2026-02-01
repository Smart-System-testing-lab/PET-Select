import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model
from src import evaluation

from prompt_techniques.Techniques import BaseGenerator

class SelfdebugGenerator(BaseGenerator):
    HumanEval_SelfDebug_init_prompt = '''
    Complete the following task in Python:
    {prompt}

    Your code should pass the test:
    {test}
    '''

    MBPP_SelfDebug_init_prompt = '''
    {prompt}

    Your code should pass the test: 
    {test}

    The function name and input variables should follow this template: {function_name}.
    '''

    SelfDebug_success_prompt = '''
    {code}
    Is the code above correct? If not, please fix it.
    '''

    SelfDebug_failed_prompt = '''
    {code}
    The code above is wrong. Please fix it.
    '''

    APPS_SelfDebug_init_prompt = '''
    Complete the following task in Python:
    {prompt}

    Your code should pass the test:
    {test}
    '''

    LiveCodeBench_SelfDebug_init_prompt = '''
      Complete the following task in Python:
      {prompt}

      Your code should pass the test input/output and type. If type is functional, ensure the function is named {function_name}:
      {test}
      '''

    def __init__(self, dataset_name, model_name, technique_name, args):
        """
        Initializes the ZeroShotGenerator with dataset, model, and additional arguments.
        """
        super().__init__(dataset_name, model_name, technique_name, args)

    def form_technique_prompt(self, prompt, test, function_name=None):
        """
        Forms the prompt string depending on the dataset_name. 
        """
        if 'HumanEval' in self.dataset_name:
            return self.HumanEval_SelfDebug_init_prompt.format(prompt=prompt, test=utils.get_first_elements_of_inputs_and_results(test))
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_SelfDebug_init_prompt.format(prompt=prompt, function_name=function_name, test=test)
        elif 'Live' in self.dataset_name:
            return self.LiveCodeBench_SelfDebug_init_prompt.format(prompt=prompt, test=test, function_name=function_name)
        else:
            return self.APPS_SelfDebug_init_prompt.format(prompt=prompt, test=test)

    def generate_prompt(self, dataset):
        """
        Generates the list of messages for each data item in the dataset.
        """
        messages = []
        for per_data in dataset:
            # Check dataset type
            if 'HumanEval' in self.dataset_name:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test'])}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test_list'][0], function_name)}
                ]
                # quit()
            elif 'Live' in self.dataset_name:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test'], per_data['entry_point'])}
                ]
            else:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, per_data['test'].split('\n')[3])}
                ]

            messages.append(message)
        return messages

    def run_model(self, message):
        if 'gpt' in self.model_name:
            return model.call_chat_gpt(message, self.args)
        elif 'gemini' in self.model_name:
            return model.call_gemini(message, self.args)
        else:
            return model.query_firework(message, self.args, self.model_name)

    def generate_result(self, messages, data, original_data):
        output_path = f'result/model_result/{self.dataset_name}_{self.technique_name}_{self.model_name}.jsonl'

        def run_func(message, per_data, per_original_data):
            tried = 0
            total_input_token, total_thought_token, total_output_token = 0, 0, 0
            result = copy.copy(per_data)

            if 'gemini' in self.model_name:
                response1, input_token, output_token, thought_token = self.run_model(message)
                total_thought_token += thought_token if thought_token is not None else 0
            else:
                response1, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_output_token += output_token
            code = utils.process_generation_to_code(response1)

            while(tried < 3):
                if 'HumanEval' in self.dataset_name:
                    one_assert = utils.extract_one_assert(per_original_data['test'])
                    passed = evaluation.check_code(per_data['prompt'], '\n'.join(code), f'def check(candidate):\n    {one_assert}\n', per_original_data['entry_point'])
                elif 'MBPP' in self.dataset_name:
                    one_assert = per_data['test_list'][0]
                    passed = evaluation.MBPP_check_code('\n'.join(code), per_data['test_list'])
                elif 'Live' in self.dataset_name:
                    if per_data['test'] is not []:
                        test_data = per_data['test'][0]
                        testtype = test_data["testtype"]
                        print(test_data)
                    else:
                        testtype = ''
                        test_data = {}

                    if testtype == 'stdin':
                        passed = evaluation.check_stdin(
                            code='\n'.join(code),
                            input_data=test_data['input'],
                            expected_output=test_data['output'],
                        )
                    elif testtype == "functional":
                        passed = evaluation.check_functional(
                            code='\n'.join(code),
                            test_data=test_data,
                        )
                    else:
                        test_code = per_data['test'].replace('candidate', 'solution')
                        passed = evaluation.check_livecodebench('\n'.join(code), test_code)
                else:
                    one_assert = per_data['test'].split('\n')[3].strip().replace('candidate', 'solution')
                    passed = evaluation.check_apps('\n'.join(code), one_assert)
                if passed:
                    debug_message = [
                        {'role': 'system', 'content': self.system_message},
                        {'role': 'user', 'content': self.SelfDebug_success_prompt.format(code='\n'.join(code))}
                    ]
                    if 'gemini' in self.model_name:
                        response2, input_token, output_token, thought_token = self.run_model(debug_message)
                        total_thought_token += thought_token if thought_token is not None else 0
                    else:
                        response2, input_token, output_token = self.run_model(debug_message)
                    total_input_token += input_token
                    total_output_token += output_token
                    code = utils.process_generation_to_code(response2)
                    break
                else:
                    debug_message = [
                        {'role': 'system', 'content': self.system_message},
                        {'role': 'user', 'content': self.SelfDebug_failed_prompt.format(code='\n'.join(code))}
                    ]
                    if 'gemini' in self.model_name:
                        response2, input_token, output_token, thought_token = self.run_model(debug_message)
                        total_thought_token += thought_token if thought_token is not None else 0
                    else:
                        response2, input_token, output_token = self.run_model(debug_message)
                    total_input_token += input_token
                    total_output_token += output_token
                    code = utils.process_generation_to_code(response2)
                    tried += 1

            result['response_code'] = '\n'.join(code)
            result['input_token'] = total_input_token
            result['output_token'] = total_output_token

            if 'gemini' in self.model_name:
                result['thought_token'] = total_thought_token

            print('success!')
            return result

        responses = []

        # Run generation concurrently
        with cfuts.ThreadPoolExecutor(max_workers=10) as executor:
            futs = {executor.submit(run_func, messages[idx], per_data, original_data[idx]): idx
                    for idx, per_data in enumerate(data)}

            for future in tqdm(cfuts.as_completed(futs), total=len(futs)):
                idx = futs[future]
                try:
                    result = future.result(timeout=300)
                    with open(output_path, "a") as f:
                        f.write(json.dumps(result) + "\n")
                except Exception as e:
                    print(f"[ERROR] idx={idx} run_func failed: {e}")
