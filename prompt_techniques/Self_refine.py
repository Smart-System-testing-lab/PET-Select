import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class SelfrefineGenerator(BaseGenerator):
    MBPP_prompt = '''
    {prompt}

    The function name and input variables should follow this template: {function_name}.
    '''

    Reflection_prompt = '''
    {prompt}

    Here is a code snippet:
    {code}

    Please review this code and suggest any improvements or identify any issues.
    '''

    HumanEval_Refinement_prompt = '''
    {prompt}

    Here is a code snippet:
    {initial_code}

    Based on the following feedback, refine the code:
    {reflection}
    '''

    MBPP_Refinement_prompt = '''
    {prompt}

    Here is a code snippet:
    {initial_code}

    Based on the following feedback, refine the code:
    {reflection}

    Refined code (The function name and input variables should follow this template: {function_name}):
    '''

    APPS_Refinement_prompt = '''
{prompt}

Here is a code snippet:
{initial_code}

Based on the following feedback, refine the code:
{reflection}
    '''



    def __init__(self, dataset_name, model_name, technique_name, args):
        """
        Initializes the ZeroShotGenerator with dataset, model, and additional arguments.
        """
        super().__init__(dataset_name, model_name, technique_name, args)

    def form_technique_prompt(self, prompt, function_name=None):
        """
        Forms the prompt string depending on the dataset_name. 
        """
        if 'HumanEval' in self.dataset_name:
            return prompt
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_prompt.format(prompt=prompt, function_name=function_name)
        else:
            return prompt

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
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, function_name)}
                ]
            else:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
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

    def generate_result(self, messages, data):
        output_path = f'result/model_result/{self.dataset_name}_{self.technique_name}_{self.model_name}.jsonl'

        def run_func(message, per_data):
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

            reflection_message = [
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': self.Reflection_prompt.format(code='\n'.join(code), prompt=per_data['prompt'])}
            ]

            if 'gemini' in self.model_name:
                response2, input_token, output_token, thought_token = self.run_model(message)
                total_thought_token += thought_token if thought_token is not None else 0
            else:
                response2, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_output_token += output_token
            
            if 'HumanEval' in self.dataset_name:
                refinement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.HumanEval_Refinement_prompt.format(initial_code='\n'.join(code), reflection=response2, prompt=per_data['prompt'])}
                ]
            elif 'MBPP' in self.dataset_name:
                refinement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.MBPP_Refinement_prompt.format(initial_code='\n'.join(code), reflection=response2, function_name=utils.get_function_info(per_data['test_list'][0]), prompt=per_data['prompt'])}
                ]
            else:
                refinement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.APPS_Refinement_prompt.format(initial_code='\n'.join(code), reflection=response2, prompt=per_data['prompt'])}
                ]

            if 'gemini' in self.model_name:
                response3, input_token, output_token, thought_token = self.run_model(message)
                total_thought_token += thought_token if thought_token is not None else 0
            else:
                response3, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_input_token += output_token
            code = utils.process_generation_to_code(response3)
            result['response_code'] = '\n'.join(code)
            result['input_token'] = total_input_token
            result['output_token'] = total_output_token
            if 'gemini' in self.model_name:
                result['thought_token'] = total_thought_token
            return result

        responses = []

        # Run generation concurrently
        with cfuts.ThreadPoolExecutor(max_workers=32) as executor:
            futs = []
            for idx, per_data in enumerate(data):
                futs.append(executor.submit(run_func, messages[idx], per_data))

            for future in tqdm(cfuts.as_completed(futs), total=len(futs)):
                responses.append(future.result())

        # Sort results by task_id if it exists in your dataset
        responses.sort(key=lambda x: int(x['task_id']))

        # Write out to a JSON lines file
        with open(output_path, 'a') as f:
            for res in responses:
                f.write(json.dumps(res) + "\n")
