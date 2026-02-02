import copy
import concurrent.futures as cfuts
import json
from types import NoneType

from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class SelfplanGenerator(BaseGenerator):
    HumanEval_plan_prompt = '''
    Intent: Write a function to find the similar elements from the given two tuple lists.

    Plan: Let's think step by step.
    1. Define a function that takes two lists of tuples as arguments.
    2. Create an empty list to store the similar elements found in both tuple lists.
    3. Use a loop to iterate through each element (tuple) in the first list.
    4. For each tuple in the first list, check if it also exists in the second list.
    5. If the tuple from the first list is found in the second list, append it to the result list.
    6. After iterating through the first list, return the list containing the similar elements.

    Intent: Write a python function to identify non-prime numbers.

    Plan: Let's think step by step.
    1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
    2. Before proceeding to the prime check, consider edge cases.
    3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
    4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
    5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
    6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
    7. If no divisors are found, the function should return False (indicating the number is prime).

    Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

    Plan: Let's think step by step.
    1. Import the Required Module.
    2. Define a function that accepts two arguments.
    3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
    4. Use the heapq.nlargest Function.
    5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

    How about this intent: {prompt}.

    Plan: Let's think step by step.
    '''

    HumanEval_implementation_prompt = '''
    {prompt}
    Please complete the task with the following steps in Python.

    {plan}
    '''

    MBPP_plan_prompt = '''
    Intent: Write a function to find the similar elements from the given two tuple lists.

    Plan: Let's think step by step.
    1. Define a function that takes two lists of tuples as arguments.
    2. Create an empty list to store the similar elements found in both tuple lists.
    3. Use a loop to iterate through each element (tuple) in the first list.
    4. For each tuple in the first list, check if it also exists in the second list.
    5. If the tuple from the first list is found in the second list, append it to the result list.
    6. After iterating through the first list, return the list containing the similar elements.

    Intent: Write a python function to identify non-prime numbers.

    Plan: Let's think step by step.
    1. Start by defining a function that accepts a single integer as an argument. This integer will be the number we want to check.
    2. Before proceeding to the prime check, consider edge cases.
    3. A prime number is a number greater than 1 that has no divisors other than 1 and itself.
    4. Loop through all numbers starting from 2 up to the square root of the number (inclusive).
    5. If any of these numbers divides the given number evenly (i.e., the remainder is 0), the number is non-prime.
    6. If a divisor is found in the loop, the function should return True (indicating the number is non-prime).
    7. If no divisors are found, the function should return False (indicating the number is prime).

    Intent: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.

    Plan: Let's think step by step.
    1. Import the Required Module.
    2. Define a function that accepts two arguments.
    3. Check if the list is empty or if n is greater than the length of the list. If either case is true, handle it appropriately.
    4. Use the heapq.nlargest Function.
    5. The heapq.nlargest function returns a list of the n largest integers. Return this list as the result of the function.

    How about this intent: {problem}.

    Plan: Let's think step by step.
    '''

    MBPP_implementation_prompt = '''
    {problem}
    Please complete the task with the following steps in Python.
    The function name and input variables should follow this template: {function_name}.

    {plan}
    '''

    APPS_plan_prompt = '''
Intent: 
Anton has the integer x. He is interested what positive integer, which doesn't exceed x, has the maximum sum of digits.

Your task is to help Anton and to find the integer that interests him. If there are several such integers, determine the biggest of them. 


-----Input-----

The first line contains the positive integer x (1 ≤ x ≤ 10^18) — the integer which Anton has. 


-----Output-----

Print the positive integer which doesn't exceed x and has the maximum sum of digits. If there are several such integers, print the biggest of them. Printed integer must not contain leading zeros.

Plan: Let's think step by step.
1. Convert the input string to an integer x.  
2. Compute the digit sum of x and store it as max_sum. Set your result to x.  
3. Convert x to a string.  
4. For each digit in this string (except where the digit is '0'):  
   - Decrease that digit by 1.  
   - Make all subsequent digits '9'.  
   - Compute the digit sum of this new number.  
   - If it's greater than the current max_sum (or equal with a larger number), update max_sum and the result.  
5. Return the final result.


Intent:
You have a long fence which consists of $n$ sections. Unfortunately, it is not painted, so you decided to hire $q$ painters to paint it. $i$-th painter will paint all sections $x$ such that $l_i \\le x \\le r_i$.

Unfortunately, you are on a tight budget, so you may hire only $q - 2$ painters. Obviously, only painters you hire will do their work.

You want to maximize the number of painted sections if you choose $q - 2$ painters optimally. A section is considered painted if at least one painter paints it.


-----Input-----

The first line contains two integers $n$ and $q$ ($3 \\le n, q \\le 5000$) — the number of sections and the number of painters availible for hire, respectively.

Then $q$ lines follow, each describing one of the painters: $i$-th line contains two integers $l_i$ and $r_i$ ($1 \\le l_i \\le r_i \\le n$).


-----Output-----

Print one integer — maximum number of painted sections if you hire $q - 2$ painters.

Plan: Let's think step by step.
1. Read and parse inputs:  
   - Extract n (number of fence sections) and q (number of painters).  
   - Record each painter's range (l_i, r_i).

2. Initialize and compute coverage:  
   - Create a coverage array of length n+1 (to index sections easily by their actual number).  
   - For each painter's range, increment coverage on the relevant indices.  
   - This gives how many times each section is painted when all painters are active.

3. Consider removing pairs of painters:  
   - Since exactly two painters will not paint, iterate through all possible pairs (i, j).  
   - Temporarily reduce the coverage for the ranges of painter i and painter j.

4. Recalculate painted sections:  
   - After removing coverage for the two painters in each pair, count how many sections remain with at least one coat of paint.  
   - Keep track of the maximum number of painted sections across all pairs of painters removed.

5. Return the maximum painted sections:  
   - Output the best result found.



Intent:
Jamie loves sleeping. One day, he decides that he needs to wake up at exactly hh: mm. However, he hates waking up, so he wants to make waking up less painful by setting the alarm at a lucky time. He will then press the snooze button every x minutes until hh: mm is reached, and only then he will wake up. He wants to know what is the smallest number of times he needs to press the snooze button.

A time is considered lucky if it contains a digit '7'. For example, 13: 07 and 17: 27 are lucky, while 00: 48 and 21: 34 are not lucky.

Note that it is not necessary that the time set for the alarm and the wake-up time are on the same day. It is guaranteed that there is a lucky time Jamie can set so that he can wake at hh: mm.

Formally, find the smallest possible non-negative integer y such that the time representation of the time x·y minutes before hh: mm contains the digit '7'.

Jamie uses 24-hours clock, so after 23: 59 comes 00: 00.


-----Input-----

The first line contains a single integer x (1 ≤ x ≤ 60).

The second line contains two two-digit integers, hh and mm (00 ≤ hh ≤ 23, 00 ≤ mm ≤ 59).


-----Output-----

Print the minimum number of times he needs to press the button.

Plan: Let's think step by step.
1. Input Parsing
   - Read the first line (integer `x`) which represents how many minutes the time will be moved backward in each step.  
   - Read the second line, split it into two integers (`hh` and `mm`) representing the starting hour and minute.

2. Lucky Check
   - Define a check (a helper function or simple check) that determines whether the current time (hours and minutes) contains the digit `'7'` when converted to a string (e.g., `'07'`, `'17'`, etc.).

3. Iterate Backwards in Time 
   - Initialize a counter (`y`) to count how many times you subtract `x` minutes until the time becomes “lucky.”  
   - Use a loop to repeatedly:
     1. Check if the current time is lucky.  
     2. If it is not lucky, subtract `x` minutes from the current time.  
     3. Handle the overflow if minutes go below zero (add 60 to minutes and decrease hour by one).  
     4. If hour goes below zero, reset it to 23.  
     5. Increment the counter each time you adjust the time.

4. Output
   - Once the current time is considered lucky, the loop ends.  
   - Return (or print) the counter (`y`), which is the number of backward steps required.

   
How about this intent: 
{prompt}.

Plan: Let's think step by step.
'''

    APPS_implementation_prompt = '''
{prompt}
Please complete the task with the following steps in Python.

{plan}
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
            return self.HumanEval_plan_prompt.format(prompt=prompt)
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_plan_prompt.format(problem=prompt)
        else:
            return self.APPS_plan_prompt.format(prompt=prompt)

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
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt)}
                ]
            elif 'MBPP' in self.dataset_name:
                function_name = utils.get_function_info(per_data['test_list'][0])
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': ''},
                    {'role': 'user', 'content': self.form_technique_prompt(prompt, function_name)}
                ]
            else:
                prompt = per_data['prompt']
                message = [
                    {'role': 'system', 'content': ''},
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
            total_input_token, total_output_token, total_thought_token = 0, 0, 0
            result = copy.copy(per_data)

            if 'gemini' in self.model_name:
                response1, input_token, output_token, thought_token = self.run_model(message)
                total_thought_token += thought_token if thought_token is not None else 0
            else:
                response1, input_token, output_token = self.run_model(message)
            total_input_token += input_token
            total_output_token += output_token
            
            if 'HumanEval' in self.dataset_name:
                implement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.HumanEval_implementation_prompt.format(prompt=per_data['prompt'], plan=response1)}
                ]
            elif 'MBPP' in self.dataset_name:
                implement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.MBPP_implementation_prompt.format(problem=per_data['prompt'], function_name=utils.get_function_info(per_data['test_list'][0]), plan=response1)}
                ]
            else:
                implement_message = [
                    {'role': 'system', 'content': self.system_message},
                    {'role': 'user', 'content': self.APPS_implementation_prompt.format(prompt=per_data['prompt'], plan=response1)}
                ]

            if 'gemini' in self.model_name:
                response2, input_token, output_token, thought_token = self.run_model(message)
                total_thought_token += thought_token if not None else 0
            else:
                response2, input_token, output_token = self.run_model(message)

            total_input_token += input_token
            total_input_token += output_token
            code = utils.process_generation_to_code(response2)
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
