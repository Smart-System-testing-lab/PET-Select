import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class FewshotGenerator(BaseGenerator):
    HumanEval_Fewshot_prompt = '''
    There are some examples of how to generate the code.

    Example 1:

    ```python
    from typing import List
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        """ Check if in given list of numbers, are any two numbers closer to each other than
        given threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
        True
        """
        for idx, elem in enumerate(numbers):
            for idx2, elem2 in enumerate(numbers):
                if idx != idx2:
                    distance = abs(elem - elem2)
                    if distance < threshold:
                        return True

        return False
    ```

    Example 2:

    ```python
    from typing import List
    def separate_paren_groups(paren_string: str) -> List[str]:
        """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
        separate those group into separate strings and return the list of those.
        Separate groups are balanced (each open brace is properly closed) and not nested within each other
        Ignore any spaces in the input string.
        >>> separate_paren_groups('( ) (( )) (( )( ))')
        ['()', '(())', '(()())']
        """
        result = []
        current_string = []
        current_depth = 0

        for c in paren_string:
            if c == '(':
                current_depth += 1
                current_string.append(c)
            elif c == ')':
                current_depth -= 1
                current_string.append(c)

                if current_depth == 0:
                    result.append(''.join(current_string))
                    current_string.clear()

        return result
    ```

    Example 3:

    ```python
    def truncate_number(number: float) -> float:
        """ Given a positive floating point number, it can be decomposed into
        and integer part (largest integer smaller than given number) and decimals
        (leftover part always smaller than 1).

        Return the decimal part of the number.
        >>> truncate_number(3.5)
        0.5
        """
        return number % 1.0
    ```

    How about this function?
    {prompt}
    '''

    MBPP_Fewshot_prompt = '''
    Here are some examples of how to generate the code.

    Example 1:
    Here is your task: Write a function to find the similar elements from the given two tuple lists.
    Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

    ```python
    def similar_elements(test_tup1, test_tup2):
        res = tuple(set(test_tup1) & set(test_tup2))
        return (res) 
    ```

    Example 2:
    Here is your task: Write a python function to identify non-prime numbers.
    Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

    ```python
    import math
    def is_not_prime(n):
        result = False
        for i in range(2,int(math.sqrt(n)) + 1):
            if n % i == 0:
                result = True
        return result
    ```

    Example 3:
    Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
    Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

    ```python
    import heapq as hq
    def heap_queue_largest(nums,n):
        largest_nums = hq.nlargest(n, nums)
        return largest_nums
    ```

    How about this task?
    Here is your task: {prompt}
    The function name and input variables should follow this template: {function_name}.
    '''

    APPS_Fewshot_prompt = '''
Here are some examples of how to generate the code.

Example 1:

Here is your task: 
Anton has the integer x. He is interested what positive integer, which doesn't exceed x, has the maximum sum of digits.

Your task is to help Anton and to find the integer that interests him. If there are several such integers, determine the biggest of them. 


-----Input-----

The first line contains the positive integer x (1 ≤ x ≤ 10^18) — the integer which Anton has. 


-----Output-----

Print the positive integer which doesn't exceed x and has the maximum sum of digits. If there are several such integers, print the biggest of them. Printed integer must not contain leading zeros.

```python
def solution(stdin: str) -> str:
    x = int(stdin.strip())
    
    # Function to calculate the sum of digits of a number
    def sum_of_digits(n):
        return sum(int(d) for d in str(n))
    
    # Start with the number itself
    max_sum = sum_of_digits(x)
    result = x
    
    # Convert x to a string to manipulate digits
    x_str = str(x)
    n = len(x_str)
    
    # Try to form numbers by reducing some digit and making all following digits 9
    for i in range(n):
        if x_str[i] == '0':
            continue
        
        # Form a new number by reducing the current digit by 1 and making all following digits 9
        new_number = int(x_str[:i] + str(int(x_str[i]) - 1) + '9' * (n - i - 1))
        
        # Calculate the sum of digits of the new number
        current_sum = sum_of_digits(new_number)
        
        # Check if this new number has a greater sum of digits
        if current_sum > max_sum or (current_sum == max_sum and new_number > result):
            max_sum = current_sum
            result = new_number
    
    return str(result)
```

Example 2:

Here is your task: 
You have a long fence which consists of $n$ sections. Unfortunately, it is not painted, so you decided to hire $q$ painters to paint it. $i$-th painter will paint all sections $x$ such that $l_i \\le x \\le r_i$.

Unfortunately, you are on a tight budget, so you may hire only $q - 2$ painters. Obviously, only painters you hire will do their work.

You want to maximize the number of painted sections if you choose $q - 2$ painters optimally. A section is considered painted if at least one painter paints it.


-----Input-----

The first line contains two integers $n$ and $q$ ($3 \\le n, q \\le 5000$) — the number of sections and the number of painters availible for hire, respectively.

Then $q$ lines follow, each describing one of the painters: $i$-th line contains two integers $l_i$ and $r_i$ ($1 \\le l_i \\le r_i \\le n$).


-----Output-----

Print one integer — maximum number of painted sections if you hire $q - 2$ painters.

```python
def solution(stdin: str) -> str:
    # Parse input
    lines = stdin.strip().split('\n')
    n, q = map(int, lines[0].split())
    painters = [tuple(map(int, line.split())) for line in lines[1:]]
    
    # Initialize a list to count coverage of each section
    coverage = [0] * (n + 1)
    
    # Calculate the initial coverage with all painters
    for l, r in painters:
        for i in range(l, r + 1):
            coverage[i] += 1
    
    # Calculate the total number of initially painted sections
    total_painted = sum(1 for i in range(1, n + 1) if coverage[i] > 0)
    
    # To find the optimal q-2 painters, we need to consider the effect of removing each pair of painters
    max_painted = 0
    
    # Iterate over all pairs of painters to remove
    for i in range(q):
        for j in range(i + 1, q):
            # Create a temporary coverage array to simulate removing painters i and j
            temp_coverage = coverage[:]
            
            # Remove the effect of painter i
            l1, r1 = painters[i]
            for k in range(l1, r1 + 1):
                temp_coverage[k] -= 1
            
            # Remove the effect of painter j
            l2, r2 = painters[j]
            for k in range(l2, r2 + 1):
                temp_coverage[k] -= 1
            
            # Calculate the number of painted sections with these two painters removed
            painted_with_removal = sum(1 for k in range(1, n + 1) if temp_coverage[k] > 0)
            
            # Update the maximum painted sections found
            max_painted = max(max_painted, painted_with_removal)
    
    return str(max_painted)
```

Example 3:

Here is your task: 
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

```python
def solution(stdin: str) -> str:
    lines = stdin.strip().split('\n')
    x = int(lines[0])
    hh, mm = map(int, lines[1].split())

    def is_lucky(h, m):
        return '7' in f"h:02d m:02d"

    y = 0
    while not is_lucky(hh, mm):
        mm -= x
        if mm < 0:
            mm += 60
            hh -= 1
            if hh < 0:
                hh = 23
        y += 1

    return str(y)
```

How about this task?
Here is your task: 
{prompt}
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
            return self.HumanEval_Fewshot_prompt.format(prompt=prompt)
        elif 'MBPP' in self.dataset_name:
            return self.MBPP_Fewshot_prompt.format(prompt=prompt, function_name=function_name)
        else:
            return self.APPS_Fewshot_prompt.format(prompt=prompt)

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
            result = copy.copy(per_data)

            if 'gemini' in self.model_name:
                response, input_token, output_token, thought_token = self.run_model(message)
                result['thought_token'] = thought_token if not None else 0
            else:
                response, input_token, output_token = self.run_model(message)

            code = utils.process_generation_to_code(response)
            result['response_code'] = '\n'.join(code)
            result['input_token'] = input_token
            result['output_token'] = output_token
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
        with open(output_path, 'w') as f:
            for res in responses:
                f.write(json.dumps(res) + "\n")
