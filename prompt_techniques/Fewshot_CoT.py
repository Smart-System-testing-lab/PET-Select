import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm

from src import utils
from src import model

from prompt_techniques.Techniques import BaseGenerator

class FewshotCoTGenerator(BaseGenerator):
    HumanEval_Fewshot_prompt = '''
    Here are some examples of how to generate the code step by step.

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
        Let's complete the following code step by step.
        """
        # Step 1: Create a variable to store the result
        result = False
        # Step 2: Loop through the list of numbers
        for i in range(len(numbers)):
            # Step 3: Check if the current number is within the threshold of any other number in the list
            for j in range(i+1, len(numbers)):
                if abs(numbers[i] - numbers[j]) <= threshold:
                    # Step 4: If the condition is met, set the result to True and break out of the loop
                    result = True
                    break
            # Step 5: If the result is already True, break out of the loop
            if result:
                break

        # Step 6: Return the result
        return result
    ```

    Example 2:

    ```python
    from typing import List
    def rescale_to_unit(numbers: List[float]) -> List[float]:
        """ Given list of numbers (of at least two elements), apply a linear transform to that list,
        such that the smallest number will become 0 and the largest will become 1
        >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])
        [0.0, 0.25, 0.5, 0.75, 1.0]
        Let's complete the following code step by step.
        """
        # Step 1: Find the smallest and largest numbers in the list
        smallest = min(numbers)
        largest = max(numbers)
        # Step 2: Calculate the difference between the largest and smallest numbers
        difference = largest - smallest
        # Step 3: Create a new list to store the rescaled numbers
        rescaled_numbers = []
        # Step 4: Loop through each number in the original list
        for number in numbers:
            # Step 5: Apply the linear transform to each number
            rescaled_number = (number - smallest) / difference
            # Step 6: Add the rescaled number to the new list
            rescaled_numbers.append(rescaled_number)
        # Step 7: Return the new list
        return rescaled_numbers
    ```

    Example 3:

    ```python
    def strlen(string: str) -> int:
        """ Return length of given string
        >>> strlen('')
        0
        >>> strlen('abc')
        3
        Let's complete the following code step by step.
        """
        # 1. Initialize a variable to store the length of the string
        length = 0
        # 2. Use a for loop to iterate through each character in the string
        for char in string:
            # 3. Increment the length variable by 1 for each character
            length += 1
        # 4. Return the length variable
        return length
    ```

    How about this function?
    {prompt}
    '''

    MBPP_Fewshot_CoT_prompt = '''
    You are an expert Python programmer, and you should write code step by step to complete the task.

    Example 1:
    Here is your task: Write a function to find the similar elements from the given two tuple lists.
    Your code should pass these tests: ["assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)", "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)", "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)"]

    ```python
    def similar_elements(test_tup1, test_tup2):
        # Convert both tuples to sets to remove duplicates and allow for set operations
        res = tuple(set(test_tup1) & set(test_tup2))
        # The '&' operator finds the intersection of the two sets, i.e., elements common to both sets
        return (res)  # Return the common elements as a tuple
    ```

    Example 2:
    Here is your task: Write a python function to identify non-prime numbers.
    Your code should pass these tests: ["assert is_not_prime(2) == False", "assert is_not_prime(10) == True", "assert is_not_prime(35) == True"]

    ```python
    import math  # Import the math module to use mathematical functions

    def is_not_prime(n):
        result = False  # Initialize a variable 'result' to False. It indicates whether 'n' is not a prime number.

        # Loop from 2 to the square root of 'n', rounded down to the nearest whole number, then add 1 to include that number in the loop.
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:  # Check if 'n' is divisible by 'i' (i.e., no remainder)
                result = True  # If 'n' is divisible by any number other than 1 and itself, set 'result' to True.

        return result  # Return the value of 'result'. True if 'n' is not a prime number, False if 'n' is a prime number.
    ```

    Example 3:
    Here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
    Your code should pass these tests: ["assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ", "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]"]

    ```python
    import heapq as hq  # Import the heapq module and rename it as hq for convenience

    def heap_queue_largest(nums, n):
        # Use the heapq.nlargest function to get the 'n' largest elements from the list 'nums'
        largest_nums = hq.nlargest(n, nums)
        return largest_nums  # Return the list of the 'n' largest elements
    ```

    How about this task?
    Here is your task: {prompt}
    The function name and input variables should follow this template: {function_name}.
    '''

    APPS_Fewshot_CoT_prompt = '''
Here are some examples of how to generate the code step by step.

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
    # 1. Read and convert the input string to an integer x
    x = int(stdin.strip())
    
    # 2. Define a helper function to calculate the sum of digits of a number
    def sum_of_digits(n):
        return sum(int(d) for d in str(n))
    
    # 3. Initialize:
    #    - max_sum to the sum of digits of x (this is our baseline maximum)
    #    - result to x (this is our baseline result)
    max_sum = sum_of_digits(x)
    result = x
    
    # 4. Convert x to a string to more easily manipulate individual digits
    x_str = str(x)
    n = len(x_str)
    
    # 5. Loop through each digit of x_str
    for i in range(n):
        # Skip if the current digit is '0' because reducing it doesn't make sense
        if x_str[i] == '0':
            continue
        
        # 6. Construct a "candidate" number by:
        #    - Taking all digits up to i unchanged
        #    - Reducing the i-th digit by 1
        #    - Replacing all subsequent digits with '9'
        new_number = int(x_str[:i] + str(int(x_str[i]) - 1) + '9' * (n - i - 1))
        
        # 7. Calculate the sum of digits of this new candidate
        current_sum = sum_of_digits(new_number)
        
        # 8. If the new candidate has a larger digit sum, or ties the digit sum but is numerically larger,
        #    update max_sum and result
        if current_sum > max_sum or (current_sum == max_sum and new_number > result):
            max_sum = current_sum
            result = new_number
    
    # 9. Return the best candidate as a string
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
    # 1. Parse the input
    lines = stdin.strip().split('\n')
    n, q = map(int, lines[0].split())         # n = number of fence sections, q = number of painters
    painters = [tuple(map(int, line.split())) for line in lines[1:]]
    
    # 2. Initialize a coverage array
    #    This array (index 1 through n) will track how many painters paint each section.
    coverage = [0] * (n + 1)
    
    # 3. Calculate the initial coverage for all painters
    for l, r in painters:
        for i in range(l, r + 1):
            coverage[i] += 1
    
    # 4. Calculate the total number of painted sections initially
    #    (where coverage > 0 means the section is painted by at least one painter).
    total_painted = sum(1 for i in range(1, n + 1) if coverage[i] > 0)
    
    # 5. We want to remove exactly two painters to maximize the painted fence length 
    #    by the remaining (q-2) painters.
    max_painted = 0
    
    # 6. Iterate over all possible pairs of painters to remove
    for i in range(q):
        for j in range(i + 1, q):
            
            # 7. Create a copy of the coverage array to simulate removing painters i and j
            temp_coverage = coverage[:]
            
            # 8. Remove coverage contributed by painter i
            l1, r1 = painters[i]
            for k in range(l1, r1 + 1):
                temp_coverage[k] -= 1
            
            # 9. Remove coverage contributed by painter j
            l2, r2 = painters[j]
            for k in range(l2, r2 + 1):
                temp_coverage[k] -= 1
            
            # 10. Calculate the number of sections still painted after removing these two painters
            painted_with_removal = sum(1 for k in range(1, n + 1) if temp_coverage[k] > 0)
            
            # 11. Update max_painted if we found a better (larger) painted count
            max_painted = max(max_painted, painted_with_removal)
    
    # 12. Return the best possible number of painted sections (as a string)
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
    # 1. Parse the input
    lines = stdin.strip().split('\n')
    x = int(lines[0])                # x is the number of minutes we move backwards each step
    hh, mm = map(int, lines[1].split())  # hh and mm are the current hour and minute

    # 2. Define a function to check if the current time is "lucky"
    def is_lucky(h, m):
        return '7' in f"h:02d m:02d"

    # 3. Initialize a counter for how many steps it takes to reach a lucky time
    y = 0

    # 4. Keep subtracting x minutes from the current time until we hit a "lucky" time
    while not is_lucky(hh, mm):
        # Subtract x minutes
        mm -= x

        # If the minutes go below zero, wrap around to the previous hour
        if mm < 0:
            mm += 60
            hh -= 1
            # If the hour goes below zero, wrap around to 23 (midnight boundary)
            if hh < 0:
                hh = 23

        # Increment the step counter
        y += 1

    # 5. Return how many steps (y) it took as a string
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
            return self.MBPP_Fewshot_CoT_prompt.format(prompt=prompt, function_name=function_name)
        else:
            return self.APPS_Fewshot_CoT_prompt.format(prompt=prompt)

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
        with open(output_path, 'a') as f:
            for res in responses:
                f.write(json.dumps(res) + "\n")
