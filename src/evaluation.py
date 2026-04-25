import subprocess
import tempfile
import sys
import textwrap
import ast
import json
import os
import re

import signal


def timeout_handler(signum, frame):
    raise TimeoutError("Test execution exceeded time limit")

def extract_function_body(code, entry_point):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                code = ast.unparse(node.body)
                indent_str = '    '
                indented_code = textwrap.indent(text=code, prefix=indent_str)
                return indented_code
    except:
        return code

def check_code(prompt, final, test, entry_point, timeout=10):
    """
    Thread-safe, hard timeout version of check_code.
    Returns True if code passes, False if it fails or times out.
    """
    # Extract function body
    final_body = extract_function_body(final, entry_point)
    if final_body is not None:
        candidate_code = prompt + final_body
    else:
        candidate_code = prompt

    # Combine candidate code + test into one script
    full_code = textwrap.dedent(f"""
    {candidate_code}

    {test}

    check({entry_point})
    """)

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tmpfile:
            tmpfile.write(full_code)
            tmpfile.flush()

            subprocess.run(
                [sys.executable, tmpfile.name],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        return True

    except subprocess.TimeoutExpired:
        print("Timeout: candidate code took too long")
        return False

    except subprocess.CalledProcessError as e:
        print("Execution failed:")
        print(e.stdout)
        print(e.stderr)
        return False


def MBPP_check_code(final, test, timeout=10):
    """
    Thread-safe, hard timeout version of MBPP_check_code.
    Executes candidate code first, then test code.
    Returns True if both pass, False otherwise.
    """
    # Step 1: Validate candidate code
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tmpfile:
            tmpfile.write(final)
            tmpfile.flush()

            result = subprocess.run(
                [sys.executable, tmpfile.name],
                check=True,
                capture_output=True,
                text=True,
                timeout=5  # Quick validation
            )
            print(final)
    except subprocess.TimeoutExpired:
        print('Candidate code timeout during validation')
        return False
    except subprocess.CalledProcessError as e:
        print('Wrong code')
        print(e.stderr)
        return False
    except Exception as e:
        print('Candidate code error:', str(e))
        return False

    # Step 2: Run the test with the candidate code
    full_code = textwrap.dedent(f"""
    {final}

    {test}
    """)

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as tmpfile:
            tmpfile.write(full_code)
            tmpfile.flush()

            subprocess.run(
                [sys.executable, tmpfile.name],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        print('Success')
        return True

    except subprocess.TimeoutExpired:
        print('Test failed due to timeout')
        return False
    except subprocess.CalledProcessError as e:
        print('Test execution failed:')
        print(e.stdout)
        print(e.stderr)
        return False
    except Exception as e:
        print('Test error:', str(e))
        return False


def eval_humaneval(prompt, code, test, entry_point):
    if entry_point not in code:
        code = prompt + code
    test = test.replace('candidate', entry_point)
    full_test = '''
{code}

{test}

check({entry_point})
    '''

    full_test = full_test.format(code=code, test=test, entry_point=entry_point)
    with open('temp.py', 'w') as f:
        f.write(full_test)

    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        # print(full_test)
        # print(e)
        print("failed")
        return False


def eval_mbpp(code, test_string, is_plus):
    if not is_plus:
        full_test = '''
{code}

test_list = {test_string}

def run_tests():
    """
    Executes each test in test_list using 'exec'.
    If all assertions pass, it prints a success message.
    """
    for test in test_list:
        # Execute each test string, which includes the assert statement
        exec(test)

if __name__ == "__main__":
    run_tests()
        '''
    else:
        full_test = '''
{code}

{test_string}
'''
        
    full_test = full_test.format(code=code, test_string=test_string)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    # print(full_test)
    # quit()
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        print("failed")
        return False

    




# Pre-amble injected before every candidate solution
_STDLIB_PREAMBLE = """import sys, os, math, re, itertools, functools, collections, heapq, bisect
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict, Counter, deque
from itertools import combinations, permutations, product
from functools import lru_cache, reduce
from math import gcd, lcm, inf, ceil, floor, sqrt, log
import sys as _sys, io as _io, inspect
""".strip()

def eval_apps(code, test_string):
        # Strip the __main__ block to prevent stdin hanging
        code = re.sub(r'if __name__\s*==\s*["\']__main__["\'].*', '', code, flags=re.DOTALL).strip()

        # Normalize function name: rename `solve` (and common variants) to `solution`
        code = re.sub(r'\bdef\s+solve\s*\(', 'def solution(', code)

        # Fix functions that take no arguments but should accept an optional input string.
        # Pattern: `def solution():` -> `def solution(stdin_input=None):`
        def _fix_solution_signature(m):
            return 'def solution(stdin_input=None):\n    if stdin_input is not None:\n        sys.stdin = io.StringIO(stdin_input)\n    '

        # Only patch if signature is truly empty: def solution():
        code = re.sub(
            r'\bdef solution\(\s*\)\s*:',
            _fix_solution_signature,
            code
        )

        # Wrapper so that check(solution) works whether solution returns a value
        # or prints to stdout, and regardless of whether it accepts an argument.
        _harness = textwrap.dedent("""
        def _wrap_solution(fn):
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            ]
            accepts_input = len(params) > 0

            def _candidate(stdin_str):
                buf = _io.StringIO()
                old_stdin  = _sys.stdin
                old_stdout = _sys.stdout
                _sys.stdin  = _io.StringIO(stdin_str)
                _sys.stdout = buf
                try:
                    result = fn(stdin_str) if accepts_input else fn()
                finally:
                    _sys.stdin  = old_stdin
                    _sys.stdout = old_stdout
                output = buf.getvalue().strip()
                if result is None:
                    return output
                return str(result).strip()
            return _candidate
            """)

        full_test = textwrap.dedent(f"""
{_STDLIB_PREAMBLE}

{code}

{_harness}

solution = _wrap_solution(solution)

{test_string}

check(solution)
""")

        with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False
        ) as f:
            f.write(full_test)
            temp_path = f.name
        try:
            subprocess.run(
                ["python3", temp_path],
                check=True,
                timeout=10,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print("correct")
            return True
        except Exception as e:
            print(code)
            print("failed")
            print(e)
            return False
    
def check_apps(code, assertion):
    full_test = '''
{code}

{assertion}
'''
    full_test = full_test.format(code=code, assertion=assertion)
    with open('temp.py', 'w') as f:
        f.write(full_test)
    try:
        # signal.signal(signal.SIGALRM, timeout_handler)
        subprocess.run(["python3", "temp.py"], check=True, timeout=5)
        print("correct")
        # signal.alarm(5)
        return True
    except Exception as e:
        print("failed")
        print(f"ERROR: {e}")
        return False


def check_stdin(code: str, input_data: str, expected_output: str, timeout=5):
    # If code defines solution() but never calls it, append the call
    if 'def solution()' in code and code.count('solution()') == 1:
        code = code + '\n\nsolution()'

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        file_path = f.name

    try:
        result = subprocess.run(
            ["python3", file_path],
            input=input_data,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=True,
        )

        # Normalize whitespace
        stdout = result.stdout.strip()
        expected = expected_output.strip()

        if stdout == expected:
            print("correct")
            return True
        else:

            print(stdout)
            print(expected)
            print(code)
            print("failed")
            return False
            return False

    except subprocess.TimeoutExpired:
        print("timeout")
        return False

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return False

def check_livecodebench(code: str, tests: str, timeout=5):
    full_program = f"{code}\n\n{tests}"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
        f.write(full_program)
        temp_path = f.name

    try:
        subprocess.run(
            ["python3", temp_path],
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("correct")
        return True

    except subprocess.TimeoutExpired:
        print("timeout")
        return False

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        print("failed")
        return False


def check_functional(code: str, test_data: dict, timeout=5):
    args = parse_input(test_data["input"])
    print(args)
    print(test_data["input"])
    expected = json.loads(test_data["output"]) if isinstance(test_data["output"], str) else test_data["output"]
    full_program = f"""
import json
from typing import List, Dict, Tuple, Optional, Set
import types
import inspect

{code}

args = {args}
expected = {expected}

if 'Solution' in dir():
    obj = Solution()
    print(dir(obj))
    methods = [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith("__")]
    if not methods:
        raise RuntimeError("No callable methods found on Solution")
    fn = getattr(obj, methods[0])
    try:
        result = fn(*args)
    except TypeError:
        try:
            flat_args = args[0] if len(args) == 1 and isinstance(args[0], list) else args
            result = fn(*flat_args)
        except TypeError:
            result = fn(flat_args)
else:
    fns = [v for v in globals().values() if isinstance(v, types.FunctionType)]
    print(fns)
    if not fns:
        raise RuntimeError("No callable functions found")
    result = fns[0](*args)

assert result == expected, f"Expected {{expected}}, got {{result}}"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_program)
        temp_path = f.name

    try:
        subprocess.run(
            ["python3", temp_path],
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("correct")
        return True
    except subprocess.TimeoutExpired as e:
        print(e.stderr)
        print(code)
        print("timeout")
        return False
    except subprocess.CalledProcessError as e:

        print(e.stderr)
        print(code)
        print("failed")
        return False
    finally:
        os.remove(temp_path)
def check_bigcodebench(code: str, tests: str, timeout=5):
    full_program = textwrap.dedent(f"""
{code}

{tests}
""")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
        f.write(full_program)
        temp_path = f.name

    try:
        subprocess.run(
            ["python3", temp_path],
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("correct")
        return True

    except subprocess.TimeoutExpired:
        print("timeout")
        return False

    except subprocess.CalledProcessError as e:
        print("failed")
        return False

    finally:
        os.remove(temp_path)


def eval_bigcodebench(code: str, tests: str, timeout=5):
    full_test = '''
import json
from typing import List, Dict, Tuple, Optional, Set
import types

{code}

{tests}
'''
    full_test = full_test.format(code=code, tests=tests)

    with open('temp.py', 'w') as f:
        f.write(full_test)

    try:
        subprocess.run(["python3", "temp.py"], check=True, timeout=timeout)
        print("correct")
        return True
    except Exception as e:
        print("failed")
        return False

def parse_input(inp_str):
    lines = inp_str.splitlines()
    args = [json.loads(line) for line in lines]
    return args

