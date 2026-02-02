import subprocess
import tempfile
import sys
import textwrap
import ast
import json
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

    

def eval_apps(code, test_string):
    full_test = '''
{code}

{test_string}

check(solution)
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
        return False


def check_stdin(code: str, input_data: str, expected_output: str, timeout=5):
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False
    ) as f:
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
            print("wrong answer")
            return False

    except subprocess.TimeoutExpired:
        print("timeout")
        return False

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return False

def check_livecodebench(code: str, tests: str, timeout=5):
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

def check_functional(code: str, test_data: dict):
    namespace = {}

    # 1️⃣ Load the code
    try:
        exec(code, namespace)
    except Exception as e:
        print(e)
        return False

    cls = namespace["Solution"]
    obj = cls()  # instantiate
    # pick first public method
    methods = [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith("__")]
    if not methods:
        print("class `Solution` has no callable methods")
        return False
    method_name = methods[0]
    fn = getattr(obj, method_name)

    if fn is None:
        print("No function or class method found to test")
        return False


    calls = [(test_data["input"], test_data["output"])]

    for inp, expected in calls:
        try:
            # parse JSON-style strings if necessary
            args = parse_input(inp)

            # if multi-argument function, unpack
            out = fn(*args)
            expected_parsed = json.loads(expected) if isinstance(expected, str) else expected


        except Exception as e:
            print("runtime error during call")
            print(e)
            return False

        if out != expected_parsed:
            print("wrong answer")
            return False

    print("correct")
    return True


def parse_input(inp_str):
    lines = inp_str.splitlines()
    args = [json.loads(line) for line in lines]
    return args