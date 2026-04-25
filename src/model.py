import google.genai.errors
import re
import openai
from openai import OpenAI
from openai import RateLimitError
import os
import requests
import json
from google.genai import Client, errors, types
from google import genai
from google.genai import types
from google.api_core import exceptions as gcp_exceptions
from google.genai import errors as genai_errors
import os
#client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

import time

# client = OpenAI(api_key=openai.api_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_json_key"


def call_gemini(message, args, max_retries=10):
    max_tok = 4096 # 2048 for all other than LiveCodeBench
    delay = 30

    client = genai.Client(
        vertexai=True,
        project="gen-lang-client-0460465708",
        location="global",
        http_options=types.HttpOptions(timeout = 180000)
    )

    for attempt in range(max_retries):

        try:
            user_content = message[1]["content"]
            print(user_content)

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=message[0]["content"],
                    temperature=1.0, # THIS SHOULD BE 1!
                    max_output_tokens=max_tok,
                    thinking_config=types.ThinkingConfig(thinking_level='medium')
                ),
            )

            text = response.text
            print(text)

            if not text:
                print(f"Malformed text encountered. Retry!")
                max_tok = 4096
                continue # Accomodate for malformed responses

            meta = response.usage_metadata
            prompt_tokens = meta.prompt_token_count
            completion_tokens = meta.candidates_token_count
            thought_tokens = meta.thoughts_token_count
            print('successful call!')

            return text, prompt_tokens, completion_tokens, thought_tokens

        except genai_errors.ClientError as e:
            delay *= 2
            print(f"{e}: Retrying in {delay}s")
            #print(f"User message preview: {message[1]['content']}")
            time.sleep(delay)

        except gcp_exceptions.ServiceUnavailable:
            delay *= 2
            print(f"[503] Model overloaded, retrying in {delay}s")
            time.sleep(delay)

        except gcp_exceptions.DeadlineExceeded as e:
            print(f"Request deadline exceeded (timeout): {e}")
            time.sleep(delay)
            continue

        except google.genai.errors.ServerError as e:
            print(f"Request deadline exceeded (timeout): {e}")
            time.sleep(delay)
            continue

        except requests.exceptions.Timeout as e:
            # HTTP-level timeout
            print(f"HTTP timeout: {e}")
            time.sleep(delay)
            continue

        except Exception as e:
            raise RuntimeError(f"Gemini call failed: {e}")

    raise RuntimeError("Max retries exceeded")

def call_chat_gpt(message, args):
    wait = 1
    client = OpenAI(api_key='my_fake_key')
    while True:
        try:
            ans = client.chat.completions.create(model=args.model,
            max_tokens=4096,
            messages=message,
            temperature=args.temperature,
            n=1)
            return ans.choices[0].message.content, ans.usage.prompt_tokens, ans.usage.completion_tokens
        except RateLimitError as e:
            print(e)
            time.sleep(min(wait, 60))
            wait *= 2
        except openai.InternalServerError as e:
            print(e)
            time.sleep(min(wait, 60))
            wait *= 2

def query_firework(message, args, model="deepseek-v3", delay=60):
    api_key = "fake key"
    retry = 6

    for r in range(0, retry):

        import json
        import time
        import requests

        # Assuming this is inside your retry loop
        if "deepseek-v3" in model:

            url = "https://api.fireworks.ai/inference/v1/chat/completions"

            payload = {
                "model": "accounts/fireworks/models/deepseek-v3p1",
                "max_tokens": 4096,
                "temperature": args.temperature,
                "messages": message,
                "stream": True,
                "stream_options": {"include_usage": True}
            }

            headers = {
                "Accept": "text/event-stream",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            # Passed stream=True to the request
            response = requests.request("POST", url, json=payload, headers=headers, stream=True)

            if response.status_code == 200:
                content = ""
                input_token = 0
                output_token = 0

                try:

                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')

                            if line_str.startswith("data: "):
                                data_str = line_str[6:]

                                if data_str == "[DONE]":
                                    break

                                data_json = json.loads(data_str)


                                if data_json.get("choices"):
                                    delta = data_json["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        chunk_text = delta["content"]
                                        print(chunk_text, end="", flush=True)
                                        content += chunk_text


                                if data_json.get("usage"):
                                    input_token = data_json["usage"].get("prompt_tokens", 0)
                                    output_token = data_json["usage"].get("completion_tokens", 0)

                    print()

                    # Return exactly what your original logic returned
                    return content, input_token, output_token

                except json.JSONDecodeError as e:
                    # Catch stream parsing issues and trigger your loop's continue
                    print(f"\nStream parsing error: {e}")
                    continue

            else:
                # Your existing exponential backoff logic
                print(response.text)
                time.sleep(min(delay, 60))
                print(f"waiting for {delay}s")
                delay *= 2
                continue
        elif model == "starcoder":
            url = "https://api.fireworks.ai/inference/v1/completions"
            payload = {
            "model": "my_fake_model",
            "max_tokens": 4096,
            "temperature": args.temperature,
            "prompt": message[1]['content']
            }
            headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }
            response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(data)
                    # Extract content
                    content = data["choices"][0]["text"]
                    input_token = data["usage"]["prompt_tokens"]
                    output_token = data["usage"]["completion_tokens"]
                    return content, input_token, output_token
                except json.JSONDecodeError as e:
                    # Return an error message if JSON decoding fails
                    return f"JSONDecodeError: {e} - Response text: {response.text}"
            else:
                return f"Error: {response.status_code}, {response.text}"
    raise RuntimeError("Exceeded retries")






def get_embedding(text, model='text-embedding-3-large'):
    client = OpenAI(
        api_key='f')
    response = client.embeddings.create(input=text, model = model)
    return response.data[0].embedding