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
import time

# client = OpenAI(api_key=openai.api_key)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "my_json_key"


def call_gemini(message, args, max_retries=10):
    max_tok = 4096 # 2048 for all other than LiveCodeBench
    delay = 30

    client = genai.Client(
        vertexai=True,
        project="vertex_ai_project_name",
        location="global",
        http_options=types.HttpOptions(timeout = 180000)
    )

    for attempt in range(max_retries):

        try:
            user_content = message[1]["content"]

            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=message[0]["content"],
                    temperature=args.temperature, # THIS SHOULD BE 1!
                    max_output_tokens=max_tok,
                    thinking_config=types.ThinkingConfig(thinking_level='medium')
                ),
            )

            text = response.text

            if not text:
                print(f"Malformed text encountered. Retry!")  # Debug: check text
                max_tok = 8192 # double to accomodate for longer prompt
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
    client = OpenAI(api_key='open_ai_key')
    while True:
        try:
            ans = client.chat.completions.create(model=args.model,
            max_tokens=1000,
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
    api_key = "fireworks_ai_key"
    retry = 6

    for r in range(0, retry):

        if "deepseek-v3" in model:

            url = "https://api.fireworks.ai/inference/v1/chat/completions"

            payload = {
                "model": f"accounts/fireworks/models/{model}",
                "max_tokens": 16384,
                "temperature": args.temperature,
                "messages": message
            }

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            response = requests.request("POST", url, json=payload, headers=headers)

            if response.status_code == 200:
                try:
                    data = response.json()

                    # Extract content
                    content = data["choices"][0]["message"]["content"]
                    input_token = data["usage"]["prompt_tokens"]
                    output_token = data["usage"]["completion_tokens"]
                    return content, input_token, output_token
                except json.JSONDecodeError as e:
                    # Return an error message if JSON decoding fails
                    print(response.text)
                    continue
            else:
                print(response.text)
                time.sleep(min(delay, 60))
                print(f"waiting for {delay}s")
                delay *= 2
                continue

        elif model == "starcoder":
            url = "https://api.fireworks.ai/inference/v1/completions"
            payload = {
            "model": "accounts/chungyuwang5507-f1662b/deployedModels/starcoder2-15b-bb7b2085",
            "max_tokens": 2048,
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
        api_key='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')
    response = client.embeddings.create(input=text, model = model)
    return response.data[0].embedding