from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1573/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen2.5-0.5b-instruct-generic-cpu:4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)