import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有检测到 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": "请用一句话解释什么是大语言模型。"
        }
    ]
)

print("模型回答：")
print(response.choices[0].message.content)