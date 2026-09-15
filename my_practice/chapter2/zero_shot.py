import os
from openai import OpenAI


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


question = """
There were 10 friends playing a video game online.
Then 7 players quit.
If each player left had 8 lives,
how many lives did they have total?
"""


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": question
        }
    ]
)


print("问题：")
print(question)

print("\n模型回答：")
print(response.choices[0].message.content)