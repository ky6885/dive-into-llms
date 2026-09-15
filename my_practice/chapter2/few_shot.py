import os
from openai import OpenAI


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


prompt = """
Q: There are 15 trees in the grove.
After workers plant some more trees, there are 21 trees.
How many trees did they plant?
A: 6

Q: There are 3 cars in a parking lot.
2 more cars arrive.
How many cars are there now?
A: 5

Q: Jason had 20 lollipops.
He gave some to Denny and now has 12.
How many did he give to Denny?
A: 8

Q: There were 10 friends playing a video game online.
Then 7 players quit.
If each player left had 8 lives,
how many lives did they have total?
A:
"""


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


print("Few-shot Prompt：")
print(prompt)

print("\n模型回答：")
print(response.choices[0].message.content)