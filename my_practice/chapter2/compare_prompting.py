import os
from openai import OpenAI


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


def ask(prompt):
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


question = """
A shop had 48 bottles of water.
It sold 17 bottles in the morning and 13 bottles in the afternoon.
Then it received 25 new bottles.
How many bottles does the shop have now?
"""


zero_shot = f"""
Answer the following question.

{question}
"""


zero_shot_cot = f"""
Answer the following question.

{question}

Let's think step by step.
"""


few_shot = f"""
Q: A box had 20 apples. 8 were eaten. How many remain?
A: 12

Q: There were 14 students. 5 left. How many remain?
A: 9

Q: {question}
A:
"""


few_shot_cot = f"""
Q: A box had 20 apples. 8 were eaten. How many remain?
A: There were 20 apples originally.
8 were eaten.
20 - 8 = 12.
The answer is 12.

Q: There were 14 students. 5 left. How many remain?
A: There were 14 students originally.
5 left.
14 - 5 = 9.
The answer is 9.

Q: {question}
A:
"""


experiments = {
    "Zero-shot": zero_shot,
    "Zero-shot CoT": zero_shot_cot,
    "Few-shot": few_shot,
    "Few-shot CoT": few_shot_cot,
}


for name, prompt in experiments.items():

    print("=" * 60)
    print(name)
    print("=" * 60)

    answer = ask(prompt)

    print(answer)
    print()