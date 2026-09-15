import os
import re
from openai import OpenAI


# ============================================================
# 1. DeepSeek
# ============================================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError("没有检测到 DEEPSEEK_API_KEY")


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 2. 测试问题
# ============================================================

question = """
There were 18 students in a classroom.

7 students left.
Then 5 new students entered.

How many students are now in the classroom?
"""


correct_answer = 16.0


# ============================================================
# 3. 三种 Prompt
# ============================================================

zero_shot = f"""
Solve the following problem.

{question}

Your final line must be:

FINAL: <number>
"""


correct_few_shot = f"""
Here are some examples.

Q:
There were 15 apples.
4 apples were eaten.
How many apples remain?

A:
15 - 4 = 11
FINAL: 11


Q:
There were 20 people.
6 people left.
3 people arrived.
How many people are there now?

A:
20 - 6 + 3 = 17
FINAL: 17


Now solve this problem.

Q:
{question}

A:
"""


wrong_few_shot = f"""
Here are some examples.

Q:
There were 15 apples.
4 apples were eaten.
How many apples remain?

A:
15 - 4 = 12
FINAL: 12


Q:
There were 20 people.
6 people left.
3 people arrived.
How many people are there now?

A:
20 - 6 + 3 = 18
FINAL: 18


Follow the examples and solve this problem.

Q:
{question}

A:
"""


prompts = {
    "Zero-shot": zero_shot,
    "Correct Few-shot": correct_few_shot,
    "Wrong Few-shot": wrong_few_shot,
}


# ============================================================
# 4. 调用模型
# ============================================================

def ask(prompt):

    response = client.chat.completions.create(
        model="deepseek-flash",

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],

        temperature=0.8,

        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    return response.choices[0].message.content


# ============================================================
# 5. 提取答案
# ============================================================

def extract_answer(text):

    matches = re.findall(
        r"FINAL\s*:\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if not matches:
        return None

    return float(matches[-1])


# ============================================================
# 6. 实验
# ============================================================

for name, prompt in prompts.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    output = ask(prompt)

    predicted = extract_answer(output)

    print(output)

    print()
    print("提取答案：", predicted)
    print("正确答案：", correct_answer)

    if predicted == correct_answer:
        print("结果：✓ 正确")
    else:
        print("结果：✗ 错误")