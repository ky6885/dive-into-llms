import os
import re
from openai import OpenAI


# ============================================================
# 1. 读取 DeepSeek API Key
# ============================================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有检测到 DEEPSEEK_API_KEY。\n"
        "请先在终端执行：\n"
        'read -s -p "请输入 DeepSeek API Key: " DEEPSEEK_API_KEY\n'
        "echo\n"
        "export DEEPSEEK_API_KEY"
    )


# ============================================================
# 2. 创建 DeepSeek 客户端
# ============================================================

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 测试题目
# ============================================================

questions = [
    {
        "question": """
A shop had 48 bottles of water.
It sold 17 bottles in the morning and 13 in the afternoon.
Then it received 25 new bottles.
How many bottles does it have now?
""",
        "answer": 43,
    },

    {
        "question": """
A farmer has 36 chickens.
He sells 1/3 of them and then buys 8 more chickens.
How many chickens does he have now?
""",
        "answer": 32,
    },

    {
        "question": """
A bus starts with 24 passengers.
At the first stop, 9 get off and 6 get on.
At the second stop, 4 get off and 11 get on.
How many passengers are now on the bus?
""",
        "answer": 28,
    },

    {
        "question": """
A student has 120 dollars.
She spends 25 dollars on a book and then spends half
of the remaining money on a jacket.
How much money does she have left?
""",
        "answer": 47.5,
    },

    {
        "question": """
A factory produces 15 boxes per hour.
Each box contains 12 items.
The factory works for 4 hours,
but 30 items are defective.
How many usable items are produced?
""",
        "answer": 690,
    },
]


# ============================================================
# 4. 调用 DeepSeek
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
    )

    return response.choices[0].message.content


# ============================================================
# 5. Zero-shot Prompt
# ============================================================

def make_zero_shot(question):
    return f"""
Answer the following math problem.

{question}

End your answer with exactly this format:

FINAL: <number>
"""


# ============================================================
# 6. Zero-shot CoT Prompt
# ============================================================

def make_zero_shot_cot(question):
    return f"""
Answer the following math problem.

{question}

Let's think step by step.

End your answer with exactly this format:

FINAL: <number>
"""


# ============================================================
# 7. Few-shot Prompt
# ============================================================

def make_few_shot(question):
    return f"""
Q: A box had 20 apples.
8 were eaten.
How many remain?

A: FINAL: 12


Q: There were 14 students.
5 left.
How many remain?

A: FINAL: 9


Q:
{question}

A:
"""


# ============================================================
# 8. Few-shot CoT Prompt
# ============================================================

def make_few_shot_cot(question):
    return f"""
Q: A box had 20 apples.
8 were eaten.
How many remain?

A:
There were 20 apples originally.
8 were eaten.

20 - 8 = 12.

FINAL: 12


Q: There were 14 students.
5 left.
How many remain?

A:
There were 14 students originally.
5 left.

14 - 5 = 9.

FINAL: 9


Q:
{question}

A:
"""


# ============================================================
# 9. 四种实验方法
# ============================================================

methods = {
    "Zero-shot": make_zero_shot,
    "Zero-shot CoT": make_zero_shot_cot,
    "Few-shot": make_few_shot,
    "Few-shot CoT": make_few_shot_cot,
}


# ============================================================
# 10. 从模型回答中提取 FINAL 数字
# ============================================================

def extract_answer(text):
    """
    首先尝试严格提取：
        FINAL: 43
        FINAL: 47.5

    如果模型没有按照规定格式输出，则返回 None。
    """

    matches = re.findall(
        r"FINAL:\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if not matches:
        return None

    return float(matches[-1])


# ============================================================
# 11. 初始化分数
# ============================================================

scores = {
    name: 0
    for name in methods
}


# ============================================================
# 12. 开始实验
# ============================================================

for index, item in enumerate(questions, start=1):

    question = item["question"]
    correct_answer = float(item["answer"])

    print()
    print("=" * 70)
    print(f"Question {index}")
    print("=" * 70)

    for method_name, prompt_function in methods.items():

        # 生成不同类型的 Prompt
        prompt = prompt_function(question)

        # 调用模型
        answer_text = ask(prompt)

        # 自动提取答案
        predicted = extract_answer(answer_text)

        # ----------------------------------------------------
        # 如果解析失败，打印模型的完整原始回答
        # ----------------------------------------------------

        if predicted is None:

            print()
            print("[解析失败]")
            print("方法：", method_name)
            print("正确答案：", correct_answer)
            print()
            print("模型原始回答：")
            print("-" * 50)
            print(answer_text)
            print("-" * 50)
            print()

        # ----------------------------------------------------
        # 判断答案是否正确
        # ----------------------------------------------------

        is_correct = (
            predicted is not None
            and abs(predicted - correct_answer) < 1e-6
        )

        if is_correct:
            scores[method_name] += 1

        # ----------------------------------------------------
        # 打印当前实验结果
        # ----------------------------------------------------

        print(
            f"{method_name:20s} "
            f"| predicted={predicted} "
            f"| expected={correct_answer} "
            f"| {'✓' if is_correct else '✗'}"
        )


# ============================================================
# 13. 输出最终正确率
# ============================================================

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for method_name, score in scores.items():

    accuracy = score / len(questions)

    print(
        f"{method_name:20s}: "
        f"{score}/{len(questions)} "
        f"accuracy={accuracy:.1%}"
    )