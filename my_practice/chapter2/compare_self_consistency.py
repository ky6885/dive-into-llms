import os
import re
from collections import Counter
from openai import OpenAI


# ============================================================
# 1. 配置
# ============================================================

MODEL = "deepseek-flash"

# 每道题采样次数
N_SAMPLES = 5

# 增加生成随机性
TEMPERATURE = 0.8


# ============================================================
# 2. 读取 API Key
# ============================================================

api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有检测到 DEEPSEEK_API_KEY。\n"
        "请先在终端中加载 DeepSeek API Key。"
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


# ============================================================
# 3. 测试题
# ============================================================

questions = [

    {
        "question": """
A water tank contains 240 liters of water.

First, 25% of the water is drained.
Then an amount equal to one third of the remaining water is added.
Finally, 40 liters are used.

How many liters of water remain?
""",
        "answer": 200.0,
    },

    {
        "question": """
A jacket originally costs $480.

The store gives a 15% discount.
After the discount, an 8% sales tax is added.

What is the final price?
""",
        "answer": 440.64,
    },

    {
        "question": """
Four workers each produce 18 parts per hour.

They work for 7 hours.
One eighth of all produced parts are defective.

How many usable parts are produced?
""",
        "answer": 441.0,
    },

    {
        "question": """
A student has scores of 78, 84, and 91 on three tests.

After a fourth test, the average score across all four tests
is exactly 85.

What was the student's score on the fourth test?
""",
        "answer": 87.0,
    },

    {
        "question": """
A car travels 180 km at 60 km/h.

Then the driver rests for 45 minutes.

After the rest, the car travels another 120 km at 80 km/h.

How many hours pass from the start of the first trip
until the end of the second trip, including the rest?
""",
        "answer": 5.25,
    },

    {
        "question": """
A father is currently four times as old as his son.

In 12 years, the father will be exactly twice as old as his son.

How old is the father now?
""",
        "answer": 24.0,
    },

]


# ============================================================
# 4. 构建 Prompt
# ============================================================

def make_prompt(question):

    return f"""
Solve the following problem carefully.

{question}

Work through the reasoning before giving the final answer.

Your final line MUST use exactly this format:

FINAL: <number>
"""


# ============================================================
# 5. 调用模型一次
# ============================================================

def ask_once(question):

    prompt = make_prompt(question)

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],

        temperature=TEMPERATURE,

        # 为了让 temperature 真正控制普通生成，
        # 这里沿用我们上一个实验的设置。
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    return response.choices[0].message.content


# ============================================================
# 6. 提取最终答案
# ============================================================

def extract_answer(text):

    matches = re.findall(
        r"FINAL\s*:\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if not matches:
        return None

    value = matches[-1].replace(",", "")

    return float(value)


# ============================================================
# 7. 判断两个数字是否相等
# ============================================================

def is_correct(predicted, expected):

    if predicted is None:
        return False

    return abs(predicted - expected) < 1e-6


# ============================================================
# 8. Self-Consistency 多数投票
# ============================================================

def majority_vote(answers):

    valid_answers = [
        answer
        for answer in answers
        if answer is not None
    ]

    if not valid_answers:
        return None, 0

    counts = Counter(valid_answers)

    answer, votes = counts.most_common(1)[0]

    return answer, votes


# ============================================================
# 9. 初始化统计结果
# ============================================================

single_correct = 0
self_consistency_correct = 0


# ============================================================
# 10. 开始实验
# ============================================================

for index, item in enumerate(questions, start=1):

    question = item["question"]
    expected = float(item["answer"])

    print()
    print("=" * 75)
    print(f"Question {index}")
    print("=" * 75)

    print(question.strip())

    print()
    print(f"Expected answer: {expected}")
    print()

    sampled_answers = []


    # --------------------------------------------------------
    # 对同一道题采样 N 次
    # --------------------------------------------------------

    for sample_id in range(1, N_SAMPLES + 1):

        output = ask_once(question)

        predicted = extract_answer(output)

        sampled_answers.append(predicted)

        print(
            f"Sample {sample_id}: "
            f"{predicted}"
        )

        # 如果无法解析，打印原始回答
        if predicted is None:

            print()
            print("[解析失败]")
            print(output)
            print()


    # ========================================================
    # Single Sample
    # ========================================================

    # 把第 1 次采样看作普通的单次推理
    single_answer = sampled_answers[0]

    single_ok = is_correct(
        single_answer,
        expected
    )

    if single_ok:
        single_correct += 1


    # ========================================================
    # Self-Consistency
    # ========================================================

    sc_answer, votes = majority_vote(
        sampled_answers
    )

    sc_ok = is_correct(
        sc_answer,
        expected
    )

    if sc_ok:
        self_consistency_correct += 1


    # ========================================================
    # 当前题结果
    # ========================================================

    print()
    print("-" * 75)

    print(
        f"Single Sample       : {single_answer} "
        f"{'✓' if single_ok else '✗'}"
    )

    print(
        f"Self-Consistency    : {sc_answer} "
        f"({votes}/{N_SAMPLES} votes) "
        f"{'✓' if sc_ok else '✗'}"
    )

    print("-" * 75)


# ============================================================
# 11. 最终统计
# ============================================================

total = len(questions)

single_accuracy = single_correct / total

sc_accuracy = self_consistency_correct / total


print()
print("=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print(
    f"Single Sample:"
    f"       {single_correct}/{total}"
    f" = {single_accuracy:.1%}"
)

print(
    f"Self-Consistency:"
    f"    {self_consistency_correct}/{total}"
    f" = {sc_accuracy:.1%}"
)


# ============================================================
# 12. 对比结论
# ============================================================

print()

if sc_accuracy > single_accuracy:

    print(
        "结果：Self-Consistency 在本次实验中提高了准确率。"
    )

elif sc_accuracy == single_accuracy:

    print(
        "结果：两种方法在本次实验中的准确率相同。"
    )

else:

    print(
        "结果：Self-Consistency 在本次实验中没有提高准确率。"
    )