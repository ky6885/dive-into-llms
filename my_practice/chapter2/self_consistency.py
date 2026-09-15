import os
import re
from collections import Counter
from openai import OpenAI


# ============================================================
# 1. API Key
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
A store sells a jacket for $250.

The jacket is first discounted by 20%.
Then a 10% tax is added to the discounted price.

What is the final price of the jacket?
"""


prompt = f"""
Solve the following problem.

{question}

Think through the problem carefully.

End your answer with exactly:

FINAL: <number>
"""


# ============================================================
# 3. 提取最终数字
# ============================================================

def extract_answer(text):
    matches = re.findall(
        r"FINAL:\s*\$?\s*(-?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )

    if not matches:
        return None

    return float(matches[-1])


# ============================================================
# 4. 单次调用
# ============================================================

def ask_once():
    response = client.chat.completions.create(
        model="deepseek-flash",

        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],

        # 增加随机性
        temperature=0.8,

        # 关键：
        # 当前 DeepSeek 默认开启 thinking mode，
        # thinking mode 下 temperature 不生效。
        # 因此这里显式关闭。
        extra_body={
            "thinking": {
                "type": "disabled"
            }
        },
    )

    return response.choices[0].message.content


# ============================================================
# 5. 多次采样
# ============================================================

num_samples = 5

answers = []


for i in range(num_samples):

    print()
    print("=" * 60)
    print(f"Sample {i + 1}")
    print("=" * 60)

    output = ask_once()

    print(output)

    answer = extract_answer(output)

    print()
    print("提取答案：", answer)

    if answer is not None:
        answers.append(answer)


# ============================================================
# 6. Self-Consistency 多数投票
# ============================================================

print()
print("=" * 60)
print("SELF-CONSISTENCY")
print("=" * 60)

print("所有提取到的答案：")
print(answers)


if not answers:
    print("没有成功解析任何答案。")

else:

    counts = Counter(answers)

    print()
    print("答案计数：")
    print(counts)

    final_answer, votes = counts.most_common(1)[0]

    print()
    print("多数投票结果：", final_answer)
    print("票数：", votes, "/", len(answers))