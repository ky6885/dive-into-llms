# Chapter 2：大模型 Prompting 与 Chain-of-Thought 实验

本目录记录上海交通大学《动手学大模型》Chapter 2 的个人学习实验。

模型：

- DeepSeek API
- deepseek-flash

环境：

- Ubuntu 22.04
- Python 3.11
- Conda 环境：dive-llms

---

## 1. DeepSeek API

文件：

`deepseek_test.py`

目标：

学习通过 Python 调用大语言模型 API。

流程：

Python
→ OpenAI Compatible SDK
→ DeepSeek API
→ deepseek-flash
→ 模型输出

---

## 2. Zero-shot Prompting

文件：

`zero_shot.py`

不给模型示例，直接提供任务。

实验答案：

24

---

## 3. Zero-shot Chain-of-Thought

文件：

`zero_shot_cot.py`

在问题中加入：

`Let's think step by step.`

模型输出更明确的分步骤推理过程。

---

## 4. Few-shot Prompting

文件：

`few_shot.py`

在 Prompt 中提供多个示例。

观察：

模型会学习示例中的任务形式以及输出格式。

例如示例只输出：

A: 6

模型也倾向于只输出：

A: 24

---

## 5. Few-shot Chain-of-Thought

文件：

`few_shot_cot.py`

示例中同时提供：

问题
→ 推理过程
→ 最终答案

模型会模仿示例中的推理形式。

---

## 6. 四种 Prompt 方法对比

文件：

`compare_prompting.py`

对比：

- Zero-shot
- Zero-shot CoT
- Few-shot
- Few-shot CoT

观察：

Few-shot 不仅可以告诉模型任务是什么，也可以影响输出格式。

---

## 7. 自动评测

文件：

`evaluate_prompting.py`

实现：

- 多题测试
- FINAL 答案提取
- 自动判断正确性
- Accuracy 统计

实验结果：

- Zero-shot：100%
- Zero-shot CoT：100%
- Few-shot：100%
- Few-shot CoT：100%

当前测试题较简单，存在明显的 ceiling effect。

---

## 8. Self-Consistency

文件：

`self_consistency.py`

流程：

同一道问题
→ 多次采样
→ 提取多个答案
→ Counter 统计
→ 多数投票

实验中 5 次均得到：

220

最终多数投票：

220

---

## 9. Single Sample vs Self-Consistency

文件：

`compare_self_consistency.py`

测试 6 道多步数学题。

结果：

Single Sample：

6 / 6 = 100%

Self-Consistency：

6 / 6 = 100%

结论：

当前题目对于模型仍较简单，因此没有观察到
Self-Consistency 带来的准确率提升。

---

## 10. 错误 Few-shot 示例实验

文件：

`wrong_examples.py`

比较：

- Zero-shot
- Correct Few-shot
- Wrong Few-shot

结果：

三种方法均得到正确答案 16。

说明在当前简单数学任务中，
模型没有被少量错误示例明显污染。

但这个结论不能推广到所有任务。

---

## 学习总结

本章主要理解了：

1. API 调用
2. Zero-shot
3. Few-shot
4. Chain-of-Thought
5. In-context Learning
6. Temperature
7. Self-Consistency
8. Majority Voting
9. Accuracy
10. 自动 Evaluator
11. Few-shot 示例质量对模型行为的影响

同时练习了：

- git clone
- git status
- git branch
- git switch
- git add
- git commit
- git log