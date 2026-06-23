"""
示例1：核心组件 — Model + Tools + System Prompt
================================================

对应文档：LangChain-Agents 核心组件章节

本示例展示：
  - 如何使用 create_agent 创建基础 Agent
  - 模型（Model）的三种配置方式
  - 工具（Tools）的定义和 @tool 装饰器
  - 系统提示词（System prompt）的作用

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# ============================================================
# 第一部分：定义工具（Tools）
# ----------------------------------------------------------
# 工具是 Agent 与外部世界交互的能力。
# 可以是普通 Python 函数，也可以用 @tool 装饰器添加更详细的描述。
#
# 关键点：
#   - 函数的 docstring 会作为工具描述传给模型
#   - 参数名和类型注解会成为工具的 schema
#   - 模型根据这些信息判断何时调用工具、如何传参
# ============================================================
@tool
def search(query: str) -> str:
    """搜索信息并返回结果。

    Args:
        query: 要搜索的关键词。
    """
    # 模拟搜索结果
    return f"关于「{query}」的搜索结果：LangChain 是一个用于构建 LLM 应用的框架。"


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。

    Args:
        expression: 要计算的数学表达式，如 "2 + 3 * 4"。
    """
    try:
        result = eval(expression)  # 注意：生产环境应使用更安全的计算方式
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 第二部分：配置模型（Model）
# ----------------------------------------------------------
# 模型配置有三种方式：
#   1. 字符串格式：model="provider:model_name"
#   2. 模型实例：传入初始化的模型对象
#   3. 使用 init_chat_model：统一配置接口
#
# 这里使用方式 2，与 03_chat_agent.py 保持一致
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# 第三部分：创建 Agent
# ----------------------------------------------------------
# create_agent 将模型、工具、系统提示词组合成完整的 Agent
#
# 参数说明：
#   - model: 模型实例或字符串
#   - tools: 工具列表
#   - system_prompt: 系统提示词，定义 Agent 的角色和行为规范
# ============================================================
agent = create_agent(
    model=model,
    tools=[search, calculator],
    system_prompt="你是一个有帮助的助手。回答要简洁准确。可以使用搜索工具查询信息，使用计算器进行数学运算。",
)


# ============================================================
# 第四部分：测试 Agent
# ============================================================
def main():
    print("=" * 60)
    print("示例1：核心组件 — Model + Tools + System Prompt")
    print("=" * 60)

    # 测试问题 1：搜索
    question1 = "帮我搜索一下 LangChain 是什么"
    print(f"\n问题 1：{question1}")
    print("-" * 40)

    result1 = agent.invoke(
        {"messages": [{"role": "user", "content": question1}]}
    )
    print(f"回答：{result1['messages'][-1].content}")

    # 测试问题 2：计算
    question2 = "计算 123 * 456 + 789"
    print(f"\n问题 2：{question2}")
    print("-" * 40)

    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": question2}]}
    )
    print(f"回答：{result2['messages'][-1].content}")

    # 测试问题 3：组合使用
    question3 = "先搜索 Python 的历史，然后计算 2024 - 1991"
    print(f"\n问题 3：{question3}")
    print("-" * 40)

    result3 = agent.invoke(
        {"messages": [{"role": "user", "content": question3}]}
    )
    print(f"回答：{result3['messages'][-1].content}")


if __name__ == "__main__":
    main()
