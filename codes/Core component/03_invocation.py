"""
示例3：调用与记忆 — Invocation with Checkpointer
=================================================

对应文档：LangChain-Agents 调用章节

本示例展示：
  - 如何使用 checkpointer 持久化对话历史
  - thread_id 的作用：追踪对话
  - context_schema 和 context 的用法：传递运行时配置
  - 多轮对话中 Agent 如何记住上下文

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.utils.uuid import uuid7
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


# ============================================================
# 第一部分：定义工具
# ============================================================
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。

    Args:
        city: 城市名称。
    """
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "广州": "雨天，30°C",
        "深圳": "阴天，27°C",
        "杭州": "晴天，26°C",
    }
    return weather_data.get(city, f"{city}：晴天，25°C")


@tool
def remember_fact(fact: str) -> str:
    """记住一个事实。

    Args:
        fact: 要记住的事实。
    """
    return f"好的，我记住了：{fact}"


# ============================================================
# 第二部分：定义上下文 Schema
# ----------------------------------------------------------
# context_schema 用于定义每次运行时传递的配置数据结构
# 工具和中间件可以通过 runtime.context 访问这些数据
#
# 常见用途：
#   - 传递用户 ID
#   - 传递 API 密钥
#   - 传递功能标志
# ============================================================
@dataclass
class UserContext:
    """用户上下文。"""

    user_id: str
    user_name: str


# ============================================================
# 第三部分：创建带 Checkpointer 的 Agent
# ----------------------------------------------------------
# checkpointer 用于持久化对话历史
# InMemorySaver 是内存存储，适合开发和测试
# 生产环境可以使用数据库存储（如 PostgreSQL、Redis 等）
#
# thread_id 用于区分不同的对话
# 同一个 thread_id 的多次调用会共享对话历史
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

# 创建内存检查点存储器
checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather, remember_fact],
    system_prompt="你是一个友好的助手。记住用户告诉你的信息，并在后续对话中使用。",
    checkpointer=checkpointer,
    context_schema=UserContext,  # 定义上下文格式
)


# ============================================================
# 第四部分：演示 thread_id 的作用
# ----------------------------------------------------------
# thread_id 是持久化对话历史的关键
# - 同一个 thread_id 的调用共享对话历史
# - 不同 thread_id 的调用互不影响
# ============================================================
def demo_thread_id():
    """演示 thread_id 如何追踪对话。"""
    print("=" * 60)
    print("示例3a：thread_id — 对话追踪")
    print("=" * 60)

    # 创建两个不同的 thread_id
    thread_1 = str(uuid7())
    thread_2 = str(uuid7())

    config_1 = {"configurable": {"thread_id": thread_1}}
    config_2 = {"configurable": {"thread_id": thread_2}}

    print(f"\n对话 1 的 thread_id: {thread_1[:8]}...")
    print(f"对话 2 的 thread_id: {thread_2[:8]}...")

    # === 对话 1 ===
    print("\n" + "-" * 40)
    print("对话 1：")
    print("-" * 40)

    # 第一轮：告诉 Agent 一个事实
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫小明，请记住我的名字"}]},
        config=config_1,
        context=UserContext(user_id="user-001", user_name="小明"),
    )
    print(f"用户：我叫小明，请记住我的名字")
    print(f"Agent：{result['messages'][-1].content}")

    # 第二轮：测试 Agent 是否记住了
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config=config_1,
        context=UserContext(user_id="user-001", user_name="小明"),
    )
    print(f"\n用户：我叫什么名字？")
    print(f"Agent：{result['messages'][-1].content}")

    # === 对话 2 ===
    print("\n" + "-" * 40)
    print("对话 2（不同的 thread_id，互不影响）：")
    print("-" * 40)

    # 在新对话中问同样的问题
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config=config_2,
        context=UserContext(user_id="user-002", user_name="未知"),
    )
    print(f"用户：我叫什么名字？")
    print(f"Agent：{result['messages'][-1].content}")
    print("（Agent 不知道，因为这是新的对话）")


# ============================================================
# 第五部分：演示 context 的用法
# ----------------------------------------------------------
# context 用于传递每次运行时的配置数据
# 与 thread_id（持久化对话）不同，context 是临时的运行时数据
# ============================================================
def demo_context():
    """演示 context 如何传递运行时配置。"""
    print("\n" + "=" * 60)
    print("示例3b：context — 运行时配置")
    print("=" * 60)

    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}

    # 不同的用户上下文
    users = [
        UserContext(user_id="alice-001", user_name="Alice"),
        UserContext(user_id="bob-002", user_name="Bob"),
    ]

    for user in users:
        print(f"\n当前用户：{user.user_name} (ID: {user.user_id})")
        print("-" * 40)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "你好，我是谁？"}]},
            config=config,
            context=user,
        )
        print(f"Agent：{result['messages'][-1].content}")


# ============================================================
# 第六部分：交互式多轮对话
# ============================================================
def main():
    print("=" * 60)
    print("示例3：调用与记忆 — 交互式多轮对话")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'reset' 重置对话\n")

    # 创建新的 thread_id
    thread_id = str(uuid7())
    config = {"configurable": {"thread_id": thread_id}}
    context = UserContext(user_id="demo-user", user_name="Demo")

    print(f"当前 thread_id: {thread_id[:8]}...\n")

    while True:
        user_input = input("你: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！ 👋")
            break

        if user_input.lower() == "reset":
            thread_id = str(uuid7())
            config = {"configurable": {"thread_id": thread_id}}
            print(f"\n已重置对话，新 thread_id: {thread_id[:8]}...\n")
            continue

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=context,
            )
            print(f"\nAgent: {result['messages'][-1].content}\n")
        except Exception as e:
            print(f"\n发生错误: {e}\n")


if __name__ == "__main__":
    import sys

    if "--demo-thread" in sys.argv:
        demo_thread_id()
    elif "--demo-context" in sys.argv:
        demo_context()
    elif "--demo" in sys.argv:
        demo_thread_id()
        demo_context()
    else:
        main()
