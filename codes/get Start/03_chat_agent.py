"""
示例3：多轮对话 Agent — 交互式聊天
====================================

对应文档：LangChain Quickstart 中提到的「对话记忆」概念

本示例展示：
  - 如何通过维护 messages 列表实现多轮对话
  - Agent 如何记住上下文，实现连续追问
  - 交互式命令行聊天循环

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# ============================================================
# 工具定义
# ----------------------------------------------------------
# 复用天气工具，演示多轮对话中 Agent 如何跨轮次使用工具
# ============================================================
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息。"""
    # 模拟不同城市的天气数据
    weather_data = {
        "san francisco": "晴天, 18°C",
        "new york": "多云, 22°C",
        "london": "雨天, 14°C",
        "tokyo": "晴朗, 28°C",
        "beijing": "有风, 20°C",
        "旧金山": "晴天, 18°C",
        "纽约": "多云, 22°C",
        "伦敦": "雨天, 14°C",
        "东京": "晴朗, 28°C",
        "北京": "有风, 20°C",
    }
    city_lower = city.lower().strip()
    return weather_data.get(city_lower, f"晴天, 25°C，位于{city}")


def get_recommendation(activity: str, weather: str) -> str:
    """根据天气状况获取活动建议。"""
    if "雨" in weather or "rain" in weather.lower():
        return f"鉴于下雨天气，我建议进行室内{activity}。可以去博物馆或咖啡馆！"
    elif "晴" in weather or "sunny" in weather.lower() or "clear" in weather.lower():
        return f"天气很适合户外{activity}！享受阳光吧！"
    else:
        return f"以当前天气来看，室内和室外{activity}都可以。"


# ============================================================
# 创建 Agent
# ----------------------------------------------------------
# 使用 ChatDeepSeek 实例化模型对象
#   - model：指定模型名称，"deepseek-v4-pro" 是 DeepSeek 的旗舰推理模型
#   - api_key：从环境变量安全读取，不在代码中硬编码
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

agent = create_agent(
    model=model,
    tools=[get_weather, get_recommendation],
    system_prompt=(
        "你是一个友好的旅行和天气助手。"
        "帮助用户根据天气状况规划活动。"
        "在给出建议之前，务必先查询天气情况。"
    ),
)


# ============================================================
# 多轮对话循环
# ----------------------------------------------------------
# 核心思路：维护一个 messages 列表，每轮对话都将用户输入
# 追加到列表中，然后将完整列表传给 agent.invoke。
# Agent 的每次回复也追加到列表中，实现上下文记忆。
# ============================================================
def main():
    print("=" * 60)
    print("示例3：多轮对话 Agent — 交互式聊天")
    print("=" * 60)
    print("输入 'quit' 或 'exit' 退出\n")

    # 对话历史：初始为空
    messages = []

    while True:
        # 获取用户输入
        user_input = input("你: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n再见！ 👋")
            break

        # 将用户消息追加到历史
        messages.append({"role": "user", "content": user_input})

        try:
            # 调用 Agent，传入完整对话历史
            result = agent.invoke({"messages": messages})

            # 获取 Agent 的最终回复
            ai_message = result["messages"][-1]
            ai_reply = ai_message.content

            # 打印回复
            print(f"\nAgent: {ai_reply}\n")

            # 更新对话历史：用 Agent 返回的完整 messages 替换
            # 这样 Agent 内部的工具调用和结果也会被保留
            messages = result["messages"]

        except Exception as e:
            print(f"\n发生错误: {e}\n")
            # 出错时移除最后一条用户消息，避免重复发送
            messages.pop()


# ============================================================
# 非交互式测试（可直接运行查看效果）
# ============================================================
def demo():
    """演示一次非交互式的多轮对话。"""
    print("=" * 60)
    print("示例3 演示模式：多轮对话")
    print("=" * 60)

    messages = []

    # 模拟三轮对话
    conversation = [
        "东京的天气怎么样？",
        "北京呢？",
        "根据两个城市的天气，哪个更适合户外活动？",
    ]

    for i, user_msg in enumerate(conversation, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"你: {user_msg}")

        messages.append({"role": "user", "content": user_msg})

        result = agent.invoke({"messages": messages})
        ai_reply = result["messages"][-1].content

        print(f"Agent: {ai_reply}")
        messages = result["messages"]

    print("\n" + "=" * 60)
    print("对话结束")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        demo()
    else:
        main()
