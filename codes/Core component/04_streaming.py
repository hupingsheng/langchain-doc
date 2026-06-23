"""
示例4：流式输出 — Streaming
===========================

对应文档：LangChain-Agents 流式输出章节

本示例展示：
  - 使用 stream_events 实现实时输出
  - 如何区分用户消息、Agent 回复、工具调用
  - 流式输出在 UI 中的应用场景

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# ============================================================
# 工具定义
# ============================================================
@tool
def search_news(topic: str) -> str:
    """搜索新闻。

    Args:
        topic: 要搜索的新闻主题。
    """
    # 模拟搜索结果
    news = {
        "AI": "最新进展：GPT-5 发布，性能大幅提升。同时，开源模型也在快速发展。",
        "科技": "苹果公司发布新款 iPhone，搭载最新芯片。",
        "天气": "全国大部分地区晴好，南方有降雨。",
    }
    return news.get(topic, f"关于{topic}的最新新闻正在更新中...")


@tool
def summarize(text: str) -> str:
    """总结文本内容。

    Args:
        text: 要总结的文本。
    """
    # 简单模拟总结
    return f"总结：{text[:50]}...（这是简短版本）"


# ============================================================
# 创建 Agent
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

agent = create_agent(
    model=model,
    tools=[search_news, summarize],
    system_prompt="你是一个新闻助手。帮助用户搜索和总结新闻。",
)


# ============================================================
# 流式输出演示
# ----------------------------------------------------------
# stream_events 返回事件流，每个事件包含当前状态的快照
# 可以实时显示：
#   - 用户消息
#   - Agent 的思考过程
#   - 工具调用
#   - 工具结果
#   - 最终回复
# ============================================================
def demo_streaming():
    """演示流式输出。"""
    print("=" * 60)
    print("示例4：流式输出 — Streaming")
    print("=" * 60)

    question = "搜索 AI 相关新闻并总结一下"
    print(f"\n用户提问：{question}")
    print("\n" + "-" * 40)
    print("实时输出：")
    print("-" * 40 + "\n")

    # 使用 stream_events 获取事件流
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": question}]},
        version="v3",
    )

    # 遍历事件流
    for event in stream:
        event_type = event.get("event", "")

        # 处理不同类型的流式事件
        if event_type == "on_chat_model_stream":
            # 模型生成的文本块（流式输出）
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                # 流式打印每个文本块
                print(chunk.content, end="", flush=True)

        elif event_type == "on_tool_start":
            # 工具开始执行
            tool_name = event.get("name", "")
            print(f"\n\n[调用工具: {tool_name}]", flush=True)

        elif event_type == "on_tool_end":
            # 工具执行完成
            tool_output = event.get("data", {}).get("output", "")
            if tool_output:
                output_preview = str(tool_output)[:100]
                print(f"[工具结果: {output_preview}...]", flush=True)

    print("\n")
    print("-" * 40)
    print("流式输出完成")
    print("-" * 40)


# ============================================================
# 简化版流式输出
# ----------------------------------------------------------
# 如果不需要那么细粒度的控制，可以使用更简单的流式接口
# ============================================================
def demo_simple_streaming():
    """简化版流式输出演示。"""
    print("\n" + "=" * 60)
    print("简化版流式输出")
    print("=" * 60)

    question = "今天有什么科技新闻？"
    print(f"\n用户提问：{question}")
    print("\n实时输出：\n")

    # 简化的流式输出
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": question}]},
        version="v3",
    )

    for snapshot in stream.values:
        # 每个快照包含该点的完整状态
        if not snapshot.get("messages"):
            continue

        latest_message = snapshot["messages"][-1]

        # 区分消息类型
        if isinstance(latest_message, HumanMessage):
            if latest_message.content:
                print(f"[用户]: {latest_message.content}")
        elif isinstance(latest_message, AIMessage):
            # 显示工具调用
            if hasattr(latest_message, "tool_calls") and latest_message.tool_calls:
                tool_names = [tc["name"] for tc in latest_message.tool_calls]
                print(f"[Agent 调用工具]: {tool_names}")
            # 显示最终回复
            elif latest_message.content:
                print(f"[Agent 回复]: {latest_message.content}")


# ============================================================
# 主函数
# ============================================================
def main():
    # 运行详细版流式输出
    demo_streaming()

    # 运行简化版流式输出
    demo_simple_streaming()


if __name__ == "__main__":
    main()
