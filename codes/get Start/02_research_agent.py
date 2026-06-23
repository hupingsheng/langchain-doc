"""
示例2：研究助手 Agent — URL 内容抓取与问答
==========================================

对应文档：LangChain Quickstart 第4节「构建真实世界 Agent」

本示例展示：
  - 如何编写详细的 System Prompt 来引导 Agent 行为
  - 如何使用 @tool 装饰器创建带元数据的工具
  - 如何使用 ChatDeepSeek 配置 DeepSeek 模型
  - 如何让 Agent 从 URL 加载内容并回答问题

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import gzip
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# ============================================================
# 第一步：定义 System Prompt
# ----------------------------------------------------------
# System Prompt 决定了 Agent 的角色、能力和行为边界。
# 写得越具体，Agent 的行为越可控。
# ============================================================
SYSTEM_PROMPT = """你是一个研究助手，能够从 URL 抓取并分析文本内容。

## 你的能力

- `fetch_text_from_url`：从指定的 URL 加载文档文本。
  当用户询问某个网页的内容时使用此工具。

## 规则

1. 在回答关于网页内容的问题之前，必须始终使用 fetch_text_from_url 工具。
2. 不要猜测或编造信息 —— 只能基于工具返回的实际结果来回答。
3. 除非用户要求详细分析，否则请简洁地总结发现。
4. 如果 URL 无效或无法访问，请清楚地说明错误。
"""


# ============================================================
# 第二步：定义工具
# ----------------------------------------------------------
# @tool 装饰器会自动：
#   - 将函数的 docstring 作为工具描述
#   - 将参数名和类型注解作为工具 schema
#   - 使函数可以被 Agent 识别和调用
#
# 工具文档质量直接影响模型调用工具的准确性！
# ============================================================
@tool
def fetch_text_from_url(url: str) -> str:
    """从指定的 URL 抓取文本内容。

    返回网页的解码后文本内容。
    当用户询问某个特定 URL 的内容时使用此工具。
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; langchain-quickstart/1.0)",
            "Accept-Encoding": "gzip, deflate",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
            # 检查是否使用了 gzip 压缩
            encoding = resp.headers.get("Content-Encoding", "")
            if encoding == "gzip":
                raw = gzip.decompress(raw)
    except urllib.error.URLError as e:
        return f"抓取失败: {e}"
    except Exception as e:
        return f"发生意外错误: {e}"

    # 解码为文本，限制长度避免上下文溢出
    text = raw.decode("utf-8", errors="replace")
    if len(text) > 20000:
        text = text[:20000] + "\n\n... [内容已截断]"
    return text


# ============================================================
# 第三步：配置模型
# ----------------------------------------------------------
# 使用 ChatDeepSeek 直接实例化模型对象
#   - model：指定模型名称，"deepseek-v4-pro" 是 DeepSeek 的旗舰推理模型
#   - api_key：从环境变量安全读取，不在代码中硬编码
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# 第四步：创建并运行 Agent
# ============================================================
agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
)


def main():
    print("=" * 60)
    print("示例2：研究助手 Agent — URL 内容抓取与问答")
    print("=" * 60)

    # 测试问题：让 Agent 抓取一个网页并总结
    question = (
        "请抓取 https://python.org 的内容，"
        "并用 2 到 3 句话总结一下 Python 是什么。"
    )
    print(f"\n用户提问：{question}\n")
    print("Agent 正在思考和调用工具...\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # 打印最终回答
    print("-" * 40)
    print("Agent 回答：")
    print("-" * 40)
    print(result["messages"][-1].content)

    # 打印消息链路概览
    print("\n" + "-" * 40)
    print("消息链路概览：")
    print("-" * 40)
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_names = [tc["name"] for tc in msg.tool_calls]
            print(f"[{i}] {role} → 调用工具: {tool_names}")
        elif hasattr(msg, "name") and msg.name:
            print(f"[{i}] ToolMessage ({msg.name}): {str(msg.content)[:80]}...")
        else:
            content = msg.content if hasattr(msg, "content") else str(msg)
            print(f"[{i}] {role}: {content[:100]}...")


if __name__ == "__main__":
    main()
