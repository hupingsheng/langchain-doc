"""
示例5：中间件 — Middleware
=========================

对应文档：LangChain-Agents 配置 Harness 章节

本示例展示：
  - 中间件的基本概念和用法
  - 几种常见的中间件类型
  - 如何组合多个中间件

注意：部分中间件（如 deepagents 相关）需要额外安装包

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
# 工具定义
# ============================================================
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件。

    Args:
        to: 收件人邮箱。
        subject: 邮件主题。
        body: 邮件正文。
    """
    return f"邮件已发送给 {to}，主题：{subject}"


@tool
def read_file(path: str) -> str:
    """读取文件内容。

    Args:
        path: 文件路径。
    """
    # 模拟文件读取
    return f"文件 {path} 的内容：这是一段示例文本..."


@tool
def write_file(path: str, content: str) -> str:
    """写入文件。

    Args:
        path: 文件路径。
        content: 要写入的内容。
    """
    return f"已写入文件 {path}，共 {len(content)} 字符"


@tool
def search(query: str) -> str:
    """搜索信息。

    Args:
        query: 搜索关键词。
    """
    return f"搜索结果：关于「{query}」的相关信息..."


# ============================================================
# 模型配置
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# 示例 5a：容错中间件（Fault Tolerance）
# ----------------------------------------------------------
# ModelRetryMiddleware：模型调用失败时自动重试
# ToolRetryMiddleware：工具调用失败时自动重试
#
# 注意：需要确保 langchain.agents.middleware 中有这些中间件
# 如果没有，可以跳过这个示例
# ============================================================
def demo_fault_tolerance():
    """演示容错中间件。"""
    print("=" * 60)
    print("示例5a：容错中间件")
    print("=" * 60)

    try:
        from langchain.agents.middleware import (
            ModelRetryMiddleware,
            ToolRetryMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[search],
            system_prompt="你是一个搜索助手。",
            middleware=[
                ModelRetryMiddleware(max_retries=3),  # 模型调用最多重试 3 次
                ToolRetryMiddleware(max_retries=2),   # 工具调用最多重试 2 次
            ],
        )

        print("\n容错中间件配置成功！")
        print("- ModelRetryMiddleware: max_retries=3")
        print("- ToolRetryMiddleware: max_retries=2")

        # 测试
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "搜索 LangChain"}]}
        )
        print(f"\nAgent 回复：{result['messages'][-1].content}")

    except ImportError as e:
        print(f"\n中间件未找到，跳过此示例: {e}")
        print("提示：请确保已安装最新版本的 langchain")


# ============================================================
# 示例 5b：护栏中间件（Guardrails）
# ----------------------------------------------------------
# PIIMiddleware：检测和脱敏个人身份信息（PII）
# 可以检测邮箱、电话号码、身份证号等敏感信息
# ============================================================
def demo_guardrails():
    """演示护栏中间件。"""
    print("\n" + "=" * 60)
    print("示例5b：护栏中间件（PII 检测）")
    print("=" * 60)

    try:
        from langchain.agents.middleware import PIIMiddleware

        agent = create_agent(
            model=model,
            tools=[send_email],
            system_prompt="你是一个邮件助手。帮助用户发送邮件。",
            middleware=[
                PIIMiddleware("email"),  # 检测邮箱地址
            ],
        )

        print("\n护栏中间件配置成功！")
        print("- PIIMiddleware: 检测邮箱地址")

        # 测试
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "发邮件给 test@example.com"}]}
        )
        print(f"\nAgent 回复：{result['messages'][-1].content}")

    except ImportError as e:
        print(f"\n中间件未找到，跳过此示例: {e}")


# ============================================================
# 示例 5c：引导中间件（Steering / Human-in-the-Loop）
# ----------------------------------------------------------
# HumanInTheLoopMiddleware：在执行某些工具前暂停，等待人工确认
# 适用于：
#   - 破坏性操作（删除文件、发送邮件）
#   - 昂贵的 API 调用
#   - 需要人工判断的操作
#
# 注意：这个中间件在实际应用中需要配合 UI 或 API 使用
# 在命令行中无法真正"暂停等待"
# ============================================================
def demo_steering():
    """演示引导中间件（概念演示）。"""
    print("\n" + "=" * 60)
    print("示例5c：引导中间件（Human-in-the-Loop）")
    print("=" * 60)

    try:
        from langchain.agents.middleware import HumanInTheLoopMiddleware

        agent = create_agent(
            model=model,
            tools=[write_file, send_email],
            system_prompt="你是一个助手。可以写入文件和发送邮件。",
            middleware=[
                HumanInTheLoopMiddleware(
                    interrupt_on={
                        "write_file": True,   # 写入文件前需要确认
                        "send_email": True,   # 发送邮件前需要确认
                    }
                ),
            ],
        )

        print("\n引导中间件配置成功！")
        print("- 写入文件前需要人工确认")
        print("- 发送邮件前需要人工确认")
        print("\n注意：实际应用中需要配合 UI 或 API 实现暂停等待功能")

        # 这里只是演示配置，不实际调用
        print("\n（此示例仅演示配置，不实际调用工具）")

    except ImportError as e:
        print(f"\n中间件未找到，跳过此示例: {e}")


# ============================================================
# 示例 5d：自定义中间件
# ----------------------------------------------------------
# 可以创建自定义中间件来实现特定逻辑
# 中间件通过钩子函数在 Agent 循环的不同阶段执行
# ============================================================
def demo_custom_middleware():
    """演示自定义中间件的概念。"""
    print("\n" + "=" * 60)
    print("示例5d：自定义中间件（概念说明）")
    print("=" * 60)

    print("""
自定义中间件可以在以下时机执行：

1. before_model：模型调用前
   - 修改提示词
   - 添加日志
   - 检查输入

2. after_model：模型调用后
   - 修改输出
   - 记录响应时间
   - 过滤内容

3. before_tool：工具调用前
   - 验证参数
   - 权限检查
   - 缓存查询

4. after_tool：工具调用后
   - 处理结果
   - 添加日志
   - 数据转换

自定义中间件示例结构：

    from langchain.agents.middleware import middleware

    @middleware
    def my_custom_middleware(state, runtime):
        # 在模型调用前执行
        if runtime.phase == "before_model":
            print("即将调用模型...")

        # 在工具调用前执行
        elif runtime.phase == "before_tool":
            tool_name = runtime.tool_name
            print(f"即将调用工具: {tool_name}")

        return state  # 返回修改后的状态
""")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("示例5：中间件 — Middleware")
    print("=" * 60)

    # 运行各个示例
    demo_fault_tolerance()
    demo_guardrails()
    demo_steering()
    demo_custom_middleware()

    print("\n" + "=" * 60)
    print("所有中间件示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
