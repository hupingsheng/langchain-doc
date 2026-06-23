

# ============================================================
# 01_basic_agent.py - 使用 LangChain 创建基础 Agent（天气查询）
# ============================================================
# 学习目标：
#   1. 了解如何用 create_agent 一步创建 Agent
#   2. 掌握工具（Tool）的定义方式 —— 普通 Python 函数即可
#   3. 理解 Agent 的调用流程：用户提问 → 模型思考 → 调用工具 → 返回结果
#   4. 学会使用 DeepSeek 模型 + LangChain Agent 的组合
#
# 核心概念：
#   Agent = 模型 + 工具 + 系统提示词
#   模型负责"思考"，工具负责"执行"，系统提示词负责"引导"。
#   LangChain 的 create_agent 将这三者组合在一起，
#   自动实现"模型决定调用哪个工具 → 执行工具 → 将结果返回给模型"的完整循环。
#
# 前置准备：
#   - pip install langchain langchain-deepseek python-dotenv
#   - 在 .env 中配置 DEEPSEEK_API_KEY
# ============================================================


# ----------------------
# Step 1: 导入核心模块
# ----------------------
# ChatDeepSeek：LangChain 为 DeepSeek 提供的专用适配器类
#   - 内置 DeepSeek 的 API 地址，无需手动指定 base_url
#   - 与 LangChain 的 Agent 系统无缝集成
from langchain_deepseek import ChatDeepSeek

# create_agent：LangChain 提供的 Agent 工厂函数
#   - 传入模型、工具列表、系统提示词，一步创建完整的 Agent
#   - 内部基于 LangGraph 实现 ReAct（推理+行动）循环
from langchain.agents import create_agent

# dotenv：从 .env 文件加载环境变量（API Key 等敏感信息）
from dotenv import load_dotenv

# os：通过 os.getenv() 读取系统环境变量
import os


# ----------------------
# Step 2: 加载环境变量
# ----------------------
# 将 .env 文件中的配置项注入到环境变量中，
# 后续通过 os.getenv() 即可安全读取，避免在代码里硬编码密钥。
load_dotenv()


# ============================================================
# Step 3: 定义工具（Tools）
# ============================================================
# 工具就是普通的 Python 函数，LangChain 会自动将其转换为模型可调用的格式。
#
# 关键规则：
#   - 函数的 docstring（"""..."""）会作为工具描述传给模型
#     模型根据这个描述来判断"什么时候该调用这个工具"
#   - 参数名和类型注解会成为工具的 schema
#     模型根据这些来生成正确的调用参数
#   - 所以文档写得越清晰，模型调用工具就越准确
#
# 这里用固定返回值模拟，实际开发中可以接入真实天气 API
# ============================================================
def get_weather(city: str) -> str:
    """获取指定城市的当前天气信息。

    Args:
        city: 要查询天气的城市名称。
    """
    return f"{city}的天气总是晴空万里！"


def get_temperature(city: str) -> str:
    """获取指定城市的当前温度。

    Args:
        city: 要查询温度的城市名称。
    """
    return f"{city}的温度是 25°C。"


# ============================================================
# Step 4: 初始化模型
# ============================================================
# 使用 ChatDeepSeek 直接实例化模型对象
#   - model：指定模型名称，"deepseek-v4-pro" 是 DeepSeek 的旗舰推理模型
#   - api_key：从环境变量安全读取，不在代码中硬编码
#
# 与 04langchain1.0模型调用.py 中的方式三保持一致
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


# ============================================================
# Step 5: 创建 Agent
# ============================================================
# create_agent 将模型、工具、系统提示词组合成一个完整的 Agent
#
# 参数说明：
#   model：         模型实例（上一步创建的 ChatDeepSeek 对象）
#   tools：         工具列表，Agent 会根据用户问题自动决定调用哪个
#   system_prompt： 系统提示词，定义 Agent 的角色和行为规范
#
# 与直接调用 model.invoke() 的区别：
#   - model.invoke()：模型只能"回答"，无法调用外部工具
#   - agent.invoke()：模型可以"思考 + 调用工具 + 再思考"，具备行动能力
# ============================================================
agent = create_agent(
    model=model,
    tools=[get_weather, get_temperature],
    system_prompt="你是一个乐于助人的天气助手。在回答用户问题之前，务必先使用工具获取真实数据。",
)


# ============================================================
# Step 6: 调用 Agent
# ============================================================
# agent.invoke() 是 LangChain 统一的调用接口
#   - 传入字典 {"messages": [消息列表]}
#   - 返回字典 {"messages": [完整消息链路]}
#
# 消息链路包含整个推理过程：
#   HumanMessage（用户提问）
#   → AIMessage（模型决定调用工具）
#   → ToolMessage（工具返回结果）
#   → AIMessage（模型根据工具结果生成最终回答）
# ============================================================
def main():
    print("=" * 60)
    print("01_basic_agent - 基础 Agent 天气查询")
    print("=" * 60)

    # 测试问题
    question = "旧金山的天气和温度怎么样？"
    print(f"\n用户提问：{question}\n")

    # 调用 Agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # ----------------------
    # 打印完整消息链路
    # ----------------------
    # 通过遍历 messages 可以看到 Agent 的完整"思考过程"
    print("-" * 40)
    print("完整消息链路：")
    print("-" * 40)
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        content = msg.content if hasattr(msg, "content") else str(msg)
        print(f"[{i}] {role}: {content[:200]}")

    # ----------------------
    # 打印最终回答
    # ----------------------
    # result["messages"][-1] 是 Agent 的最终回复（AIMessage）
    # .content 属性包含回复的纯文本内容
    print("\n" + "-" * 40)
    print("最终回答：")
    print("-" * 40)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
