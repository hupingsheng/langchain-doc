"""
示例2：结构化输出 — Structured Output
=====================================

对应文档：LangChain-Agents 结构化输出章节

本示例展示：
  - 如何使用 response_format 让 Agent 返回结构化数据
  - 使用 Pydantic BaseModel 定义输出格式
  - 从 result["structured_response"] 获取结构化结果

运行前请设置环境变量：
  在 .env 中配置 DEEPSEEK_API_KEY
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

load_dotenv()


# ============================================================
# 第一部分：定义输出格式（Pydantic BaseModel）
# ----------------------------------------------------------
# 使用 Pydantic 定义 Agent 返回的数据结构
# 模型会根据这个 schema 生成符合格式的输出
#
# Field 用于添加字段描述，帮助模型理解每个字段的含义
# ============================================================
class MovieReview(BaseModel):
    """电影评价的结构化输出格式。"""

    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，1-10 分")
    summary: str = Field(description="一句话总结")
    pros: list[str] = Field(description="优点列表")
    cons: list[str] = Field(description="缺点列表")
    recommend: bool = Field(description="是否推荐观看")


class WeatherInfo(BaseModel):
    """天气信息的结构化输出格式。"""

    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况，如晴、多云、雨等")
    humidity: int = Field(description="湿度百分比")
    suggestion: str = Field(description="出行建议")


# ============================================================
# 第二部分：定义工具
# ============================================================
@tool
def get_movie_info(movie_name: str) -> str:
    """获取电影信息。

    Args:
        movie_name: 电影名称。
    """
    # 模拟电影数据
    movies = {
        "流浪地球": "科幻电影，评分8.0，讲述人类带着地球逃离太阳系的故事",
        "肖申克的救赎": "经典剧情片，评分9.7，关于希望和自由的故事",
        "哪吒之魔童降世": "动画电影，评分8.5，改编自中国神话",
    }
    return movies.get(movie_name, f"未找到电影《{movie_name}》的信息")


@tool
def get_weather_data(city: str) -> str:
    """获取城市天气数据。

    Args:
        city: 城市名称。
    """
    # 模拟天气数据
    weather = {
        "北京": "晴天，25°C，湿度40%",
        "上海": "多云，28°C，湿度65%",
        "广州": "雨天，30°C，湿度80%",
    }
    return weather.get(city, f"{city}：晴天，25°C，湿度50%")


# ============================================================
# 第三部分：创建带结构化输出的 Agent
# ----------------------------------------------------------
# 关键参数：response_format=MovieReview 或 WeatherInfo
# Agent 会根据这个格式返回结构化数据
# ============================================================
model = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)


def create_movie_agent():
    """创建电影评价 Agent。"""
    return create_agent(
        model=model,
        tools=[get_movie_info],
        system_prompt="你是一个专业的电影评论家。请根据电影信息给出详细评价。",
        response_format=MovieReview,  # 指定输出格式
    )


def create_weather_agent():
    """创建天气信息 Agent。"""
    return create_agent(
        model=model,
        tools=[get_weather_data],
        system_prompt="你是一个天气助手。请根据天气数据提供出行建议。",
        response_format=WeatherInfo,  # 指定输出格式
    )


# ============================================================
# 第四部分：测试结构化输出
# ============================================================
def demo_movie():
    """演示电影评价的结构化输出。"""
    print("=" * 60)
    print("示例2a：电影评价 — 结构化输出")
    print("=" * 60)

    agent = create_movie_agent()

    question = "评价一下电影《流浪地球》"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # 获取结构化响应
    structured = result.get("structured_response")

    if structured:
        print("\n结构化输出：")
        print(f"  电影名称: {structured.title}")
        print(f"  评分: {structured.rating}/10")
        print(f"  总结: {structured.summary}")
        print(f"  优点: {', '.join(structured.pros)}")
        print(f"  缺点: {', '.join(structured.cons)}")
        print(f"  推荐: {'是' if structured.recommend else '否'}")
    else:
        print("\n普通文本输出：")
        print(result["messages"][-1].content)


def demo_weather():
    """演示天气信息的结构化输出。"""
    print("\n" + "=" * 60)
    print("示例2b：天气信息 — 结构化输出")
    print("=" * 60)

    agent = create_weather_agent()

    question = "广州今天天气怎么样？"
    print(f"\n问题：{question}")
    print("-" * 40)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    structured = result.get("structured_response")

    if structured:
        print("\n结构化输出：")
        print(f"  城市: {structured.city}")
        print(f"  温度: {structured.temperature}°C")
        print(f"  天气: {structured.condition}")
        print(f"  湿度: {structured.humidity}%")
        print(f"  建议: {structured.suggestion}")
    else:
        print("\n普通文本输出：")
        print(result["messages"][-1].content)


def main():
    demo_movie()
    demo_weather()


if __name__ == "__main__":
    main()
