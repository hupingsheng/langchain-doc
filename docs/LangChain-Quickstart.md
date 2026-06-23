# LangChain Python 快速入门指南

> 原文来源：[LangChain Docs - Quickstart](https://python.langchain.com/docs/quickstart)

## 1. 安装依赖

```bash
pip install -U langchain deepagents
```

## 2. 设置 API Key

支持多种模型提供商，任选其一：

| 提供商 | 环境变量 |
|--------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GOOGLE_API_KEY` |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Fireworks | `FIREWORKS_API_KEY` |
| Baseten | `BASETEN_API_KEY` |
| Ollama (本地) | `OLLAMA_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT_NAME` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION` |
| HuggingFace | `HUGGINGFACEHUB_API_TOKEN` |

示例：

```bash
export OPENAI_API_KEY="your-api-key"
```

## 3. 构建基础 Agent

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",      # 或 "claude-sonnet-4-6" 等
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
```

核心概念：

- **`create_agent`** — 一步创建 Agent，无需手动编排图结构
- **Tools** — 普通 Python 函数即可作为工具，Agent 自动识别并调用
- **模型字符串** — `"provider:model_name"` 格式，支持所有主流 LLM

## 4. 构建真实世界 Agent（研究助手）

这是一个更完整的示例，构建一个可以回答文本文件问题的研究 Agent。

### 4.1 定义 System Prompt

System Prompt 定义 Agent 的角色和行为，要具体且可操作：

```python
SYSTEM_PROMPT = """You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
  Do not guess line counts or positions—ground them in tool results from the saved file.
"""
```

### 4.2 创建工具（`@tool` 装饰器）

工具让模型与外部系统交互，`@tool` 装饰器自动添加元数据：

```python
import urllib.error
import urllib.request

from langchain.tools import tool


@tool
def fetch_text_from_url(url: str) -> str:
    """Fetch the document from a URL."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; quickstart-research/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text
```

> **提示**：工具应该有良好的文档说明，名称、描述和参数名都会成为模型 prompt 的一部分。

### 4.3 配置模型

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai:gpt-5.5",
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
)
```

### 4.4 涉及的高级概念

| 概念 | 说明 |
|------|------|
| **System Prompt** | 定义 Agent 角色和行为，要具体可操作 |
| **Tools** | 让模型与外部系统交互，`@tool` 装饰器自动添加元数据 |
| **Model Config** | `temperature`、`timeout`、`max_tokens` 等参数调优 |
| **对话记忆** | 支持类聊天式的多轮对话 |
| **Deep Agents** | 内置高级特性（如子代理、文件操作等） |

## 5. 调试与追踪

推荐使用 **LangSmith** 追踪 Agent 内部行为，可以可视化每一步的推理和工具调用过程。

参考 [LangSmith 追踪快速入门](https://docs.smith.langchain.com/) 进行设置。

---

## 总结

LangChain 的核心流程：

```
安装 → 设置 API Key → 定义工具 → 创建 Agent → 调用
```

API 非常简洁，几行代码即可构建功能完整的 AI Agent。
