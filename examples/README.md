# LangChain 学习样例

基于 [LangChain Quickstart](https://python.langchain.com/docs/quickstart) 文档的代码示例，循序渐进地学习 LangChain Agent 开发。

## 环境准备

- Python 3.10+
- 一个 LLM 提供商的 API Key（OpenAI / Anthropic / Google / Ollama 等）

## 安装

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
```

## 设置 API Key

根据你使用的模型提供商，设置对应的环境变量：

```bash
# 任选其一
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export OLLAMA_API_KEY="your-key"
```

Windows PowerShell 使用 `$env:OPENAI_API_KEY="your-key"`。

## 示例列表

### 示例1：基础 Agent — 天气查询

```bash
python 01_basic_agent.py
```

**学习要点：**
- `create_agent` 的基本用法
- 如何用普通 Python 函数定义工具
- 模型字符串 `"provider:model_name"` 格式
- `agent.invoke` 的调用方式和返回结构

---

### 示例2：研究助手 Agent — URL 内容问答

```bash
python 02_research_agent.py
```

**学习要点：**
- 编写详细的 System Prompt 引导 Agent 行为
- `@tool` 装饰器创建带元数据的工具
- `init_chat_model` 配置模型参数（temperature、timeout、max_tokens）
- 工具调用链路：用户提问 → Agent 决定调用工具 → 工具返回结果 → Agent 生成回答

---

### 示例3：多轮对话 Agent — 交互式聊天

```bash
# 交互模式（命令行聊天）
python 03_chat_agent.py

# 演示模式（自动执行预设对话）
python 03_chat_agent.py --demo
```

**学习要点：**
- 通过维护 `messages` 列表实现对话记忆
- 多轮对话中 Agent 如何跨轮次使用工具
- 交互式循环的实现模式

---

## 切换模型

每个示例默认使用 OpenAI，可以通过修改代码中的 `model` 参数切换到其他提供商：

```python
# OpenAI
model="openai:gpt-5.5"

# Anthropic Claude
model="claude-sonnet-4-6"

# Google Gemini
model="google_genai:gemini-2.5-flash-lite"

# 本地 Ollama
model="ollama:llama3.1"
```

## 常见问题

**Q: 报错 `ModuleNotFoundError: No module named 'langchain'`**
A: 确认已激活虚拟环境并执行 `pip install -r requirements.txt`。

**Q: 报错 `AuthenticationError`**
A: 检查 API Key 环境变量是否正确设置。

**Q: 使用 Ollama 时连接失败**
A: 确认 Ollama 服务正在运行（`ollama serve`），模型已下载（`ollama pull llama3.1`）。
