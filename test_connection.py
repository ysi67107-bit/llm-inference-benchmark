from openai import OpenAI

# 创建客户端，连接本地 Ollama 服务
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# 向模型发送请求
response = client.chat.completions.create(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "请解释一下什么是 PagedAttention。"
        }
    ],
    temperature=0,
    max_tokens=200
)

# 打印模型回答
print(response.choices[0].message.content)