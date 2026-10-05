"""
step4_answer.py —— 第四步：调用大模型，生成「带出处」的答案

【这一步做什么】
    把上一步检索到的几块内容，和用户的问题一起，打包成一段 Prompt
    发给 DeepSeek，要求它：只根据给定材料回答，并且标注引用编号。

【为什么必须"只根据材料回答"？—— 这是 RAG 的核心价值】

    如果只是把问题直接扔给大模型，它会用"训练时记住的知识"回答，
    可能出现两种情况：
      1. 它不知道 → 一本正经地编（幻觉 hallucination）
      2. 它知道，但记错了/过时了 → 答案和你的文档不符

    RAG 的思路是：把"事实"交给文档，"表达"交给模型。
    模型不再需要"记住"知识，只需要"读懂并组织"我们递给它的材料。
    这能大幅降低幻觉，也是企业愿意为 RAG 付费的原因。

【怎么让模型乖乖只用材料？—— Prompt 设计的三个关键】

    1. 明确角色和约束："你是一个严格的知识库助手..."
    2. 给出免责出口："如果材料里没有，就说'文档中没有提到'"
       ← 这条极其重要！不给台阶，模型就会硬编。
    3. 给出输出格式：要求标 [1][2]，方便我们在后端校验和展示

【跑法】
    python src/step4_answer.py "你的问题"
    或直接 python src/step4_answer.py 进入交互模式
"""

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

# 把上一步的检索器引进来
# sys.path 是 Python 找模块时搜索的目录列表。
# 因为 step4 和 step3 在同一个目录，正常情况下可以直接 import。
# 这里为了稳妥（从别的目录运行时也能用），手动把 src 加进去。
sys.path.insert(0, str(Path(__file__).parent))

from step3_retrieve import load_retriever, show_results  # noqa: E402

# load_dotenv() 会读取 .env 文件，把里面的变量塞进环境变量
# 这样 os.getenv("API_KEY") 就能拿到，而不用把密钥写死在代码里
load_dotenv()


# ============================================================
#  一、构造 Prompt
# ============================================================

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。

你的工作规则（必须严格遵守）：
1. 你只能依据「参考资料」中的内容回答问题，绝对不能使用你自己的知识。
2. 每次给出一个结论，必须在句末标注来源编号，格式为 [1]、[2]。
3. 如果参考资料中没有足够信息回答问题，你必须明确回答：
   「文档中没有提到相关信息」，不要猜测、不要编造。
4. 如果参考资料只能部分回答问题，就只回答能回答的部分，
   并说明哪部分资料中未涉及。
5. 用简洁、有条理的方式回答。如果问题有多个方面，分点作答。"""


def build_context(chunks: list[dict]) -> str:
    """
    把检索到的块拼成「参考资料」文本。

    格式设计要点：给每块编号 [1] [2]，这样模型能引用，用户也能对上。
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        # strip() 去掉首尾空白，避免块之间粘在一起
        lines.append(f"[{i}] {chunk['text'].strip()}")

    # 用两个换行分隔不同块，让模型能看清边界
    return "\n\n".join(lines)


def build_user_prompt(question: str, chunks: list[dict]) -> str:
    """组装发给模型的用户消息。"""
    context = build_context(chunks)

    return f"""参考资料：

{context}

---

请依据上面的参考资料回答问题。

问题：{question}"""


# ============================================================
#  二、调用 DeepSeek API
# ============================================================


def call_llm(question: str, chunks: list[dict]) -> str:
    """
    调用 DeepSeek，返回答案文本。

    【为什么用 requests 手写 HTTP 请求，而不是用 openai 库？】

        因为 DeepSeek 完全兼容 OpenAI 的接口协议，
        手写一次你就彻底明白"调用大模型"到底发生了什么 ——
        无非就是发一个 POST，body 里放 messages 数组。

        你之前问过 chat.completions 的结构，这里就是它的真身：
            messages 是一个数组，每个元素有 role 和 content
            role 有三种：
              - "system"    : 设定角色和规则（优先级最高）
              - "user"      : 用户说的话
              - "assistant" : 模型之前的回复（多轮对话时用）
    """

    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com").rstrip("/")
    model = os.getenv("MODEL", "deepseek-chat")

    # ---- 提前检查配置，给出清晰报错 ----
    if not api_key or api_key.startswith("sk-your-key"):
        raise RuntimeError(
            "没有配置 API_KEY。请把 .env.example 复制成 .env，填入你的真实密钥。"
        )

    url = f"{base_url}/chat/completions"

    headers = {
        # Bearer 是标准的令牌认证格式，几乎所有大模型服务都用这个
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ],
        # temperature 控制随机性：0 表示尽量稳定、别发挥。
        # 知识库问答要的是"忠实"，不需要创意，所以调低。
        "temperature": 0.2,
        # 限制输出长度，省钱也防跑偏
        "max_tokens": 1000,
    }

    print(f"[调用] {url}")
    print(f"[模型] {model}")

    # timeout 一定要设！否则网络卡住时程序会永远挂着
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    # ---- 处理错误 ----
    # 大模型 API 的错误码很有信息量，别只写 "请求失败"
    if resp.status_code != 200:
        raise RuntimeError(
            f"接口返回错误 {resp.status_code}：{resp.text[:500]}"
        )

    data = resp.json()

    # 返回结构：{"choices": [{"message": {"content": "..."}}], "usage": {...}}
    answer = data["choices"][0]["message"]["content"]

    # 顺手打印 token 消耗，让你对成本有感觉（DeepSeek 便宜，但要有概念）
    usage = data.get("usage", {})
    if usage:
        print(f"[消耗] 输入 {usage.get('prompt_tokens', '?')} tokens，"
              f"输出 {usage.get('completion_tokens', '?')} tokens")

    return answer


# ============================================================
#  三、把答案和出处一起展示
# ============================================================


def answer_question(retriever, question: str, top_k: int = 3) -> dict:
    """
    完整流程：检索 → 生成 → 打包结果。

    返回一个字典，包含答案和引用来源 —— 这就是后面 Web 接口要返回的数据结构。
    """
    # ---- 1. 检索 ----
    chunks = retriever.search(question, top_k=top_k)

    if not chunks:
        return {
            "answer": "抱歉，我在知识库中没有找到与这个问题相关的内容。",
            "sources": [],
            "chunks": [],
        }

    # ---- 2. 生成 ----
    answer = call_llm(question, chunks)

    # ---- 3. 打包来源 ----
    # 只保留前端需要的字段，不要把整个块都传出去（太长、也没必要）
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        sources.append(
            {
                "id": i,
                "chunk_index": chunk["index"],
                "score": chunk.get("score", 0),
                "preview": chunk["text"][:150] + (
                    "..." if len(chunk["text"]) > 150 else ""
                ),
            }
        )

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,  # 原始块留着调试用
    }


def main() -> None:
    print("=" * 70)
    print("步骤 4：生成带出处的答案")
    print("=" * 70)

    # 先检查 Key 有没有配好，别等用户问完才发现没配
    if not os.getenv("API_KEY") or os.getenv("API_KEY", "").startswith("sk-your-key"):
        print()
        print("⚠️  还没配置 API_KEY")
        print()
        print("请执行：")
        print("    cp .env.example .env")
        print("然后用记事本打开 .env，把 API_KEY 换成你自己的 Key")
        print()
        return

    retriever = load_retriever()
    top_k = int(os.getenv("TOP_K", "3"))

    # ---- 单次提问模式 ----
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        _handle_one(retriever, question, top_k)
        return

    # ---- 交互模式 ----
    print()
    print("交互模式：输入问题后回车，输入 q 退出")
    print()

    while True:
        try:
            question = input("你的问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            break

        _handle_one(retriever, question, top_k)
        print()


def _handle_one(retriever, question: str, top_k: int) -> None:
    """处理一个问题的完整展示。"""
    print()
    print("=" * 70)
    print(f"问题：{question}")
    print("=" * 70)

    result = answer_question(retriever, question, top_k)

    print()
    print("【回答】")
    print("-" * 70)
    print(result["answer"])

    if result["sources"]:
        print()
        print("【出处】")
        print("-" * 70)
        for s in result["sources"]:
            print(f"  [{s['id']}] 块 #{s['chunk_index']}　"
                  f"(相关度 {s['score']})")
            print(f"      {s['preview']}")


if __name__ == "__main__":
    main()
