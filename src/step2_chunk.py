"""
step2_chunk.py —— 第二步：把长文本切成小块（chunk）

【为什么必须切块？这是 RAG 的第一个核心问题，面试必问】

    原因有三条，任何一条都足以逼你必须切：

    1. 【长度限制】大模型的上下文窗口有限（比如 64K token）。
       一本 200 页的 PDF 可能有 30 万字，塞不进去。
       就算能塞，费用也按 token 算，你会破产。

    2. 【效果变差】研究表明，上下文越长，模型对中间部分的信息越不敏感
       （俗称 "lost in the middle"）。塞一大堆无关内容，反而降低回答质量。

    3. 【无法给出处】如果不切块，模型只能笼统回答；
       切了块，我们才能说"答案来自第 5 块"，实现引用溯源。

    所以：知识库问答的本质 = 「先找到最相关的几块」+ 「只把这几个块喂给模型」。

【切块的关键权衡】

    块太大 ──> 一块里混着好几个主题，检索时容易"抓错重点"，
              也会浪费上下文额度
    块太小 ──> 一句话被切断，语义不完整，
              检索到它也没用（比如只查到"因此可知"四个字）

    常见做法：按段落/句子边界切，块之间留一点重叠。

【为什么要有"重叠"(overlap)】

    假设一个关键结论横跨句子的边界：
        ...块的结尾是「快速排序的平均时间复杂度是」
        下一块的开头是「O(n log n)，最坏情况是 O(n²)」...

    如果不重叠，你查"快排时间复杂度"，可能两块都匹配不好。
    加 80 字符的重叠，就保证了跨边界的句子在某一整块里是完整的。

【跑法】
    python src/step2_chunk.py
"""

import re
from pathlib import Path


def split_into_sentences(text: str) -> list[str]:
    """
    把文本切成句子。

    为什么先切句子、再组块？
        因为"按固定字数硬切"会把句子拦腰砍断。
        先切句子，再按"攒够一定字数就断"来组块，就不会切断句子。

    中文断句：句号、问号、感叹号、分号、换行
    """
    # 用正则"在标点后面切开，但保留标点"
    # (?<=[。！？；\n]) 是"后向断言"：要求当前位置前面是这些标点
    # 这样切出来的句子依然带着句末标点，读起来完整
    parts = re.split(r"(?<=[。！？；\n])", text)

    # 去掉空白句，并且丢掉只有一个字的碎片（多半是排版噪音）
    sentences = []
    for part in parts:
        s = part.strip()
        if len(s) >= 2:
            sentences.append(s)

    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 80,
) -> list[dict]:
    """
    把长文本切成若干块。

    参数：
        text:       完整文本
        chunk_size: 每块目标字符数
        overlap:    相邻块的重叠字符数

    返回：
        块的列表。每个块是一个字典，包含：
            - "index":  第几块（从 0 开始）
            - "text":   这一块的文字
            - "start":  在原文本中的起始位置（用于调试和定位）

    【C++ 类比】
        这有点像把一个大 vector<string> 分页读取：
        不是一次读完，而是按固定窗口滑动，每次取一块。
    """

    sentences = split_into_sentences(text)
    print(f"[切块] 先切成 {len(sentences)} 个句子")

    chunks: list[dict] = []

    current: list[str] = []       # 当前正在攒的句子
    current_len = 0               # 当前块已攒了多少字符

    for sentence in sentences:

        # ---- 情况 1：这一句本身就超长 ----
        # 比如整段没有标点，或者是一大段代码/表格。
        # 这种情况不能无限攒，必须硬切，否则这一块会撑爆。
        if len(sentence) > chunk_size:
            # 先把已经攒的存下来
            if current:
                chunks.append(_make_chunk(current, chunks))
                current, current_len = [], 0
            # 再把超长句硬切成几段
            for i in range(0, len(sentence), chunk_size):
                piece = sentence[i : i + chunk_size]
                chunks.append(
                    {
                        "index": len(chunks),
                        "text": piece,
                        "start": 0,  # 硬切的块不好算原位置，先填 0
                    }
                )
            continue

        # ---- 情况 2：加上这一句会超出块大小 ----
        # 这时候要"封口"：把当前攒的句子做成一块，
        # 并且留下最后 overlap 个字符，放进下一块的开头。
        if current_len + len(sentence) > chunk_size:
            if current:
                chunks.append(_make_chunk(current, chunks))

                # ---- 制造重叠 ----
                # 从尾部倒着取，直到凑够 overlap 个字符
                tail: list[str] = []
                tail_len = 0
                for s in reversed(current):
                    if tail_len + len(s) > overlap:
                        break
                    tail.insert(0, s)  # 插到最前面，保持原顺序
                    tail_len += len(s)

                current = tail
                current_len = tail_len

        # ---- 把这一句加入当前块 ----
        current.append(sentence)
        current_len += len(sentence)

    # ---- 循环结束，别忘了最后没封口的那一块 ----
    # 这是新手最容易漏的地方：循环里只处理"满了就封"，
    # 最后不足一块的内容会丢掉。必须补上。
    if current:
        chunks.append(_make_chunk(current, chunks))

    return chunks


def _make_chunk(sentences: list[str], existing: list[dict]) -> dict:
    """辅助函数：把一组句子拼成一个块。下划线开头表示"内部使用"的函数。"""
    return {
        "index": len(existing),
        "text": "".join(sentences),  # 直接相连，因为句子自带标点
        "start": 0,
    }


def main() -> None:
    # ---- 读取上一步提取的文本 ----
    src = Path("data/extracted.txt")
    if not src.exists():
        print("找不到 data/extracted.txt")
        print("请先运行：python src/step1_extract.py data/pdf/你的文件.pdf")
        return

    text = src.read_text(encoding="utf-8")

    print("=" * 60)
    print("步骤 2：切块")
    print("=" * 60)

    # 从环境变量读配置，读不到就用默认值
    # 你可以在 .env 里调整这些参数，体会它们对效果的影响
    import os

    chunk_size = int(os.getenv("CHUNK_SIZE", "500"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "80"))

    print(f"参数：每块 {chunk_size} 字符，重叠 {overlap} 字符")
    print()

    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    print()
    print("-" * 60)
    print(f"切块完成：共 {len(chunks)} 块")
    print("-" * 60)

    # ---- 打印前 3 块让你直观感受 ----
    for c in chunks[:3]:
        print(f"\n【第 {c['index']} 块】({len(c['text'])} 字符)")
        print("-" * 40)
        # 最多显示 200 字，太长不好看
        preview = c["text"][:200]
        print(preview + ("..." if len(c["text"]) > 200 else ""))

    # ---- 统计一下块大小分布，判断参数是否合理 ----
    lengths = [len(c["text"]) for c in chunks]
    print()
    print("=" * 60)
    print("块大小统计（判断你的切块参数是否合理）")
    print("=" * 60)
    print(f"最短：{min(lengths)} 字符")
    print(f"最长：{max(lengths)} 字符")
    print(f"平均：{sum(lengths) // len(lengths)} 字符")
    print()
    print("怎么看这个结果：")
    print("  - 如果最长块远超设定值，说明有超长句（可能缺标点）")
    print("  - 如果平均只有几十字符，说明块太碎了，检索会不准")

    # ---- 保存，供下一步检索使用 ----
    import json

    out = Path("data/chunks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    # ensure_ascii=False 保证中文正常显示而不是 \u4e2d\u6587
    out.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print()
    print(f"[保存] 已写入 {out}")


if __name__ == "__main__":
    main()
