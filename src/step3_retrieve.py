"""
step3_retrieve.py —— 第三步：检索（找出最相关的几块）

【这是整个 RAG 项目最重要的一个文件，面试 80% 的时间会聊这里】

    前面两步（提取、切块）是体力活。真正的技术含量在这一步：
    给定用户的问题，从上万个块里找出最相关的 3 块。

    这一步做得好不好，直接决定了最终答案的质量。
    因为大模型只能看到我们递给它的那 3 块 —— 递错了，神仙也答不对。
    这就是所谓的 "Garbage In, Garbage Out"。

【本文件的策略：关键词 + 词频加权（类似 BM25 的简化版）】

    第 1 阶段我们故意不用向量检索。原因见下面「方案对比」。

【两个方案的对比（这就是第 4 阶段做对比实验的素材）】

    ┌────────────┬──────────────────────┬──────────────────────┐
    │            │ 关键词检索（本文件）    │ 向量检索（第 4 阶段）   │
    ├────────────┼──────────────────────┼──────────────────────┤
    │ 原理        │ 数「词」的重叠         │ 把文本变成一串数字，    │
    │            │                      │ 比较数字向量的方向      │
    │ 优点        │ 快（毫秒级）、免费、    │ 懂语义：查「内存不够」   │
    │            │ 结果可解释（能说清      │ 也能匹配「虚拟内存」     │
    │            │ 为什么命中）           │                      │
    │ 缺点        │ 不懂语义：查「内存不够」 │ 慢（要调 API）、要花钱、 │
    │            │ 匹配不到「虚拟内存」    │ 结果是黑盒，说不清为什么 │
    │ 适用        │ 术语查询、精确匹配      │ 口语化提问、同义表达     │
    └────────────┴──────────────────────┴──────────────────────┘

    真实工业界两者都用，融合排序 —— 那是你第 4 阶段可以做的进阶。

【核心算法：怎么给一个块打分？】

    朴素想法：数问题和块里有多少个相同的词。
    但有个致命问题：像「的」「是」「在」这种字，几乎每块都有，
    它们不携带任何区分信息，却会污染打分。

    解决：给每个词加权重。越「稀有」的词权重越高。
    这就是 TF-IDF 的思想：
        TF  = 这个词在这个块里出现几次（越多越相关）
        IDF = log(总块数 / 含这个词的块数)  ← 越稀有分越高
        权重 = TF × IDF

    举例：
        「快速排序」 只出现在 2 个块里  → IDF 很高 → 权重高 ✓
        「算法」     出现在 50 个块里   → IDF 很低 → 权重低 ✓
        「的」       出现在 100 个块里  → IDF ≈ 0  → 权重≈0 ✓

【跑法】
    交互模式：python src/step3_retrieve.py
    单次查询：python src/step3_retrieve.py "什么是快速排序"
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


# ============================================================
#  中文分词：一个简化但有说法的方案
# ============================================================

# 中文和英文不一样，英文单词天然用空格分开，中文没有分隔符。
# "研究生物" 到底是 "研究/生物" 还是 "研究生/物"？这就是中文分词的难点。
#
# 生产环境会用 jieba 等分词库。但第 1 阶段我们先用一个更简单的方法：
# 「二元组」(bigram) —— 把连续的 2 个汉字当成一个"词"。
#
#   例如 "快速排序算法"
#        → 快速, 速排, 排序, 序算, 算法
#
# 为什么这样能work？
#   如果问题和文档里有相同的词组（比如都含"排序"），
#   那么它们的 bigram 集合一定有交集。不需要词典也能匹配。
#
# 优点：零依赖、对术语匹配效果好
# 缺点：无法理解同义词（"快排" vs "快速排序" 匹配不上）
# 这正好是第 4 阶段升级向量检索的动机 —— 你有了对比的素材。


def tokenize(text: str) -> list[str]:
    """
    把一段文本切成"词"列表。

    策略：
      - 英文/数字：按单词切（连续字母数字算一个词）
      - 中文：切成连续的 2 字组合（bigram）

    【C++ 类比】
        返回 vector<string>。这里用 Python 的列表，就是动态数组，语义一样。
    """
    tokens: list[str] = []

    # ---- 1. 提取英文单词和数字 ----
    # [a-zA-Z0-9]+ 匹配一个或多个字母数字
    # 转成小写，让 "FastAPI" 和 "fastapi" 算同一个词
    for word in re.findall(r"[a-zA-Z0-9]+", text.lower()):
        # 过滤掉超短的无意义词（a, is, 1 这种）
        if len(word) >= 2:
            tokens.append(word)

    # ---- 2. 提取中文，切成 bigram ----
    # [\u4e00-\u9fff] 是常用汉字的 Unicode 范围
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) == 1:
            # 单独一个汉字也没法切 bigram，直接当词用
            tokens.append(run)
        else:
            # 滑动窗口取 2 个连续汉字
            for i in range(len(run) - 1):
                tokens.append(run[i : i + 2])

    return tokens


# ============================================================
#  打分器：TF-IDF 的简化实现
# ============================================================


class TfIdfRetriever:
    """
    基于 TF-IDF 的检索器。

    【C++ 类比】
        这个类相当于一个"索引对象"。你写搜索引擎时会建一个
        unordered_map<string, vector<int>> 倒排索引，这里是同样的东西，
        只是用 Python 的 dict（哈希表）实现。
    """

    def __init__(self) -> None:
        self.chunks: list[dict] = []           # 所有块
        self.doc_freq: Counter = Counter()     # 每个词出现在多少个块里（算 IDF 用）
        self.chunk_vectors: list[Counter] = [] # 每个块的词频（算 TF 用）
        self.total_chunks: int = 0

    # --------------------------------------------------------
    #  建索引
    # --------------------------------------------------------
    def build_index(self, chunks: list[dict]) -> None:
        """
        遍历所有块，建立索引。

        【C++ 类比】
            相当于在构造时预处理一遍数据，把 O(n) 次计算提前做完，
            这样后面每次查询都很快（典型的"空间换时间"）。
        """
        self.chunks = chunks
        self.total_chunks = len(chunks)
        self.chunk_vectors = []
        self.doc_freq = Counter()

        for chunk in chunks:
            tokens = tokenize(chunk["text"])

            # Counter 会自动统计每个词出现次数
            # 例：["排序","排序","算法"] → {"排序": 2, "算法": 1}
            freq = Counter(tokens)
            self.chunk_vectors.append(freq)

            # 注意：doc_freq 统计的是"出现在几个块里"，不是"总共出现几次"
            # 所以用 set 去重
            for word in set(tokens):
                self.doc_freq[word] += 1

        print(f"[索引] 建立完成：{self.total_chunks} 个块，"
              f"共 {len(self.doc_freq)} 个不同的词")

    # --------------------------------------------------------
    #  计算一个词的 IDF 权重
    # --------------------------------------------------------
    def _idf(self, word: str) -> float:
        """
        IDF = log(总块数 / 含该词的块数)

        加 1 是平滑处理：防止除零，也避免某个词在所有块都出现时
        log(1) = 0 导致该词完全失效。
        """
        df = self.doc_freq.get(word, 0)
        if df == 0:
            return 0.0  # 这个词在知识库里不存在，权重为 0
        return math.log((self.total_chunks + 1) / (df + 1)) + 1.0

    # --------------------------------------------------------
    #  查询：返回最相关的 K 个块
    # --------------------------------------------------------
    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        给定问题，返回打分最高的 top_k 个块。

        返回的每个元素在原来的块基础上，多了两个字段：
            - "score": 相关性得分（越高越相关）
            - "matched_words": 命中了哪些词（用于调试，能让你看懂"为什么是它"）
        """

        # ---- 1. 把问题也切词 ----
        query_tokens = tokenize(query)

        if not query_tokens:
            print("[警告] 问题里没有可识别的词")
            return []

        # ---- 2. 问题的词也去重 ----
        # 为什么要去重？因为通常我们关心"问题里的这个意思覆盖到了没有"，
        # 而不是它在问题里出现了几次。
        unique_query_tokens = set(query_tokens)

        # ---- 3. 逐个块打分 ----
        results = []

        for i, chunk_freq in enumerate(self.chunk_vectors):
            score = 0.0
            matched = []

            for word in unique_query_tokens:
                if word in chunk_freq:
                    # TF：这个词在块里出现了几次
                    tf = chunk_freq[word]
                    # IDF：这个词有多稀有（越稀有越值钱）
                    idf = self._idf(word)

                    # 核心公式：TF × IDF
                    # 用 1 + log(tf) 而不是直接用 tf，是为了抑制"刷词"——
                    # 一个词重复 100 次不代表比出现 10 次相关 10 倍
                    score += (1 + math.log(tf)) * idf
                    matched.append(word)

            if score > 0:
                item = dict(self.chunks[i])  # 复制一份，不污染原数据
                item["score"] = round(score, 3)
                item["matched_words"] = matched
                results.append(item)

        # ---- 4. 按分数从高到低排序 ----
        # key 指定"按哪个字段排"，reverse=True 表示降序
        # 相当于 C++ 的 std::sort(v.begin(), v.end(),
        #     [](auto&a, auto&b){ return a.score > b.score; });
        results.sort(key=lambda x: x["score"], reverse=True)

        if not results:
            print("[警告] 没有任何块和你的问题匹配上")
            print("       可能原因：问题用词和文档差异太大（这正是关键词检索的短板）")
            return []

        return results[:top_k]


# ============================================================
#  主流程
# ============================================================


def load_retriever() -> TfIdfRetriever:
    """读取上一步切好的块，建好索引。"""
    path = Path("data/chunks.json")
    if not path.exists():
        print("找不到 data/chunks.json")
        print("请先运行：python src/step1_extract.py <pdf> && python src/step2_chunk.py")
        sys.exit(1)

    chunks = json.loads(path.read_text(encoding="utf-8"))
    retriever = TfIdfRetriever()
    retriever.build_index(chunks)
    return retriever


def show_results(query: str, results: list[dict]) -> None:
    """把检索结果打印成人能看懂的样子。"""
    print()
    print("=" * 70)
    print(f"问题：{query}")
    print("=" * 70)

    if not results:
        print("（没有找到相关内容）")
        return

    for rank, r in enumerate(results, start=1):
        print()
        print(f"【第 {rank} 名】块 #{r['index']}　得分 {r['score']}")
        print(f"　　命中词：{'、'.join(r['matched_words'][:12])}")
        print("-" * 70)
        text = r["text"]
        print(text[:280] + ("..." if len(text) > 280 else ""))


def main() -> None:
    print("=" * 70)
    print("步骤 3：检索")
    print("=" * 70)

    retriever = load_retriever()

    import os

    top_k = int(os.getenv("TOP_K", "3"))

    # ---- 模式 1：命令行直接给了问题 ----
    # sys.argv[1:] 表示"除脚本名外的所有参数"
    # 用 " ".join 拼起来，这样问题里有空格也能正确传入
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = retriever.search(query, top_k=top_k)
        show_results(query, results)
        return

    # ---- 模式 2：交互模式，可以一直问 ----
    print()
    print("交互模式：输入问题后回车，输入 q 退出")
    print()

    while True:
        try:
            query = input("你的问题 > ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+C / Ctrl+D 时优雅退出，不打印一堆报错
            print()
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            break

        results = retriever.search(query, top_k=top_k)
        show_results(query, results)
        print()


if __name__ == "__main__":
    main()
