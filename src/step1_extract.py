"""
step1_extract.py —— 第一步：把 PDF 里的文字抠出来

【这一步在整个链路里的位置】
    PDF 文件  ──[本文件]──>  纯文本字符串  ──>  切块  ──>  检索  ──>  大模型

【为什么需要这一步】
    大模型只吃"文本"，不吃 PDF。PDF 本质上是一种"排版文件"——
    它记录的是"在第 100 行、第 200 列画一个字母 A"，而不是"这句话是 A"。
    所以必须有一个库把它翻译回人类可读的文字。PyMuPDF 就是干这个的。

【跑法】
    python src/step1_extract.py data/pdf/你的文件.pdf

【C++ 类比】
    PDF 有点像序列化后的二进制对象 —— 信息都在，但不是你直接能读的形式。
    这一步相当于"反序列化"：把二进制布局翻译回结构化文本。
"""

import sys
from pathlib import Path

# PyMuPDF 的导入名历史上叫 fitz（因为它的底层渲染引擎是 MuPDF）。
# 新版库同时提供了 pymupdf 这个名字，用来避免弃用警告。
# 这里做兼容处理：优先用新名字，老版本回退到 fitz。
try:
    import pymupdf as fitz
except ImportError:
    import fitz


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 文件里提取全部文字。

    参数：
        pdf_path: PDF 文件的路径

    返回：
        整个 PDF 的文字内容，页面之间用换行分隔

    抛错：
        文件不存在 / 不是有效 PDF 时抛异常，让调用者知道出了问题
    """
    path = Path(pdf_path)

    # ---- 检查文件是否存在 ----
    # 这是个好习惯：早点报错，比后面莫名其妙的错误好排查
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{pdf_path}")

    # ---- 打开 PDF ----
    # fitz.open() 会读入整个文件并建立索引，类似 ifstream 打开文件
    # 用 with 语句保证用完自动关闭（相当于 C++ 的 RAII / 智能指针）
    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        print(f"[提取] 打开成功，共 {total_pages} 页")

        page_texts = []

        # ---- 逐页提取 ----
        # enumerate 你之前问过：它同时给你"下标"和"元素"，
        # 相当于 C++ 的 for (int i = 0; i < n; i++) 但不用手写 i++
        for page_index, page in enumerate(doc):
            # get_text() 是核心方法，把这一页的文字抽成一个字符串
            text = page.get_text()

            if text.strip():  # strip() 去掉首尾空白；空页面就跳过
                page_texts.append(text)
                print(f"  - 第 {page_index + 1} 页：{len(text)} 个字符")
            else:
                # 有些页是纯图片（扫描件），PyMuPDF 抽不出文字
                # 这是常见情况，不是 bug。后面第 4 阶段可以加 OCR 解决
                print(f"  - 第 {page_index + 1} 页：无文字（可能是图片页，已跳过）")

    # ---- 拼接所有页 ----
    # "\n" 是换行符。用 join 而不是循环 += 拼接，
    # 因为 Python 的字符串是不可变的（类似 C++ 的 const string），
    # 每次 += 都会新建一个字符串，循环拼会越来越慢（O(n²)）。
    # join 一次性算好总长度，是 O(n)。
    full_text = "\n".join(page_texts)

    return full_text


def clean_text(text: str) -> str:
    """
    做一点点最基本的文本清洗。

    【重要提醒】清洗的原则是「只去掉明显是排版垃圾的东西」，
    不要过度清洗 —— 你删掉的东西，可能就是用户想查的内容。

    这里只做两件事：
      1. 把连续 3 个以上的换行，压成最多 2 个（去掉 PDF 里的空白行）
      2. 去掉行尾多余的空格
    """
    import re  # re = regular expression 正则表达式，用来做文本匹配

    # r"\n{3,}" 意思是"匹配 3 个及以上的换行符"
    # r 前缀表示"这是原始字符串"，里面的 \n 不被 Python 提前转义
    text = re.sub(r"\n{3,}", "\n\n", text)

    # r"[ \t]+\n" 意思是"行尾的空格或制表符，后面跟一个换行"
    # 把它替换成"\n"，就去掉了行尾空白
    text = re.sub(r"[ \t]+\n", "\n", text)

    return text.strip()


def main() -> None:
    """命令行入口：python src/step1_extract.py <pdf路径>"""

    if len(sys.argv) < 2:
        # sys.argv[0] 是脚本名本身，sys.argv[1] 才是第一个真实参数
        print("用法：python src/step1_extract.py data/pdf/你的文件.pdf")
        sys.exit(1)  # 非 0 退出码表示"出错了"

    pdf_path = sys.argv[1]

    print("=" * 60)
    print("步骤 1：从 PDF 提取文字")
    print("=" * 60)

    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)

    print()
    print("-" * 60)
    print(f"提取完成：共 {len(cleaned)} 个字符")
    print("-" * 60)

    # 打印前 300 个字符让你确认"确实是这份 PDF 的内容"
    # 这一步很关键 —— 如果提取出来是乱码，后面全白搭
    print("内容预览（前 300 字）：")
    print(cleaned[:300])
    print("...")

    # 顺手存一份纯文本，方便你直接用记事本打开检查
    out_path = Path("data/extracted.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")
    print()
    print(f"[保存] 完整文字已写入 {out_path}（用记事本打开看看，确认没乱码）")


if __name__ == "__main__":
    # 这行的意思是：只有"直接运行本文件"时才执行 main()
    # 如果被别的文件 import，则不执行。
    # 相当于 C++ 里 int main() 的效果，但更灵活。
    main()
