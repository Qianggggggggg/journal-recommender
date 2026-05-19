"""解析 CCF 推荐期刊目录 PDF，生成期刊数据"""
import json
import re
from PyPDF2 import PdfReader
from pathlib import Path


def parse_ccf_journals():
    """从 CCF PDF 中解析期刊数据"""
    reader = PdfReader('/Users/qian/Downloads/第七版中国计算机学会推荐国际学术会议和期刊目录（正式版）.pdf')

    journals_by_category = {}
    current_category = None

    # 领域映射
    category_map = {
        "计算机体系结构/并行与分布计算/存储系统": ["体系结构", "并行计算", "分布式系统", "存储系统"],
        "计算机网络": ["网络", "通信", "无线", "移动计算"],
        "网络与信息安全": ["安全", "隐私", "密码学", "网络安全"],
        "软件工程/系统软件/程序设计语言": ["软件工程", "程序设计语言", "系统软件"],
        "数据库/数据挖掘/内容检索": ["数据库", "数据挖掘", "信息检索"],
        "计算机科学理论": ["理论计算", "算法", "逻辑"],
        "计算机图形学与多媒体": ["图形学", "多媒体", "视觉", "图像处理"],
        "人工智能": ["人工智能", "机器学习", "计算机视觉", "NLP"],
        "人机交互与普适计算": ["人机交互", "普适计算", "HCI"],
        "交叉/综合/新兴": ["交叉学科", "综合"],
    }

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or len(text.strip()) < 50:
            continue

        # 检测类别
        for cat in category_map:
            if cat in text and ("期刊" in text or "会议" in text):
                current_category = cat
                break

        # 解析期刊表格
        lines = text.split('\n')
        for line in lines:
            # 匹配期刊行：A类/B类/C类
            if match := re.match(r'[一二三]+[、.．]([ABC])[类.]?\s*(\d+)', line):
                rank = match.group(1)
                continue
            # 匹配具体期刊数据
            if match := re.match(r'\d+[、.．]?\s*([^\s]+?)\s+([^\s]+?)\s+(ACM|IEEE|Springer|Elsevier|Wiley|MIT Press|IOSPress|SIAM|CCF|科学出版社|清华大学出版社|中国科技大学出版社)\s+http', line):
                abbrev = match.group(1).strip()
                full_name = match.group(2).strip()
                publisher = match.group(3).strip()
                # 提取 URL
                url_match = re.search(r'http[^\s]+', line)
                url = url_match.group(0) if url_match else None

                if abbrev and full_name:
                    journal_id = abbrev.lower().replace('.', '_')
                    journals_by_category.setdefault(current_category, []).append({
                        "journal_id": journal_id,
                        "journal_name": f"{abbrev} - {full_name}",
                        "abbrev": abbrev,
                        "full_name": full_name,
                        "publisher": publisher,
                        "subject_tags": [current_category] if current_category else [],
                        "ccf_rank": rank if 'rank' in locals() else "C",
                        "homepage_url": url,
                    })

    return journals_by_category


def parse_journal_section(text):
    """解析期刊段落"""
    journals = []

    # 分割成独立期刊条目
    # 格式：序号 期刊简称 期刊全称 出版社 网址
    pattern = r'(\d+)[、.]?\s*([A-Za-z\d\-/]+)\s+([^\s]+(?:[^\s]{5,})?)\s+(ACM|IEEE|Springer|Elsevier|Wiley|MIT Press|IOSPress|SIAM|CCF|科学出版社|清华大学出版社|中国科技大学出版社)\s*(http[^\s]+)?'

    matches = re.findall(pattern, text)
    for m in matches:
        seq, abbrev, full_name, publisher, url = m
        if len(abbrev) > 2 and len(full_name) > 3:
            journals.append({
                "journal_id": abbrev.lower().replace('.', '_').replace('-', '_'),
                "journal_name": f"{abbrev} - {full_name}",
                "publisher": publisher,
                "url": url if url else None,
            })

    return journals


if __name__ == "__main__":
    result = parse_ccf_journals()
    print(json.dumps(result, ensure_ascii=False, indent=2))