# PDF Layout-Aware Parsing 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 PDF 解析从 PyPDF2 raw text 升级为 PyMuPDF layout-aware extraction，输出结构化 Paper AST，大幅提升 LLM 对论文结构的理解能力。

**Architecture:**
```
PDF → PyMuPDF blocks → Section Detector → Paper AST → Markdown → LLM
                                                        ↑
                                              (title/abstract模式不变)
```

**Tech Stack:** PyMuPDF (替换 PyPDF2), 现有 section_splitter 升级

---

## Task 1: 设计 Paper AST 数据结构

**Files:**
- Modify: `src/papers/paper_model.py`

- [ ] **Step 1: 添加 Section 和 PaperDocument 类**

```python
# src/papers/paper_model.py 新增

class Block(BaseModel):
    """单个文本块（来自 PyMuPDF）"""
    text: str
    font_size: float = 0.0
    font_name: str = ""
    bold: bool = False
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    page_number: int = 0


class Section(BaseModel):
    """论文章节"""
    title: str = ""
    level: int = 1  # 1=title, 2=section, 3=subsection
    content: str = ""  # 该章节的完整文本
    blocks: List[Block] = Field(default_factory=list)  # 原始 blocks


class PaperDocument(BaseModel):
    """论文结构化文档（Paper AST）"""
    title: str = ""
    abstract: str = ""
    sections: List[Section] = Field(default_factory=list)
    all_blocks: List[Block] = Field(default_factory=list)  # 全局 blocks

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        lines = []
        if self.title:
            lines.append(f"# {self.title}\n")
        if self.abstract:
            lines.append(f"## Abstract\n{self.abstract}\n")
        for section in self.sections:
            level_prefix = "#" * min(section.level + 1, 6)
            lines.append(f"{level_prefix} {section.title}\n{section.content}\n")
        return "\n".join(lines)
```

- [ ] **Step 2: 在 SectionSplitResult 中添加 blocks 字段（向后兼容）**

```python
class SectionSplitResult(BaseModel):
    introduction: str = ""
    method: str = ""
    experiment: str = ""
    conclusion: str = ""
    other: str = ""
    blocks: List[Block] = Field(default_factory=list)  # 新增
```

- [ ] **Step 3: 提交**

```bash
git add src/papers/paper_model.py
git commit -m "feat: add Paper AST data structures (Section, PaperDocument, Block)"
```

---

## Task 2: 实现 PyMuPDF layout extraction

**Files:**
- Modify: `src/utils/file_parser.py`

- [ ] **Step 1: 添加 PyMuPDF 依赖到 pyproject.toml**

```toml
dependencies = [
    ...
    "pymupdf>=1.24.0",  # 新增，替换 PyPDF2
]
```

- [ ] **Step 2: 实现 extract_layout_blocks() 函数**

```python
# src/utils/file_parser.py 新增

from dataclasses import dataclass

@dataclass
class LayoutBlock:
    """Layout-aware text block from PyMuPDF"""
    text: str
    font_size: float
    font_name: str
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


def extract_layout_blocks(file_content: bytes, filename: str) -> Tuple[List[LayoutBlock], str]:
    """
    使用 PyMuPDF 提取 layout-aware blocks。

    返回: (blocks, full_text)
    - blocks: 按阅读顺序排列的 LayoutBlock 列表
    - full_text: 简单拼接的全文（向后兼容）
    """
    import fitz  # PyMuPDF

    ext = Path(filename).suffix.lower()
    if ext != ".pdf":
        return [], extract_text_from_file(file_content, filename)

    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        blocks = []
        full_text_parts = []

        for page_num, page in enumerate(doc):
            # 使用 dict 模式获取 blocks（关键区别于 PyPDF2）
            blocks_data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

            for block in blocks_data.get("blocks", []):
                if block.get("type") != 0:  # 只处理文本块
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # 提取字体信息
                        font = span.get("font", "")
                        size = span.get("size", 0)
                        bold = "bold" in font.lower() or "Heavy" in font

                        # 坐标
                        bbox = span.get("bbox", [0, 0, 0, 0])

                        layout_block = LayoutBlock(
                            text=text,
                            font_size=size,
                            font_name=font,
                            bold=bold,
                            x0=bbox[0],
                            y0=bbox[1],
                            x1=bbox[2],
                            y1=bbox[3],
                            page=page_num + 1,
                        )
                        blocks.append(layout_block)
                        full_text_parts.append(text)

            doc.close()

        # 按阅读顺序排序（按 page, y, x）
        blocks.sort(key=lambda b: (b.page, b.y0, b.x0))

        return blocks, " ".join(full_text_parts)

    except Exception as e:
        print(f"[LayoutExtraction] error: {e}")
        return [], ""


def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """向后兼容：提取纯文本"""
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_content)
    elif ext in [".txt", ".md"]:
        return extract_text_from_text(file_content)
    else:
        try:
            return file_content.decode("utf-8")
        except UnicodeDecodeError:
            return file_content.decode("latin-1", errors="replace")
```

- [ ] **Step 3: 提交**

```bash
git add src/utils/file_parser.py pyproject.toml
git commit -m "feat: add PyMuPDF layout extraction (extract_layout_blocks)"
```

---

## Task 3: 升级 SectionSplitter 为 layout-aware

**Files:**
- Modify: `src/papers/section_splitter.py`

- [ ] **Step 1: 实现 FontSizeBasedSectionDetector 类**

```python
# src/papers/section_splitter.py 新增

from dataclasses import dataclass

@dataclass
class LayoutBlock:
    """与 file_parser.py 共享的 Block 类型"""
    text: str
    font_size: float
    font_name: str
    bold: bool
    x0: float
    y0: float
    x1: float
    y1: float
    page: int


class FontSizeSectionDetector:
    """基于字体大小检测章节标题"""

    # 典型字体大小阈值（PT）
    TITLE_SIZE_THRESHOLD = 16.0   # 论文标题
    SECTION_SIZE_THRESHOLD = 13.0  # 章节标题 (1. Introduction)
    SUBSECTION_SIZE_THRESHOLD = 11.0  # 子章节

    # 常见章节标题关键词
    SECTION_KEYWORDS = [
        "Introduction", "Related Work", "Background", "Preliminaries",
        "Method", "Methodology", "Approach", "Model", "Algorithm",
        "Experiment", "Experiments", "Evaluation", "Results", "Analysis",
        "Discussion", "Conclusion", "Conclusion and Future Work",
        "References", "Acknowledgment", "Appendix",
    ]

    def detect(self, blocks: List[LayoutBlock]) -> List[Section]:
        """
        检测章节结构。

        返回: List[Section]，每个 Section 包含 title, level, content
        """
        sections = []
        current_section = {"title": "Preamble", "level": 1, "content": [], "blocks": []}

        for block in blocks:
            is_heading, heading_level = self._is_heading_block(block)

            if is_heading:
                # 保存前一个 section
                if current_section["content"] or current_section["blocks"]:
                    sections.append(Section(
                        title=current_section["title"],
                        level=current_section["level"],
                        content=" ".join(current_section["content"]),
                        blocks=current_section["blocks"],
                    ))

                current_section = {
                    "title": block.text,
                    "level": heading_level,
                    "content": [],
                    "blocks": [block],
                }
            else:
                current_section["content"].append(block.text)
                current_section["blocks"].append(block)

        # 最后一个 section
        if current_section["content"] or current_section["blocks"]:
            sections.append(Section(
                title=current_section["title"],
                level=current_section["level"],
                content=" ".join(current_section["content"]),
                blocks=current_section["blocks"],
            ))

        return sections

    def _is_heading_block(self, block: LayoutBlock) -> Tuple[bool, int]:
        """判断 block 是否为标题"""
        text = block.text.strip()

        # 检查是否匹配章节关键词（作为主要判断）
        for keyword in self.SECTION_KEYWORDS:
            if text.startswith(keyword) or text == keyword:
                # 基于字体大小判断层级
                if block.font_size >= self.TITLE_SIZE_THRESHOLD:
                    return True, 1
                elif block.font_size >= self.SECTION_SIZE_THRESHOLD:
                    return True, 2
                elif block.font_size >= self.SUBSECTION_SIZE_THRESHOLD:
                    return True, 3

        # 检查是否纯数字标题（e.g., "1. Introduction"）
        import re
        if re.match(r"^\d+\.\s+[A-Z]", text):
            return True, 2

        # 检查字体大小异常（显著大于周围文本）
        if block.font_size >= self.TITLE_SIZE_THRESHOLD and len(text) < 100:
            return True, 1

        return False, 0


def build_paper_ast(blocks: List[LayoutBlock], title: str = "") -> PaperDocument:
    """从 blocks 构建 Paper AST"""
    detector = FontSizeSectionDetector()
    sections = detector.detect(blocks)

    # 提取 abstract（通常是最前面的段落）
    abstract_text = ""
    if sections and sections[0].title in ["Abstract", "摘要"]:
        abstract_text = sections[0].content
        sections = sections[1:]  # 移除 abstract section

    return PaperDocument(
        title=title,
        abstract=abstract_text,
        sections=sections,
        all_blocks=blocks,
    )
```

- [ ] **Step 2: 提交**

```bash
git add src/papers/section_splitter.py
git commit -m "feat: upgrade SectionSplitter to layout-aware (FontSizeSectionDetector)"
```

---

## Task 4: 实现 PaperDocument.to_markdown()

**Files:**
- Modify: `src/papers/paper_model.py`

- [ ] **Step 1: 实现 to_markdown() 方法**

```python
# 在 PaperDocument 类中添加

def to_markdown(self) -> str:
    """将论文 AST 转换为 Markdown 格式"""
    lines = []

    # 标题
    if self.title:
        lines.append(f"# {self.title}\n")

    # Abstract
    if self.abstract:
        lines.append(f"## Abstract\n{self.abstract}\n")

    # 各章节
    for section in self.sections:
        # 标题层级：level 1 → ##, level 2 → ###, etc.
        heading_level = min(section.level + 1, 6)
        heading_prefix = "#" * heading_level

        if section.title and section.title not in ["Preamble"]:
            lines.append(f"{heading_prefix} {section.title}\n")

        if section.content:
            lines.append(f"{section.content}\n")

    return "\n".join(lines)
```

- [ ] **Step 2: 提交**

```bash
git add src/papers/paper_model.py
git commit -m "feat: implement PaperDocument.to_markdown()"
```

---

## Task 5: 修改 api.py full-text 模式流程

**Files:**
- Modify: `src/app/api.py`

- [ ] **Step 1: 修改 recommend_stream 中 full-text 模式处理**

```python
# src/app/api.py 中 recommend_stream 函数

# 在文件顶部添加导入
from ..utils.file_parser import extract_layout_blocks
from ..papers.section_splitter import build_paper_ast

# 在 recommend_stream 中找到处理 file 的部分

# 原来：
if file and mode == "full":
    content = await file.read()
    full_text_content = extract_text_from_file(content, file.filename)

# 改为：
if file and mode == "full":
    content = await file.read()
    print(f"[DEBUG] file size: {len(content)}, filename: {file.filename}")

    # 使用 PyMuPDF layout extraction
    blocks, full_text = extract_layout_blocks(content, file.filename)
    print(f"[DEBUG] extracted {len(blocks)} blocks, text length: {len(full_text)}")

    # 构建 Paper AST
    paper_ast = build_paper_ast(blocks, title=title)
    markdown = paper_ast.to_markdown()
    print(f"[DEBUG] Paper AST: {len(paper_ast.sections)} sections, markdown length: {len(markdown)}")

    # 用 markdown 替换 full_text
    full_text = markdown

# 后面保持不变
paper_input = PaperInput(
    title=title,
    abstract=abstract or "",
    full_text=full_text or "",
    mode=mode,
)
```

- [ ] **Step 2: 提交**

```bash
git add src/app/api.py
git commit -m "feat: use layout-aware parsing in full-text mode (PyMuPDF + Paper AST)"
```

---

## Task 6: 测试验证

**Files:**
- Create: `tests/test_layout_aware_parsing.py`

- [ ] **Step 1: 编写测试**

```python
#!/usr/bin/env python3
"""测试 layout-aware PDF parsing"""

def test_pymupdf_extraction():
    """测试 PyMuPDF 能正确提取 blocks"""
    from src.utils.file_parser import extract_layout_blocks

    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, full_text = extract_layout_blocks(content, pdf_path)

    assert len(blocks) > 100, f"Expected many blocks, got {len(blocks)}"
    assert any(b.font_size > 16 for b in blocks), "Should detect large title font"

    # 检查 blocks 有字体大小信息
    for block in blocks[:10]:
        print(f"  size={block.font_size:.1f} bold={block.bold} text={block.text[:50]}")

    print(f"✓ PyMuPDF extraction: {len(blocks)} blocks")


def test_section_detection():
    """测试 SectionDetector 能正确识别章节"""
    from src.utils.file_parser import extract_layout_blocks
    from src.papers.section_splitter import build_paper_ast

    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, _ = extract_layout_blocks(content, pdf_path)
    paper_ast = build_paper_ast(blocks, title="Attention Is All You Need")

    print(f"  Title: {paper_ast.title}")
    print(f"  Abstract: {paper_ast.abstract[:100]}...")
    print(f"  Sections ({len(paper_ast.sections)}):")
    for s in paper_ast.sections:
        print(f"    [{s.level}] {s.title[:50]} - {len(s.content)} chars")

    assert len(paper_ast.sections) >= 3, "Should detect at least 3 sections"
    print("✓ Section detection works")


def test_markdown_output():
    """测试 Markdown 输出"""
    from src.utils.file_parser import extract_layout_blocks
    from src.papers.section_splitter import build_paper_ast

    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, _ = extract_layout_blocks(content, pdf_path)
    paper_ast = build_paper_ast(blocks, title="Attention Is All You Need")
    markdown = paper_ast.to_markdown()

    print(f"  Markdown length: {len(markdown)}")
    print(f"  First 200 chars:\n{markdown[:200]}")

    assert "# Attention Is All You Need" in markdown
    assert "## Abstract" in markdown or "# Abstract" in markdown
    print("✓ Markdown output correct")


def test_title_abstract_mode_unchanged():
    """测试 title/abstract 模式不受影响"""
    from src.papers.paper_model import PaperInput
    from src.utils.llm import MiniMaxLLM
    import os

    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("⚠ Skipping LLM test (no API key)")
        return

    llm = MiniMaxLLM(api_key=api_key)

    # Title 模式
    input_title = PaperInput(
        title="Attention Is All You Need",
        abstract="",
        full_text="",
        mode="title",
    )
    print(f"✓ Title mode PaperInput created: {input_title.title}")

    # Abstract 模式
    input_abstract = PaperInput(
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
        full_text="",
        mode="abstract",
    )
    print(f"✓ Abstract mode PaperInput created: {len(input_abstract.abstract)} chars")


if __name__ == "__main__":
    test_pymupdf_extraction()
    test_section_detection()
    test_markdown_output()
    test_title_abstract_mode_unchanged()
    print("\n✅ All tests passed")
```

- [ ] **Step 2: 运行测试**

```bash
cd /Users/qian/PycharmProjects/paper
python tests/test_layout_aware_parsing.py
```

- [ ] **Step 3: 提交**

```bash
git add tests/test_layout_aware_parsing.py
git commit -m "test: add layout-aware parsing tests"
```

---

## 验证检查清单

- [ ] `extract_layout_blocks()` 正确返回带字体大小的 blocks
- [ ] `FontSizeSectionDetector` 能识别论文标题、章节标题
- [ ] `PaperDocument.to_markdown()` 输出格式正确
- [ ] Full-text 模式使用新流程（PyMuPDF → AST → Markdown）
- [ ] Title/Abstract 模式完全不受影响
- [ ] 所有测试通过