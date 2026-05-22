"""章节切分（全文模式）"""
import re
from typing import Dict, List, Tuple

from .paper_model import Section, PaperDocument, Block
from ..utils.file_parser import LayoutBlock


def _convert_to_block(lb: LayoutBlock) -> Block:
    """将 LayoutBlock 转换为 Block (pydantic model)"""
    return Block(
        text=lb.text,
        font_size=lb.font_size,
        font_name=lb.font_name,
        bold=lb.bold,
        x0=lb.x0,
        y0=lb.y0,
        x1=lb.x1,
        y1=lb.y1,
        page_number=lb.page,
    )


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
                        blocks=[_convert_to_block(b) for b in current_section["blocks"]],
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
                blocks=[_convert_to_block(b) for b in current_section["blocks"]],
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
        all_blocks=[_convert_to_block(b) for b in blocks],
    )


class SectionSplitter:
    """论文章节切分器"""

    # 常见章节标题模式
    SECTION_PATTERNS = {
        "introduction": [r"1\s*\.?\s*Introduction", r"1\s*引言", r"引\s*言"],
        "method": [r"2\s*\.?\s*(Proposed|Method|Approach)", r"2\s*方法", r"算法", r"技术"],
        "experiment": [r"3\s*\.?\s*(Experiment|Evaluation|Results)", r"3\s*实验", r"评估", r"结果"],
        "conclusion": [r"(Conclusion|Discussion|Summary)", r"结[论语]", r"总结"],
    }

    def split(self, full_text: str) -> "SectionSplitResult":
        """切分论文章节"""
        from .paper_model import SectionSplitResult

        lines = full_text.split("\n")
        sections: Dict[str, list] = {
            "introduction": [],
            "method": [],
            "experiment": [],
            "conclusion": [],
            "other": [],
        }
        current_section = "other"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测章节标题
            detected = self._detect_section(line)
            if detected:
                current_section = detected

            # 跳过页眉页脚
            if self._is_header_footer(line):
                continue

            sections[current_section].append(line)

        return SectionSplitResult(
            introduction=" ".join(sections["introduction"]),
            method=" ".join(sections["method"]),
            experiment=" ".join(sections["experiment"]),
            conclusion=" ".join(sections["conclusion"]),
            other=" ".join(sections["other"]),
        )

    def _detect_section(self, line: str) -> str:
        """检测章节类型"""
        for section_name, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    return section_name
        return ""

    def _is_header_footer(self, line: str) -> bool:
        """判断是否页眉页脚"""
        if len(line) < 10:
            return True
        if re.match(r"^\d+\s+\d+$", line):
            return True
        return False