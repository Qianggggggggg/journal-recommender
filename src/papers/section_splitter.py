"""章节切分（全文模式）"""
import re
from typing import Dict

from .paper_model import SectionSplitResult


class SectionSplitter:
    """论文章节切分器"""

    # 常见章节标题模式
    SECTION_PATTERNS = {
        "introduction": [r"1\s*\.?\s*Introduction", r"1\s*引言", r"引\s*言"],
        "method": [r"2\s*\.?\s*(Proposed|Method|Approach)", r"2\s*方法", r"算法", r"技术"],
        "experiment": [r"3\s*\.?\s*(Experiment|Evaluation|Results)", r"3\s*实验", r"评估", r"结果"],
        "conclusion": [r"(Conclusion|Discussion|Summary)", r"结[论语]", r"总结"],
    }

    def split(self, full_text: str) -> SectionSplitResult:
        """切分论文章节"""
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