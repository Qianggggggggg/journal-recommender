"""推荐结果 PDF 导出"""
import logging
import platform
import sys
from typing import List, Optional
from datetime import datetime

from fpdf import FPDF

from ..journals.journal_model import JournalMatch
from ..papers.paper_model import PaperProfile

logger = logging.getLogger(__name__)


def _get_cjk_font_paths():
    """获取跨平台 CJK 字体路径列表"""
    system = platform.system()
    paths = []

    if system == "Darwin":
        paths.extend([
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ])
    elif system == "Windows":
        paths.extend([
            "C:/Windows/Fonts/arialuni.ttf",
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
        ])
    else:  # Linux
        paths.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ])

    # 当前工作目录下的字体
    import os
    cwd = os.getcwd()
    paths.extend([
        os.path.join(cwd, "fonts", "Arial Unicode.ttf"),
        os.path.join(cwd, "fonts", "arialuni.ttf"),
    ])

    return paths


class PDFExporter:
    """推荐结果 PDF 导出器"""

    def export(
        self,
        title: str,
        abstract: str,
        recommendations: List[JournalMatch],
        paper_profile: Optional[PaperProfile] = None,
    ) -> bytes:
        """
        生成推荐结果 PDF

        Args:
            title: 论文标题
            abstract: 论文摘要
            recommendations: 推荐期刊列表
            paper_profile: 论文特征（可选）

        Returns:
            PDF 字节流
        """
        pdf = PDFReport()
        pdf.add_page()

        # 标题
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.chapter_title("论文投稿推荐报告")
        pdf.ln(3)

        # 论文信息
        pdf.chapter_body("论文信息")
        pdf.info_row("论文标题", title)

        if abstract:
            # 不截取摘要，完整显示
            pdf.info_row("摘要", abstract)

        if paper_profile:
            if paper_profile.research_area:
                pdf.info_row("研究领域", ", ".join(paper_profile.research_area))
            if paper_profile.method_type:
                pdf.info_row("方法类型", paper_profile.method_type)
            if paper_profile.keywords:
                pdf.info_row("关键词", ", ".join(paper_profile.keywords))
            if paper_profile.techniques:
                pdf.info_row("技术方法", ", ".join(paper_profile.techniques))
            if paper_profile.ccf_research_area:
                pdf.info_row("CCF领域", ", ".join(paper_profile.ccf_research_area))

        pdf.ln(3)

        # 推荐期刊列表
        pdf.chapter_body("推荐期刊列表")

        # 论文质量信息（评级、强度、置信度）- 放在推荐列表上方
        if paper_profile and paper_profile.quality_level:
            quality_parts = []
            if paper_profile.quality_level:
                quality_parts.append(f"论文评级: {paper_profile.quality_level}")
            if paper_profile.paper_strength is not None:
                quality_parts.append(f"论文强度: {int(paper_profile.paper_strength * 100)}%")
            if paper_profile.quality_confidence is not None and paper_profile.quality_confidence > 0:
                quality_parts.append(f"评估置信度: {int(paper_profile.quality_confidence * 100)}%")
            if quality_parts:
                pdf.set_font("CJK", size=10)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 5, " | ".join(quality_parts))
                pdf.ln(6)

        pdf.ln(2)

        for i, match in enumerate(recommendations, 1):
            journal = match.journal
            pdf.journal_entry(
                rank=i,
                name=journal.journal_name,
                ccf=journal.ccf_rating or "N/A",
                score=match.score,
                reasons=match.match_reasons,
                submission_url=journal.submission_url or "",
            )
            pdf.ln(3)

        # 页脚
        pdf.ln(10)
        pdf.set_font("CJK", size=8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")
        pdf.ln(3)
        pdf.cell(0, 5, "由论文投稿推荐系统自动生成", align="C")

        return pdf.output()


class PDFReport(FPDF):
    """PDF 报告样式"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        # 添加中文字体支持 - 尝试多个跨平台路径
        font_added = False
        for font_path in _get_cjk_font_paths():
            try:
                self.add_font('CJK', '', font_path)
                font_added = True
                break
            except Exception:
                continue

        if not font_added:
            logger.warning(
                "未找到可用的 CJK 中文字体，PDF 中文可能无法正确渲染。"
                " 请安装 Arial Unicode.ttf 或其他 CJK 字体。"
            )

    def header(self):
        """页眉"""
        pass

    def footer(self):
        """页脚"""
        pass

    def chapter_title(self, title: str):
        """章节标题"""
        self.set_font("CJK", size=14)
        self.set_text_color(25, 25, 112)  # 深蓝色
        self.cell(0, 8, title)
        self.ln(8)

        # 下划线
        self.set_draw_color(25, 25, 112)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def chapter_body(self, text: str):
        """正文段落标题"""
        self.set_font("CJK", size=11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, text)
        self.ln(6)

    def info_row(self, label: str, value: str):
        """信息行"""
        self.set_font("CJK", size=10)
        self.set_text_color(60, 60, 60)
        # 紧凑布局：标签和值之间用空格分隔，不占用固定宽度
        self.multi_cell(0, 5, f"{label} {value}")
        self.ln(1)

    def journal_entry(
        self,
        rank: int,
        name: str,
        ccf: str,
        score: float,
        reasons: List[str],
        submission_url: str,
    ):
        """期刊条目"""
        self.set_font("CJK", size=11)
        self.set_text_color(0, 51, 102)

        # 序号和名称 - CJK字体不支持bold
        self.set_font("CJK", size=11)
        self.cell(0, 6, f"{rank}. {name}")
        self.ln(6)

        # 标签行
        self.set_font("CJK", size=9)
        self.set_text_color(100, 100, 100)

        # 彩色标签 - 与前端一致的 CCF 等级颜色
        ccf_colors = {
            "A": (212, 160, 23),   # 金色
            "B": (30, 64, 175),    # 蓝色
            "C": (22, 163, 74),    # 绿色
            "D": (107, 114, 128),  # 灰色
        }
        bg_color = ccf_colors.get(ccf.upper(), (107, 114, 128))

        self.set_fill_color(*bg_color)
        self.set_text_color(255, 255, 255)
        self.cell(18, 5, f" CCF-{ccf} ", fill=True)

        self.set_fill_color(220, 220, 220)
        self.set_text_color(60, 60, 60)
        self.cell(28, 5, f" 匹配度:{score:.2f} ", fill=True)

        self.ln(6)
        self.set_text_color(60, 60, 60)

        # 推荐理由
        if reasons:
            self.set_font("CJK", size=9)
            for reason in reasons[:10]:
                # 不截取理由，完整显示
                reason_display = reason
                # 确保在左边界开始，如空间不足则新建页
                if self.get_y() > 270:
                    self.add_page()
                self.set_x(10)
                # 使用 cell + Ln 代替 multi_cell，避免对齐问题
                self.cell(5, 4, "•")
                self.cell(0, 4, reason_display)
                self.ln(4)

        # 投稿链接
        if submission_url:
            self.ln(1)
            self.set_font("CJK", size=8)
            self.set_text_color(0, 102, 204)
            self.cell(0, 4, f"投稿: {submission_url}")

        self.ln(4)

        # 分隔线
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())