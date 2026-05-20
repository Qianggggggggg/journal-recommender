"""推荐结果 PDF 导出"""
from typing import List, Optional
from datetime import datetime

from fpdf import FPDF

from ..journals.journal_model import JournalMatch
from ..papers.paper_model import PaperProfile


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
            # 截取过长摘要
            abstract_display = abstract[:500] + "..." if len(abstract) > 500 else abstract
            pdf.info_row("摘要", abstract_display)

        if paper_profile:
            if paper_profile.research_area:
                pdf.info_row("研究领域", ", ".join(paper_profile.research_area))
            if paper_profile.method_type:
                pdf.info_row("方法类型", paper_profile.method_type)
            if paper_profile.keywords:
                pdf.info_row("关键词", ", ".join(paper_profile.keywords[:8]))
            if paper_profile.techniques:
                pdf.info_row("技术方法", ", ".join(paper_profile.techniques[:5]))
            if paper_profile.ccf_research_area:
                pdf.info_row("CCF领域", ", ".join(paper_profile.ccf_research_area))

        pdf.ln(3)

        # 推荐期刊列表
        pdf.chapter_body("推荐期刊列表")
        pdf.ln(2)

        for i, match in enumerate(recommendations, 1):
            journal = match.journal
            pdf.journal_entry(
                rank=i,
                name=journal.journal_name,
                quartile=journal.quartile or "N/A",
                ccf=journal.ccf_rating or "N/A",
                score=match.score,
                reasons=match.match_reasons,
                submission_url=journal.submission_url or "",
            )
            pdf.ln(3)

        # 页脚
        pdf.ln(10)
        pdf.set_font("Helvetica", size=8)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 5, f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C")
        pdf.ln(3)
        pdf.cell(0, 5, "由论文投稿推荐系统自动生成", align="C")

        return pdf.output()


class PDFReport(FPDF):
    """PDF 报告样式"""

    def header(self):
        """页眉"""
        pass

    def footer(self):
        """页脚"""
        pass

    def chapter_title(self, title: str):
        """章节标题"""
        self.set_font("Helvetica", size=14)
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
        self.set_font("Helvetica", size=11)
        self.set_text_color(80, 80, 80)
        self.cell(0, 6, text)
        self.ln(6)

    def info_row(self, label: str, value: str):
        """信息行"""
        self.set_font("Helvetica", size=10)
        self.set_text_color(60, 60, 60)

        # 标签加粗
        self.set_font("Helvetica", size=10, style="B")
        self.cell(25, 5, label)
        self.set_font("Helvetica", size=10)
        self.multi_cell(0, 5, value)
        self.ln(1)

    def journal_entry(
        self,
        rank: int,
        name: str,
        quartile: str,
        ccf: str,
        score: float,
        reasons: List[str],
        submission_url: str,
    ):
        """期刊条目"""
        self.set_font("Helvetica", size=11)
        self.set_text_color(0, 51, 102)

        # 序号和名称
        self.set_font("Helvetica", size=11, style="B")
        self.cell(0, 6, f"{rank}. {name}")
        self.ln(6)

        # 标签行
        self.set_font("Helvetica", size=9)
        self.set_text_color(100, 100, 100)

        # 彩色标签
        self.set_fill_color(0, 51, 102)
        self.set_text_color(255, 255, 255)
        self.cell(18, 5, f" {quartile} ", fill=True)

        self.set_fill_color(102, 102, 102)
        self.cell(18, 5, f" CCF-{ccf} ", fill=True)

        self.set_fill_color(220, 220, 220)
        self.set_text_color(60, 60, 60)
        self.cell(28, 5, f" 匹配度:{score:.2f} ", fill=True)

        self.ln(6)
        self.set_text_color(60, 60, 60)

        # 推荐理由
        if reasons:
            self.set_font("Helvetica", size=9)
            for reason in reasons[:5]:
                # 截取过长理由
                reason_display = reason[:60] + "..." if len(reason) > 60 else reason
                self.cell(5, 4, "•")
                self.multi_cell(0, 4, reason_display)

        # 投稿链接
        if submission_url:
            self.ln(1)
            self.set_font("Helvetica", size=8)
            self.set_text_color(0, 102, 204)
            self.cell(0, 4, f"投稿: {submission_url}")

        self.ln(4)

        # 分隔线
        self.set_draw_color(220, 220, 220)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())