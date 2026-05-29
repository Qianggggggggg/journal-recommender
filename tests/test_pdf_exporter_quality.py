"""测试 PDF 导出包含论文质量信息"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.pdf_exporter import PDFExporter
from src.papers.paper_model import PaperProfile
from src.journals.journal_model import JournalMatch, Journal


def create_mock_journal():
    """创建测试用期刊"""
    return Journal(
        journal_id="test-journal-1",
        journal_name="Test Journal",
        publisher="Test Publisher",
        scope_text="Test scope",
        subject_tags=["AI"],
        keywords=["test"],
        oa_type="subscription",
        submission_url="https://test.com",
        homepage_url="https://test.com",
        quartile="Q1",
        impact_like_score=5.0,
        review_time="3 months",
        apc=1000.0,
        ccf_rating="B",
    )


def test_pdf_contains_quality_info():
    """验证 PDF 导出包含论文质量信息（评级、强度、置信度）"""
    exporter = PDFExporter()

    # 创建带有质量信息的 PaperProfile
    profile = PaperProfile(
        title="Test Paper",
        abstract="Test abstract",
        research_area=["人工智能"],
        paper_strength=0.75,
        quality_level="A",
        quality_confidence=0.85,
        readiness="Ready",
    )

    journal = create_mock_journal()
    match = JournalMatch(
        journal=journal,
        score=0.92,
        confidence=0.9,
        match_reasons=["CCF领域匹配", "方法契合"],
        matched_fields=["research_area", "method_type"],
    )

    # 生成 PDF
    pdf_bytes = exporter.export(
        title="Test Paper",
        abstract="Test abstract",
        recommendations=[match],
        paper_profile=profile,
    )

    assert pdf_bytes is not None, "PDF should not be None"
    assert len(pdf_bytes) > 1000, f"PDF too small: {len(pdf_bytes)} bytes"

    # 将 PDF 转为文本检查（使用 PyMuPDF 提取）
    import fitz
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf_text_parts = []
    for page in pdf_doc:
        pdf_text_parts.append(page.get_text())
    pdf_doc.close()
    pdf_text = "\n".join(pdf_text_parts)

    print(f"PDF size: {len(pdf_bytes)} bytes")
    print(f"Extracted text: {pdf_text}")

    # 检查质量信息是否在 PDF 中（位于推荐期刊列表下方）
    assert "论文评级: B" in pdf_text, f"PDF should contain '论文评级: B', got: {pdf_text}"
    assert "论文强度: 75%" in pdf_text, f"PDF should contain '论文强度: 75%', got: {pdf_text}"
    assert "评估置信度: 85%" in pdf_text, f"PDF should contain '评估置信度: 85%', got: {pdf_text}"

    # 验证位置：质量信息在"推荐期刊列表"之后、期刊条目之前
    idx_quality = pdf_text.find("论文评级")
    idx_journal = pdf_text.find("Test Journal")
    idx_section = pdf_text.find("推荐期刊列表")
    assert idx_section < idx_quality < idx_journal, "Quality info should be between section header and first journal"

    # 必须包含：质量等级(Q2)、强度(75%)、置信度(85%)
    assert "B" in pdf_text, f"PDF should contain quality level B, got: {pdf_text[:300]}"
    assert "75" in pdf_text or "0.75" in pdf_text, f"PDF should contain paper_strength 75%, got: {pdf_text[:300]}"
    assert "85" in pdf_text or "0.85" in pdf_text, f"PDF should contain confidence 85%, got: {pdf_text[:300]}"

    print("✓ PDF contains quality info (level, strength, confidence)")


def test_pdf_quality_with_null_strength():
    """验证 paper_strength 为 None 时 PDF 仍能正常生成"""
    exporter = PDFExporter()

    profile = PaperProfile(
        title="Test Paper",
        abstract="Test abstract",
        paper_strength=None,  # 无质量评估
        quality_level=None,
        # quality_confidence 默认值是 0.0，不传即可
    )

    journal = create_mock_journal()
    match = JournalMatch(
        journal=journal,
        score=0.92,
        confidence=0.9,
        match_reasons=["CCF领域匹配"],
        matched_fields=["research_area"],
    )

    # 不应抛出异常
    pdf_bytes = exporter.export(
        title="Test Paper",
        abstract="Test abstract",
        recommendations=[match],
        paper_profile=profile,
    )

    assert pdf_bytes is not None, "PDF should be generated even without quality info"
    print("✓ PDF handles null paper_strength gracefully")


if __name__ == "__main__":
    test_pdf_contains_quality_info()
    test_pdf_quality_with_null_strength()
    print("\n✅ All tests passed")