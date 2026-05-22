"""测试 layout-aware parsing (PyMuPDF)"""

from src.utils.file_parser import extract_layout_blocks
from src.papers.section_splitter import build_paper_ast
from src.papers.paper_model import PaperInput


def test_pymupdf_extraction():
    """测试 PyMuPDF 能正确提取 blocks"""
    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, full_text = extract_layout_blocks(content, pdf_path)

    assert len(blocks) > 100, f"Expected many blocks, got {len(blocks)}"
    assert any(b.font_size > 16 for b in blocks), "Should detect large title font"

    print(f"✓ PyMuPDF extraction: {len(blocks)} blocks")


def test_section_detection():
    """测试 SectionDetector 能正确识别章节"""
    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, _ = extract_layout_blocks(content, pdf_path)
    paper_ast = build_paper_ast(blocks, title="Attention Is All You Need")

    assert len(paper_ast.sections) >= 3, f"Should detect at least 3 sections, got {len(paper_ast.sections)}"
    print(f"✓ Section detection works: {len(paper_ast.sections)} sections")


def test_markdown_output():
    """测试 Markdown 输出"""
    pdf_path = '/Users/qian/Documents/论文/1706.03762v7.pdf'
    with open(pdf_path, 'rb') as f:
        content = f.read()

    blocks, _ = extract_layout_blocks(content, pdf_path)
    paper_ast = build_paper_ast(blocks, title="Attention Is All You Need")
    markdown = paper_ast.to_markdown()

    assert "# Attention Is All You Need" in markdown
    print(f"✓ Markdown output correct: {len(markdown)} chars")


def test_title_abstract_mode_unchanged():
    """测试 title/abstract 模式不受影响"""
    # Title 模式
    input_title = PaperInput(
        title="Attention Is All You Need",
        abstract="",
        full_text="",
        mode="title",
    )
    assert input_title.title == "Attention Is All You Need"
    print("✓ Title mode PaperInput works")

    # Abstract 模式
    input_abstract = PaperInput(
        title="Attention Is All You Need",
        abstract="The dominant sequence transduction models...",
        full_text="",
        mode="abstract",
    )
    assert len(input_abstract.abstract) > 0
    print("✓ Abstract mode PaperInput works")


if __name__ == "__main__":
    test_pymupdf_extraction()
    test_section_detection()
    test_markdown_output()
    test_title_abstract_mode_unchanged()
    print("\n✓ All tests passed")