"""期刊模型测试"""
import pytest
from src.journals.journal_model import Journal, JournalMatch


def test_journal_model():
    journal = Journal(
        journal_id="tpami",
        journal_name="IEEE TPAMI",
        publisher="IEEE",
        subject_tags=["cv", "ai"],
        keywords=["deep learning", "neural network"],
        scope_text="Computer vision and pattern recognition",
        oa_type="subscription",
    )
    assert journal.journal_id == "tpami"
    assert "cv" in journal.subject_tags


def test_journal_build_profile():
    journal = Journal(
        journal_id="tpami",
        journal_name="IEEE TPAMI",
        scope_text="Computer vision",
        keywords=["cv"],
        subject_tags=["cv"],
    )
    profile = journal.build_profile()
    assert "IEEE TPAMI" in profile
    assert "Computer vision" in profile