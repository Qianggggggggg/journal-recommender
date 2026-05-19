"""期刊存储测试"""
import pytest
import tempfile
import os
from src.journals.journal_model import Journal
from src.journals.journal_store import JournalStore


def test_journal_store_add_and_list():
    """测试添加和列出期刊"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(
            store_path=os.path.join(tmpdir, "journals.jsonl"),
        )
        journal = Journal(
            journal_id="test-journal",
            journal_name="Test Journal",
            subject_tags=["ai"],
            oa_type="full_oa",
        )
        store.add_journal(journal)
        assert store.count == 1

        listed = store.list_journals()
        assert len(listed) == 1
        assert listed[0].journal_id == "test-journal"


def test_journal_store_search_by_text():
    """测试文本搜索"""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JournalStore(
            store_path=os.path.join(tmpdir, "journals.jsonl"),
        )
        journal = Journal(
            journal_id="ai-journal",
            journal_name="AI Journal",
            subject_tags=["ai", "ml"],
            keywords=["machine learning", "deep learning"],
            journal_profile="AI Journal machine learning deep learning",
        )
        store.add_journal(journal)

        results = store.search_by_text("machine learning", top_k=5)
        assert len(results) >= 1
        assert results[0][0].journal_id == "ai-journal"