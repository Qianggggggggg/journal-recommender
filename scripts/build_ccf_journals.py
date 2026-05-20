#!/usr/bin/env python3
"""
Parse CCF journal directory PDF and generate journals_ccf.jsonl

Fast version: uses existing journal DB for scope matching, skips OpenAlex API.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from PyPDF2 import PdfReader

# ============================================================================
# Configuration
# ============================================================================

PDF_PATH = "/Users/qian/Downloads/第七版中国计算机学会推荐国际学术会议和期刊目录（正式版）.pdf"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_ccf.jsonl"
JOURNALS_DB_PATH = "/Users/qian/PycharmProjects/paper/data/processed/journals.jsonl"

# Journal page ranges (0-indexed)
JOURNAL_PAGES = {
    "计算机体系结构/并行与分布计算/存储系统": [(1, 2), (2, 3), (3, 4)],
    "计算机网络": [(9, 10), (10, 11), (11, 12)],
    "网络与信息安全": [(16, 17), (17, 18), (18, 19)],
    "软件工程/系统软件/程序设计语言": [(23, 24), (24, 25), (25, 26)],
    "数据库/数据挖掘/内容检索": [(31, 32), (32, 33), (33, 34)],
    "计算机科学理论": [(37, 38), (38, 39), (39, 40)],
    "计算机图形学与多媒体": [(43, 44), (44, 45), (45, 46)],
    "人工智能": [(50, 51), (51, 52), (52, 53)],
    "人机交互与普适计算": [(60, 61), (61, 62), (62, 63)],
    "交叉/综合/新兴": [(66, 67), (67, 68), (68, 69)],
}

CLASS_MAP = {0: "A", 1: "B", 2: "C"}

# ============================================================================
# Load existing journals
# ============================================================================

def load_existing_journals() -> Dict[str, dict]:
    """Load journals.jsonl into dict keyed by lower name and common abbrevs."""
    journal_map = {}
    try:
        with open(JOURNALS_DB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line.strip())
                name = j.get("journal_name", "").lower().strip()
                if name:
                    journal_map[name] = j
    except Exception as e:
        print(f"Warning: Could not load journals: {e}", file=sys.stderr)
    return journal_map


# ============================================================================
# PDF Parsing
# ============================================================================

def parse_journal_pages():
    """Extract raw journal entries from PDF."""
    print(f"Reading PDF: {PDF_PATH}")
    reader = PdfReader(PDF_PATH)

    all_journals = []

    for subject_area, page_ranges in JOURNAL_PAGES.items():
        for class_idx, (page_idx, _) in enumerate(page_ranges):
            ccf_rating = CLASS_MAP[class_idx]
            page = reader.pages[page_idx]
            text = page.extract_text() or ""

            entries = extract_from_text(text, ccf_rating, subject_area)
            for e in entries:
                e["subject_tags"] = [subject_area]
            all_journals.extend(entries)

    return all_journals


def extract_from_text(text: str, ccf_rating: str, subject_area: str) -> List[dict]:
    """Parse journal entries from page text."""
    entries = []
    lines = text.split('\n')

    # Find header line index
    header_idx = -1
    for i, line in enumerate(lines):
        if '序号' in line and ('期刊简称' in line or '期刊全称' in line):
            header_idx = i
            break

    if header_idx < 0:
        return entries

    # Join continuation lines (entries that span multiple lines)
    # An entry has: number, abbreviation, full name, publisher, URL
    # Long full names can span multiple lines
    merged_lines = []
    current_entry = ""

    i = header_idx + 1
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        # Skip section headers
        if any(kw in line for kw in ['中国计算机学会', '推荐国际学术', '会议简称', '会议全称',
                                     '序号', '一、A类', '二、B类', '三、C类']):
            if 'ACM' not in line and 'IEEE' not in line and 'Springer' not in line and 'Elsevier' not in line:
                continue

        # Check if this line has a URL
        has_url = 'http' in line

        if has_url:
            # Flush any pending entry first
            if current_entry:
                merged_lines.append(current_entry)
                current_entry = ""
            # This is a complete entry (or start of one)
            merged_lines.append(line)
        else:
            # Continuation line - append to current entry
            if current_entry:
                current_entry += " " + line
            else:
                # No current entry but no URL - this is unusual, treat as new
                current_entry = line

    # Flush any remaining
    if current_entry:
        merged_lines.append(current_entry)

    # Process merged lines
    for line in merged_lines:
        line = line.strip()
        if not line:
            continue

        entry = parse_line(line, ccf_rating)
        if entry:
            entries.append(entry)

    return entries


def parse_line(line: str, ccf_rating: str) -> Optional[dict]:
    """Parse single journal line."""
    if len(line) < 5:
        return None

    # Skip conference lines
    skip = [' symposium', ' conference', ' proceedings', 'ACMSIG', 'IEEE/', 'USENIX']
    if any(s in line for s in skip) and 'ACM' not in line and 'IEEE' not in line:
        return None

    # Find URL
    url_match = re.search(r'http[^\s]+', line)
    url = url_match.group(0) if url_match else ""
    if not url:
        return None

    # Find publisher - it's right before the URL with minimal/no space
    pattern = r'(ACM|IEEE|Springer|Elsevier|Wiley|MIT Press|IOSPress|SIAM|CCF|Cambridge University Press|Taylor & Francis|Oxford University Press|World Scientific|科学出版社|清华大学出版社)\s*http'
    m = re.search(pattern, line)
    if not m:
        return None

    publisher = m.group(1)
    before_pub = line[:m.start()].strip()

    # Extract sequence number
    seq_match = re.match(r'^[\d一二三四五六七八九十]+[、.．、]?\s*', before_pub)
    if seq_match:
        before_pub = before_pub[seq_match.end():]

    tokens = before_pub.split()
    abbrev = ""
    full_name = ""

    if len(tokens) >= 2:
        abbrev = tokens[0].strip()
        abbrev = re.sub(r'[,，.．、\s]+', '', abbrev)
        full_name = ' '.join(tokens[1:])
    elif len(tokens) == 1:
        token = tokens[0].strip()
        # Handle cases where abbreviation and full name are concatenated without space
        # e.g., "TODAESACMTransactionson..." -> TODAES + ACM Transactions on...
        # The "publisher name" inside the journal name is the key indicator
        # Look for "ACM", "IEEE", "Springer", etc. embedded in the token
        embedded_pub_match = re.search(r'(ACM|IEEE|Springer|Elsevier|Wiley|MIT Press|IOSPress|SIAM|CCF)', token)
        if embedded_pub_match and embedded_pub_match.start() > 2:
            # Split at the publisher name
            abbrev = token[:embedded_pub_match.start()].strip()
            full_name = token[embedded_pub_match.start():].strip()
            # Clean up full_name - remove leading/trailing punctuation
            full_name = full_name.strip('.,，、')
        else:
            # Check for "IEEE" in the middle (like TCADIEEE)
            ieee_pos = token.find('IEEE')
            if ieee_pos > 0:
                abbrev = token[:ieee_pos].strip()
                full_name = token[ieee_pos:].strip()
            else:
                # Look for known journal words
                known_words = ['Transactions', 'Journal', 'Proceedings', 'Letters', 'Review',
                              'Systems', 'Computing', 'Computer', 'Science', 'Engineering', 'Design']
                split_pos = -1
                for word in known_words:
                    pos = token.find(word)
                    if pos > 0:
                        if split_pos < 0 or pos < split_pos:
                            split_pos = pos
                if split_pos > 2:
                    abbrev = token[:split_pos].strip()
                    abbrev_match = re.match(r'^([A-Z]{2,6}[0-9]*)', abbrev)
                    if abbrev_match:
                        abbrev = abbrev_match.group(1)
                    else:
                        abbrev = ""
                        full_name = token
                else:
                    abbrev = ""
                    full_name = token
    else:
        return None

    # Clean up full_name
    full_name = re.sub(r'\s+', ' ', full_name).strip()
    full_name = full_name.strip('.,，、')

    if len(full_name) < 3 and abbrev and tokens:
        rest = before_pub[len(tokens[0]):] if len(tokens) >= 1 else before_pub
        if abbrev in rest:
            candidate = rest[rest.find(abbrev) + len(abbrev):].strip()
            if len(candidate) > 5:
                full_name = candidate

    if len(full_name) < 3:
        candidate = re.sub(r'[,，.．、]+', '', before_pub).strip()
        if len(candidate) > 3:
            full_name = candidate

    if len(full_name) < 3:
        return None

    # Clean abbreviation
    abbrev = re.sub(r'[,，.．、\s]+', '', abbrev).strip()
    if abbrev and not re.match(r'^[A-Z0-9]{2,10}$', abbrev):
        m2 = re.match(r'^([A-Z]{2,6}[0-9]*)', before_pub)
        if m2:
            abbrev = m2.group(1)

    if abbrev:
        journal_id = abbrev.lower().replace('.', '_').replace('-', '_')
    else:
        journal_id = re.sub(r'[^a-z0-9]', '_', full_name.lower()[:30])

    # Clean up full_name: insert spaces at lowercase-to-uppercase transitions
    # (except for known acronyms like IEEE, ACM, etc.)
    full_name = _add_spaces_to_name(full_name)

    return {
        "journal_id": journal_id,
        "journal_name": full_name,
        "abbrev": abbrev,
        "publisher": publisher,
        "ccf_rating": ccf_rating,
        "homepage_url": url,
        "submission_url": "",
    }


def _add_spaces_to_name(name: str) -> str:
    """Add spaces to journal name at word boundaries."""
    if not name:
        return name
    # Don't modify if already has spaces
    if ' ' in name:
        return name

    # Handle CamelCase: insert space at lowercase->uppercase boundary
    # e.g., "IEEETransactionsonComputers" -> "IEEE Transactions on Computers"
    result = []
    i = 0
    while i < len(name):
        if i > 0 and name[i].isupper():
            prev_lower = name[i-1].islower()
            if prev_lower:
                result.append(' ')
        result.append(name[i])
        i += 1

    # Clean up common patterns:
    # "VLSI" stays together, "I" (roman numeral) stays with preceding word
    # Fix cases like "Systems IRegular" -> "Systems I: Regular"
    name = ''.join(result)
    # Post-process: handle "I:Regular" type patterns
    name = re.sub(r'(\w)\s+I:\s*([A-Z])', r'\1 I: \2', name)
    name = re.sub(r'(\w)\s+I\s+([A-Z])', r'\1 I \2', name)

    return name.strip()


# ============================================================================
# Scope enrichment
# ============================================================================

FALLBACK_SCOPES = {
    "计算机体系结构/并行与分布计算/存储系统": "computer architecture, parallel computing, distributed systems, storage systems, high-performance computing, cloud computing, GPU computing, multi-core processors, microarchitecture",
    "计算机网络": "computer networks, network protocols, wireless communications, mobile computing, internet architecture, network security, distributed systems, sensor networks, 5G, SDN",
    "网络与信息安全": "network security, information security, cryptography, privacy, cyber security, intrusion detection, secure computing, malware analysis, digital forensics",
    "软件工程/系统软件/程序设计语言": "software engineering, programming languages, system software, software verification, software testing, software architecture, compiler design, programming methodology, formal methods",
    "数据库/数据挖掘/内容检索": "database systems, data mining, information retrieval, knowledge engineering, big data, data warehousing, query processing, NoSQL, text mining",
    "计算机科学理论": "theoretical computer science, algorithms, computational complexity, logic, automata theory, computability, formal methods, cryptography theory",
    "计算机图形学与多媒体": "computer graphics, visualization, multimedia, image processing, computer vision, virtual reality, augmented reality, rendering, visual analytics",
    "人工智能": "artificial intelligence, machine learning, deep learning, natural language processing, computer vision, robotics, knowledge representation, neural networks, reinforcement learning",
    "人机交互与普适计算": "human-computer interaction, ubiquitous computing, pervasive computing, user interface design, interaction design, accessibility, mobile devices, social computing",
    "交叉/综合/新兴": "interdisciplinary research, emerging technologies, cross-domain applications, computational intelligence, smart systems, novel computing paradigms",
}

# Abbreviation to full name mapping for knowledge base lookup
ABBREV_MAP = {
    "tocs": "ACM Transactions on Computer Systems",
    "tos": "ACM Transactions on Storage",
    "tcad": "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems",
    "tc": "IEEE Transactions on Computers",
    "tpds": "IEEE Transactions on Parallel and Distributed Systems",
    "taco": "ACM Transactions on Architecture and Code Optimization",
    "jsac": "IEEE Journal on Selected Areas in Communications",
    "tmc": "IEEE Transactions on Mobile Computing",
    "ton": "IEEE/ACM Transactions on Networking",
    "tods": "ACM Transactions on Database Systems",
    "tois": "ACM Transactions on Information Systems",
    "tkde": "IEEE Transactions on Knowledge and Data Engineering",
    "toplas": "ACM Transactions on Programming Languages and Systems",
    "tosem": "ACM Transactions on Software Engineering and Methodology",
    "tse": "IEEE Transactions on Software Engineering",
    "tsc": "IEEE Transactions on Services Computing",
    "tifs": "IEEE Transactions on Information Forensics and Security",
    "tochi": "ACM Transactions on Computer-Human Interaction",
    "tvcg": "IEEE Transactions on Visualization and Computer Graphics",
    "tmm": "IEEE Transactions on Multimedia",
    "ai": "Artificial Intelligence",
    "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "jmlr": "Journal of Machine Learning Research",
    "jacm": "Journal of the ACM",
    "vldbj": "The VLDB Journal",
    "alg": "Algorithmica",
    "spe": "Software: Practice and Experience",
    "cn": "Computer Networks",
    "dmkd": "Data Mining and Knowledge Discovery",
    "acm": "ACM Computing Surveys",
}


def get_scope(journal: dict, db: Dict[str, dict]) -> str:
    """Get scope_text from existing DB or fallback."""
    name = journal.get("journal_name", "").lower().strip()

    # Direct match
    if name in db:
        scope = db[name].get("scope_text", "")
        if scope and len(scope) >= 15:
            return scope

    # Partial match
    for known, data in db.items():
        if name in known or known in name:
            scope = data.get("scope_text", "")
            if scope and len(scope) >= 15:
                return scope

    # Abbreviation match
    abbrev = journal.get("abbrev", "").lower().replace('.', '').replace('-', '')
    if abbrev and abbrev in ABBREV_MAP:
        full_name = ABBREV_MAP[abbrev].lower()
        if full_name in db:
            scope = db[full_name].get("scope_text", "")
            if scope and len(scope) >= 15:
                return scope

    # Fallback from subject area
    tags = journal.get("subject_tags", [])
    if tags:
        scope = FALLBACK_SCOPES.get(tags[0], "computer science research")
    else:
        scope = "computer science research"

    return scope


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("CCF Journal Parser (fast mode)")
    print("=" * 60)

    # Load existing DB
    print("\nLoading existing journals...")
    db = load_existing_journals()
    print(f"  Loaded {len(db)} entries")

    # Parse PDF
    print("\nParsing PDF...")
    raw = parse_journal_pages()
    print(f"  Extracted {len(raw)} journal entries")

    # Count by rating
    rating_counts = {"A": 0, "B": 0, "C": 0}
    for j in raw:
        r = j.get("ccf_rating", "C")
        rating_counts[r] = rating_counts.get(r, 0) + 1
    print(f"  A: {rating_counts['A']}, B: {rating_counts['B']}, C: {rating_counts['C']}")

    # Enrich and finalize
    print("\nEnriching and writing...")
    ccf_to_quartile = {"A": "Q1", "B": "Q2", "C": "Q3"}
    scope_db = 0
    scope_fallback = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for j in raw:
            # Get scope
            scope = get_scope(j, db)
            if scope in [FALLBACK_SCOPES.get(t, "computer science research") for t in j.get("subject_tags", [])]:
                scope_fallback += 1
            else:
                scope_db += 1

            # Finalize
            entry = {
                "journal_id": j.get("journal_id", ""),
                "journal_name": j.get("journal_name", ""),
                "publisher": j.get("publisher", ""),
                "subject_tags": j.get("subject_tags", []),
                "ccf_rating": j.get("ccf_rating", "C"),
                "quartile": ccf_to_quartile.get(j.get("ccf_rating", "C"), "Q3"),
                "scope_text": scope,
                "submission_url": j.get("homepage_url", "") or j.get("submission_url", ""),
                "homepage_url": j.get("homepage_url", ""),
                "keywords": [],
                "oa_type": "subscription",
                "impact_like_score": 0.0,
                "review_time": "",
                "apc": 0.0,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nScope sources: DB={scope_db}, fallback={scope_fallback}")

    # Build index
    print("\nBuilding FAISS index...")
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "/Users/qian/PycharmProjects/paper/scripts/build_journal_index.py"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            print("  FAISS index built successfully")
        else:
            print(f"  Index build returned: {result.returncode}")
    except Exception as e:
        print(f"  Index build skipped: {e}")

    # Final stats
    with open(OUTPUT_PATH, "r") as f:
        total = sum(1 for _ in f)

    print(f"\nOutput: {OUTPUT_PATH}")
    print(f"Total journals: {total}")
    print("Done!")


if __name__ == "__main__":
    main()