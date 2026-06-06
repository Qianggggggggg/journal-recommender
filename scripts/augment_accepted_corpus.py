#!/usr/bin/env python3
"""Augment accepted_paper corpus with 7 missing venues for holdout240 coverage.

For each missing venue, search S2 for ~5-8 real papers 2020-2025 and write
to data/accepted_papers/<journal_id>.json in the same format as existing 102 files.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

sys.path.insert(0, '.')
JOURNALS_PATH = Path('data/processed/journals.jsonl')
CORPUS_DIR = Path('data/accepted_papers')

TARGETS = [
    ('tocl',     'ACM Transactions on Computational Logic'),
    ('tos',      'ACM Transactions on Storage'),
    ('apal',     'Annals of Pure and Applied Logic'),
    ('discovercomputing', 'Discover Computing'),
    ('iets',     'IET Software'),
    ('iwc',      'Interacting with Computers'),
    ('ijgis',    'International Journal of Geographical Information Science'),
]

S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"


def s2_get(url, max_retries=3):
    key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY', '')
    headers = {'User-Agent': 'corpus-augment/1.0'}
    if key:
        headers['x-api-key'] = key
    for attempt in range(max_retries):
        try:
            r = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(r, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f'  [429] backoff {wait}s', flush=True)
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f'  [S2 {e.code}]', flush=True)
                return None
        except Exception as e:
            print(f'  [net] {e}', flush=True)
            time.sleep(2)
    return None


def search_journal_papers(journal_name, year_min=2020, year_max=2025, limit=10):
    """Search S2 for papers in a specific journal, year range."""
    q = f'"{journal_name}"'
    url = f'{S2_SEARCH}?query={urllib.parse.quote(q)}&limit={limit}&year={year_min}-{year_max}&fields=title,venue,publicationVenue,year,abstract,externalIds,authors'
    return s2_get(url)


def normalize_title(t):
    return ' '.join((t or '').casefold().split())


def main():
    print(f'=== Augmenting accepted_paper corpus for {len(TARGETS)} venues ===\n', flush=True)
    existing_titles = set()
    for f in CORPUS_DIR.glob('*.json'):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for p in data.get('papers', []):
                existing_titles.add(normalize_title(p.get('title', '')))
        except Exception:
            pass
    print(f'Existing titles in corpus: {len(existing_titles)}', flush=True)

    new_files = []
    for jid, jname in TARGETS:
        out_path = CORPUS_DIR / f'{jid}.json'
        if out_path.exists():
            print(f'[{jname}] {out_path} already exists; skipping', flush=True)
            continue
        print(f'\n[{jname}] searching S2 ...', flush=True)
        data = search_journal_papers(jname, year_min=2020, year_max=2025, limit=20)
        if not data or 'data' not in data:
            print(f'  no results', flush=True)
            continue
        papers = []
        seen = set()
        for cand in data['data']:
            v = cand.get('venue') or (cand.get('publicationVenue') or {}).get('name') or ''
            if v.casefold() != jname.casefold():
                continue
            t = cand.get('title', '')
            t_norm = normalize_title(t)
            if not t or t_norm in seen or t_norm in existing_titles:
                continue
            if not cand.get('abstract'):
                continue
            ext = cand.get('externalIds', {}) or {}
            paper = {
                'title': t,
                'abstract': cand.get('abstract', ''),
                'year': cand.get('year'),
                'source': 'semantic_scholar',
                'doi': ext.get('DOI', ''),
                'arxiv': ext.get('ArXiv', ''),
                'corpus_id': ext.get('CorpusId', ''),
                'url': f"https://www.semanticscholar.org/paper/{cand.get('paperId', '')}" if cand.get('paperId') else '',
            }
            papers.append(paper)
            seen.add(t_norm)
            if len(papers) >= 6:
                break
        if not papers:
            print(f'  no valid papers found', flush=True)
            continue
        out = {
            'journal_id': jid,
            'journal_name': jname,
            'papers': papers,
        }
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        new_files.append((jname, len(papers)))
        print(f'  -> {len(papers)} papers saved to {out_path}', flush=True)
        time.sleep(0.5)

    print(f'\n=== Summary ===')
    print(f'New corpus files created: {len(new_files)}')
    for jname, n in new_files:
        print(f'  {jname}: {n} papers')


if __name__ == '__main__':
    main()
