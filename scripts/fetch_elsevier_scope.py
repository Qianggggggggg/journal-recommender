#!/usr/bin/env python3
"""
从 ScienceDirect (Elsevier) 抓取期刊的 Aims & Scope
"""

import json
import time
import re
import requests
from bs4 import BeautifulSoup

INPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
OUTPUT_PATH = "/Users/qian/PycharmProjects/paper/data/journals_output.jsonl"
SLEEP_INTERVAL = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# 已知的 ScienceDirect 期刊 URL 映射 (journal_id -> URL path 部分)
# 这些是从 DBLP 和 CCF PDF 已确认的
KNOWN_SCIENCEDIRECT = {
    'jpdc': 'journal-of-parallel-and-distributed-computing',
    'jsa': 'journal-of-systems-architecture',
    'parallelcomputing': 'parallel-computing',
    'performanceevaluationaninternationaljournal': 'performance-evaluation',
    'fgcs': 'future-generation-computer-systems',
    'jcs': 'journal-of-computer-security',
    'scp': 'science-of-computer-programming',
    'dmkd': 'data-mining-and-knowledge-discovery',
    'ejis': 'european-journal-of-information-systems',
    'ipm': 'information-processing-and-management',
    'is': 'information-systems',
    'aei': 'advanced-engineering-informatics',
    'jws': 'journal-of-web-semantics',
    'kais': 'knowledge-and-information-systems',
    'ijcis': 'international-journal-of-cooperative-information-systems',
    'ijswis': 'international-journal-on-semantic-web-and-information-systems',
    'jcis': 'journal-of-computer-information-systems',
    'jiis': 'journal-of-intelligent-information-systems',
    'jsis': 'the-journal-of-strategic-information-systems',
    'neuralnetworks': 'neural-networks',
    'pr': 'pattern-recognition',
    'eswa': 'expert-systems-with-applications',
    'ijprai': 'international-journal-of-pattern-recognition-and-artificial-intelligence',
    'ijufks': 'international-journal-of-uncertainty-fuzziness-and-knowledge-based-systems',
    'kbs': 'knowledge-based-systems',
    'prl': 'pattern-recognition-letters',
    'dss': 'decision-support-systems',
    'ivc': 'image-and-vision-computing',
    'ida': 'intelligent-data-analysis',
    'iwc': 'interacting-with-computers',
    'eaaai': 'engineering-applications-of-artificial-intelligence',
    'jsa': 'journal-of-systems-architecture',
    'cc': 'computer-communications',
    'jgc': 'journal-of-grid-computing',
    'jss': 'journal-of-systems-and-software',
    'jetta': 'journal-of-electronic-testing-theory-and-applications',
    'integration': 'integration-the-vlsi-journal',
    'jeta': 'journal-of-electronic-testing-theory-and-applications',
    'paa': 'pattern-analysis-and-applications',
    'nle': 'natural-language-engineering',
    'wi': 'web-intelligence',
    'neurocomputing': 'neurocomputing',
    'nca': 'neural-computing-and-applications',
    'caa': 'computers-and-electrical-engineering',
    'ms': 'multimedia-systems',
    'mta': 'multimedia-tools-and-applications',
    'sigpro': 'signal-processing',
    'spl': 'signal-processing-letters',
    'iet-ipr': 'iet-image-processing',
    'iet-signal-processing': 'iet-signal-processing',
    'iet-cvi': 'iet-computer-vision',
    'iet-cvi': 'iet-computer-vision',
}

# 已知的非 Elsevier 期刊（避免误匹配）
KNOWN_NOT_SCIENCEDIRECT = {
    'tocs', 'tos', 'tcad', 'tc', 'tpds', 'taco', 'taas', 'todaes', 'tecs', 'trets',
    'tvlsi', 'jetc', 'concurrency', 'dc', 'rts', 'tjsc', 'tcasi', 'ccf-thpc', 'tsusc',
    'jsac', 'tmc', 'ton', 'toit', 'tomm', 'tosn', 'cn', 'tcom', 'twc', 'adhoenetworks',
    'cc',  # Computer Communications - could be Elsevier, will handle separately
    'tdsc', 'tifs', 'journalofcryptology', 'tops', 'computersandsecurity',
    'toplas', 'tosem', 'tse', 'tsc', 'ase', 'ese', 'iets', 'ist', 'jfp',
    'journalswevol', 're', 'sosym', 'stvr', 'spe', 'cl', 'ijseke', 'sttt',
    'jlamp', 'jwe', 'soca', 'sqj', 'tplp', 'pacmpl',
    'tods', 'tois', 'tkde', 'vldbj', 'tkdd', 'tweb', 'dke', 'geoinformatica',
    'informationsciences', 'jasist',
    'tit', 'iandc', 'siamjcomp', 'talg', 'tocl', 'toms', 'algorithmica',
    'computationalcomplexity', 'formal Aspects', 'fmsd', 'injcomput', 'jcss',
    'jgo', 'jsc', 'mscs', 'tcs',
    'actam', 'actainformatica', 'apal', 'dam', 'fuin', 'ipl', 'jcomplexity',
    'logcom', 'jsl', 'lmcs', 'sidma', 'theoryofcomputingsystems', 'tqc',
    'tog', 'tip', 'tvcg', 'tmm', 'cagd', 'cgf', 'cad', 'tcsvt', 'siims',
    'speechcommunication', 'cvmj', 'cgta', 'cavw', 'computersandgraphics', 'dcg',
    'image', 'thevisualcomputer', 'visualinformatics', 'vrih', 'gmod',
    'ai', 'tpami', 'ijcv', 'jmlr', 'tap', 'aamas', 'computationalinguistics',
    'cviu', 'tac', 'taslp', 'ieeetc', 'tec', 'tfs', 'tnnls', 'ijar', 'jair',
    'jauto', 'jslhr', 'machinelearning', 'neuralcomputation', 'tacl', 'tallip',
    'appliedintelligence', 'aim', 'artificiallife', 'ci', 'computerspeech',
    'connection science', 'tg', 'ijcia', 'ijns', 'ijdcar', 'jetai', 'kbs',
    'machinetranslation', 'machinevision', 'naturalcomputing', 'npl',
    'softcomputing', 'tiis', 'telo', 'jats',
    'tochi', 'ijhcs', 'cscw', 'hci', 'huma', 'umuai', 'tsmc', 'ccft pci',
    'bit', 'puc', 'pmc', 'pacmhci', 'thri',
    'jacm', 'procieee', 'scis', 'bioinformatics', 'briefingsinbioinformatics',
    'cognition', 'tase', 'tgars', 'tits', 'tmi', 'tr', 'tcbb', 'jcst', 'jamia',
    'ploscompbio', 'thecomputerjournal', 'www', 'fcs', 'bcra', 'bmcbioinformatics',
    'cyberneticsand systems', 'jbi', 'medicalimageanalysis', 'tii', 'tcps',
    'tocE', 'eitee', 'tcss', 'ieeetr', 'health', 'acmdlt',
}


def build_sciencedirect_url(journal_id, journal_name):
    """构建 ScienceDirect URL"""
    # 先查已知映射
    if journal_id in KNOWN_SCIENCEDIRECT:
        path = KNOWN_SCIENCEDIRECT[journal_id]
        return f"https://www.sciencedirect.com/journal/{path}/about/aims-and-scope"

    # 跳过已知的非 Elsevier
    if journal_id in KNOWN_NOT_SCIENCEDIRECT:
        return None

    # 尝试从 journal_name 构建
    name = journal_name.lower()
    # 替换特殊字符为空格，然后转成 URL 友好的格式
    name = re.sub(r'[^a-z0-9\s]', ' ', name)
    name = re.sub(r'\s+', '-', name.strip())
    name = name[:60]  # 截断过长的名称

    # 常见的 elsevier 期刊名映射
    known_names = {
        'journal of parallel and distributed computing': 'journal-of-parallel-and-distributed-computing',
        'journal of systems architecture': 'journal-of-systems-architecture',
        'parallel computing': 'parallel-computing',
        'performance evaluation': 'performance-evaluation',
        'future generation computer systems': 'future-generation-computer-systems',
        'journal of computer security': 'journal-of-computer-security',
        'information and computation': 'information-and-computation',
        'information systems': 'information-systems',
        'information processing and management': 'information-processing-and-management',
        'data mining and knowledge discovery': 'data-mining-and-knowledge-discovery',
        'knowledge-based systems': 'knowledge-based-systems',
        'expert systems with applications': 'expert-systems-with-applications',
        'pattern recognition': 'pattern-recognition',
        'neural networks': 'neural-networks',
        'pattern recognition letters': 'pattern-recognition-letters',
        'decision support systems': 'decision-support-systems',
        'engineering applications of artificial intelligence': 'engineering-applications-of-artificial-intelligence',
        'web intelligence': 'web-intelligence',
        'neurocomputing': 'neurocomputing',
        'signal processing': 'signal-processing',
        'multimedia tools and applications': 'multimedia-tools-and-applications',
        'multimedia systems': 'multimedia-systems',
    }

    for kw, path in known_names.items():
        if kw in name:
            return f"https://www.sciencedirect.com/journal/{path}/about/aims-and-scope"

    return None


def fetch_scope(url):
    """从 ScienceDirect 页面抓取 scope"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')

            # 尝试多种方式获取 scope

            # 1. meta description
            meta = soup.find('meta', attrs={'name': 'description'})
            if meta and meta.get('content'):
                desc = meta['content'].strip()
                if len(desc) > 50 and 'Copyright' not in desc:
                    return desc

            # 2. og:description
            og = soup.find('meta', property='og:description')
            if og and og.get('content'):
                desc = og['content'].strip()
                if len(desc) > 50:
                    return desc

            # 3. 查找包含 "scope" 的 section
            for section in soup.find_all(['div', 'section']):
                text = section.get_text()
                if 'aim' in text.lower() and 'scope' in text.lower():
                    # 提取相邻的段落
                    for p in section.find_all('p'):
                        p_text = p.get_text(strip=True)
                        if len(p_text) > 100:
                            return p_text
    except Exception as e:
        pass
    return None


def process():
    # 读取所有期刊
    journals = []
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            journals.append(json.loads(line))

    print(f"Loaded {len(journals)} journals")

    updated = 0
    already_has_scope = 0
    failed = 0

    for i, journal in enumerate(journals, 1):
        journal_id = journal['journal_id']
        journal_name = journal['journal_name']

        # 已经有 scope 且不是空的，跳过
        if journal.get('scope_text') and journal['scope_text'] not in ['', '暂无scope']:
            already_has_scope += 1
            continue

        url = build_sciencedirect_url(journal_id, journal_name)
        if not url:
            continue

        print(f"[{i}/{len(journals)}] {journal_id} - {journal_name[:40]}")
        print(f"  URL: {url}")

        scope = fetch_scope(url)
        if scope:
            # 清理 scope
            scope = scope.strip()
            if len(scope) > 800:
                scope = scope[:800] + "..."

            journal['scope_text'] = scope
            updated += 1
            print(f"  SUCCESS: {scope[:80]}...")
        else:
            failed += 1
            print(f"  FAILED: Could not fetch scope")

        time.sleep(SLEEP_INTERVAL)

    # 写回文件
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for journal in journals:
            f.write(json.dumps(journal, ensure_ascii=False) + '\n')

    print(f"\n{'='*60}")
    print(f"完成!")
    print(f"更新了 {updated} 个期刊的 scope")
    print(f"已有 scope: {already_has_scope}")
    print(f"失败: {failed}")
    print(f"输出: {OUTPUT_PATH}")


if __name__ == "__main__":
    process()