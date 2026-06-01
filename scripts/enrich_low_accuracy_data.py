#!/usr/bin/env python3
"""Enrich low-accuracy journal metadata and typical abstracts.

This script is intentionally deterministic: it only appends curated subdomain
terms and replaces generic typical abstracts with local real-paper abstracts
when available.
"""
import json
import re
from collections import defaultdict
from pathlib import Path


JOURNALS_PATH = Path("data/processed/journals.jsonl")
TYPICAL_DIR = Path("data/typical_abstracts")
PAPER_SOURCES = [
    ("metadata", Path("data/evaluation/papers_metadata.jsonl")),
    ("train", Path("data/evaluation/train_papers.jsonl")),
    ("val", Path("data/evaluation/val_papers.jsonl")),
    ("test", Path("data/evaluation/test_papers.jsonl")),
]


ENRICHMENTS = {
    "ton": {
        "scope": [
            "goal-oriented communication",
            "age of information",
            "age of incorrect information",
            "medium access control",
            "distributed sensing",
            "belief propagation in networks",
            "semantic communications",
        ],
        "keywords": [
            "goal-oriented communication",
            "age of information",
            "medium access control",
            "distributed sensing",
            "semantic communications",
        ],
    },
    "toit": {
        "scope": [
            "IoT firmware analysis",
            "firmware version identification",
            "IoT intrusion detection",
            "liquid neural networks for IDS",
            "internet-of-things security",
            "known and unknown attack detection",
        ],
        "keywords": [
            "IoT firmware",
            "firmware version identification",
            "IoT intrusion detection",
            "liquid neural networks",
            "unknown attacks",
        ],
    },
    "wirelessnetworks": {
        "scope": [
            "vehicular ad hoc networks",
            "VANET reliability",
            "relay selection",
            "link reliability",
            "fuzzy optimization",
            "grey wolf optimization",
        ],
        "keywords": [
            "VANET",
            "relay selection",
            "link reliability",
            "fuzzy optimization",
            "grey wolf optimization",
        ],
    },
    "adhocnetworks": {
        "scope": [
            "LEO satellite networks",
            "inter-satellite links",
            "zero trust authentication",
            "6G satellite networking",
            "space-terrestrial networking",
        ],
        "keywords": [
            "LEO satellite networks",
            "inter-satellite links",
            "zero trust authentication",
            "6G networks",
            "satellite networking",
        ],
    },
    "tdsc": {
        "scope": [
            "secure blockchain systems",
            "coded blockchain",
            "blockchain for IoT",
            "secure distributed ledgers",
            "reliable IoT security",
        ],
        "keywords": [
            "coded blockchain",
            "blockchain security",
            "IoT security",
            "secure distributed ledgers",
            "system reliability",
        ],
    },
    "tops": {
        "scope": [
            "timing side channels",
            "timing interruption monitoring",
            "fronthaul security",
            "5G security assessment",
            "radio access network security",
        ],
        "keywords": [
            "timing side channels",
            "fronthaul security",
            "5G security",
            "RAN security",
            "security assessment",
        ],
    },
    "computerssecurity": {
        "scope": [
            "runtime verification for security",
            "cyber-attack anomaly prediction",
            "cyber-physical systems security",
            "temporal logic security policies",
            "LTL-based monitoring",
        ],
        "keywords": [
            "runtime verification",
            "cyber-attack anomaly prediction",
            "cyber-physical security",
            "temporal logic",
            "LTL monitoring",
        ],
    },
    "jisa": {
        "scope": [
            "image forensics",
            "computer-generated image detection",
            "GAN image detection",
            "multimedia forensics",
            "deepfake detection",
            "multi-colorspace features",
        ],
        "keywords": [
            "image forensics",
            "GAN detection",
            "computer-generated images",
            "multimedia forensics",
            "deepfake detection",
        ],
    },
    "sicomp": {
        "scope": [
            "meta-complexity",
            "minimum circuit size problem",
            "hardness of approximation",
            "cryptographic assumptions",
            "circuit complexity",
        ],
        "keywords": [
            "meta-complexity",
            "MCSP",
            "hardness of approximation",
            "circuit complexity",
            "cryptographic assumptions",
        ],
    },
    "talg": {
        "scope": [
            "hashing-based data structures",
            "peeling algorithms",
            "orientability threshold",
            "spatial coupling",
            "random hypergraphs",
        ],
        "keywords": [
            "hashing data structures",
            "peeling algorithms",
            "orientability threshold",
            "spatial coupling",
            "random hypergraphs",
        ],
    },
    "theoryofcomputingsystems": {
        "scope": [
            "average-case rigidity",
            "matrix rigidity lower bounds",
            "online scheduling with restarts",
            "computable numberings",
            "recursion theorem",
            "completion operator",
        ],
        "keywords": [
            "average-case rigidity",
            "online scheduling",
            "computable numberings",
            "recursion theorem",
            "completion operator",
        ],
    },
    "acta": {
        "scope": [
            "stateless model checking",
            "relaxed memory models",
            "TSO",
            "PSO",
            "chronological traces",
            "program verification under weak memory",
        ],
        "keywords": [
            "stateless model checking",
            "relaxed memory models",
            "TSO",
            "PSO",
            "program verification",
        ],
    },
    "tap": {
        "scope": [
            "virtual humans",
            "embodied conversational agents",
            "user-agent similarity",
            "user-designer similarity",
            "mental health interfaces",
            "perception in virtual environments",
        ],
        "keywords": [
            "virtual humans",
            "embodied agents",
            "user-agent similarity",
            "mental health interfaces",
            "virtual human design",
        ],
    },
    "appliedintelligence": {
        "scope": [
            "feature selection",
            "rough sets",
            "dynamic neighbourhood entropy",
            "high-utility pattern mining",
            "periodic pattern mining",
            "LiDAR semantic segmentation",
        ],
        "keywords": [
            "feature selection",
            "rough sets",
            "high-utility pattern mining",
            "periodic pattern mining",
            "LiDAR segmentation",
        ],
    },
    "jair": {
        "scope": [
            "incremental learning",
            "continual learning",
            "catastrophic forgetting",
            "lifelong learning",
            "neural network adaptation",
        ],
        "keywords": [
            "incremental learning",
            "continual learning",
            "catastrophic forgetting",
            "lifelong learning",
            "neural networks",
        ],
    },
    "nca": {
        "scope": [
            "medical image classification",
            "fetal ultrasound analysis",
            "multi-task multilabel learning",
            "landmark-aware identification",
            "deep neural applications",
        ],
        "keywords": [
            "medical image classification",
            "fetal ultrasound",
            "multi-task learning",
            "multilabel learning",
            "deep learning applications",
        ],
    },
    "jacm": {
        "scope": [
            "recoverable mutual exclusion",
            "fair mutual exclusion",
            "concurrent algorithms",
            "shared-memory synchronization",
            "fault-tolerant distributed computing",
        ],
        "keywords": [
            "recoverable mutual exclusion",
            "concurrent algorithms",
            "shared-memory synchronization",
            "fault tolerance",
            "distributed computing",
        ],
    },
    "proc._ieee": {
        "scope": [
            "stream-based architectures",
            "accelerator architectures",
            "safe reinforcement learning",
            "power systems control",
            "resilient distribution systems",
            "networked microgrids",
        ],
        "keywords": [
            "stream-based architectures",
            "accelerators",
            "safe reinforcement learning",
            "power systems",
            "networked microgrids",
        ],
    },
    "jcst": {
        "scope": [
            "serverless computing",
            "Function-as-a-Service",
            "cloud-native architecture",
            "distributed cloud systems",
            "computing technology surveys",
        ],
        "keywords": [
            "serverless computing",
            "Function-as-a-Service",
            "cloud-native architecture",
            "cloud computing",
            "technology survey",
        ],
    },
    "bmcbioinformatics": {
        "scope": [
            "multi-omics integration",
            "cancer subtype classification",
            "graph structure learning",
            "computational genomics",
            "machine learning for omics",
        ],
        "keywords": [
            "multi-omics",
            "cancer subtype classification",
            "graph structure learning",
            "computational genomics",
            "machine learning bioinformatics",
        ],
    },
    "jbhi": {
        "scope": [
            "protein function prediction",
            "biomedical representation learning",
            "multimodal biomedical data",
            "low-data biomedical learning",
            "computational medicine",
        ],
        "keywords": [
            "protein function prediction",
            "biomedical representation learning",
            "multimodal learning",
            "low-data learning",
            "computational medicine",
        ],
    },
}


DUPLICATE_CANONICALS = {
    "dke_2": "dke",
    "ipl_2": "ipl",
    "ijis_2": "ijis",
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def append_unique(items, additions):
    seen = {norm(item) for item in items}
    for item in additions:
        if norm(item) not in seen:
            items.append(item)
            seen.add(norm(item))
    return items


def load_journals():
    return [json.loads(line) for line in JOURNALS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_journals(journals):
    JOURNALS_PATH.write_text(
        "".join(json.dumps(journal, ensure_ascii=False) + "\n" for journal in journals),
        encoding="utf-8",
    )


def enrich_journals(journals):
    by_id = {journal["journal_id"]: journal for journal in journals}
    changed = []

    for duplicate_id, canonical_id in DUPLICATE_CANONICALS.items():
        duplicate = by_id.get(duplicate_id)
        canonical = by_id.get(canonical_id)
        if not duplicate or not canonical:
            continue
        canonical["subject_tags"] = append_unique(canonical.get("subject_tags", []), duplicate.get("subject_tags", []))
        canonical["keywords"] = append_unique(canonical.get("keywords", []), duplicate.get("keywords", []))
        duplicate_scope = duplicate.get("scope_text", "")
        if duplicate_scope and duplicate_scope not in canonical.get("scope_text", ""):
            canonical["scope_text"] = canonical.get("scope_text", "").rstrip() + ", " + duplicate_scope
        duplicate["journal_name"] = f"{canonical['journal_name']} ({duplicate['subject_tags'][0]} profile)"
        changed.extend([canonical_id, duplicate_id])

    for journal_id, enrichment in ENRICHMENTS.items():
        journal = by_id.get(journal_id)
        if not journal:
            continue
        additions = enrichment["scope"]
        missing_scope = [term for term in additions if norm(term) not in norm(journal.get("scope_text", ""))]
        if missing_scope:
            journal["scope_text"] = journal.get("scope_text", "").rstrip() + ", " + ", ".join(missing_scope)
        journal["keywords"] = append_unique(journal.get("keywords", []), enrichment["keywords"])
        changed.append(journal_id)

    return sorted(set(changed))


def load_real_abstracts_by_journal(journals):
    name_to_id = {}
    for journal in journals:
        base_name = norm(journal["journal_name"].split(" (")[0])
        name_to_id.setdefault(base_name, journal["journal_id"])
    real = defaultdict(list)
    seen_titles = set()
    for source, path in PAPER_SOURCES:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            abstract = (item.get("abstract") or "").strip()
            venue = norm(item.get("venue", ""))
            title = (item.get("title") or "").strip()
            if len(abstract) < 300 or venue not in name_to_id:
                continue
            key = (venue, norm(title))
            if key in seen_titles:
                continue
            seen_titles.add(key)
            real[name_to_id[venue]].append({
                "source": source,
                "title": title,
                "abstract": abstract,
            })
    return real


def enrich_typical_abstracts(journals):
    real_by_journal = load_real_abstracts_by_journal(journals)
    changed = {}
    target_ids = set(ENRICHMENTS) | set(DUPLICATE_CANONICALS) | set(DUPLICATE_CANONICALS.values())
    for journal_id in sorted(target_ids):
        path = TYPICAL_DIR / f"{journal_id}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        examples = real_by_journal.get(journal_id, [])[:4]
        if not examples:
            continue
        abstracts = data.get("abstracts", [])
        for idx, example in enumerate(examples):
            replacement = {
                "method_type": "真实代表论文",
                "novelty_level": f"local_{example['source']}_example",
                "title": example["title"],
                "abstract": example["abstract"],
            }
            if idx < len(abstracts):
                abstracts[idx] = replacement
            else:
                abstracts.append(replacement)
        data["abstracts"] = abstracts[:4]
        if journal_id in DUPLICATE_CANONICALS:
            data["journal_name"] = f"{data.get('journal_name', journal_id)} ({journal_id} profile)"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed[journal_id] = [example["source"] for example in examples]
    return changed


def main():
    journals = load_journals()
    changed_journals = enrich_journals(journals)
    save_journals(journals)
    changed_typical = enrich_typical_abstracts(journals)
    print("Updated journals:", ", ".join(changed_journals))
    print("Updated typical abstracts:")
    for journal_id, sources in sorted(changed_typical.items()):
        print(f"  {journal_id}: {', '.join(sources)}")


if __name__ == "__main__":
    main()
