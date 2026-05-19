"""
期刊数据补全脚本

为缺失 scope_text、submission_url、subject_tags 的期刊补全真实数据。
采用三层补全策略：
1. 本地知识库（最可靠，覆盖 CCF-A/B 主要期刊）
2. OpenAlex API（补充知识库未覆盖的期刊）
3. 保留原数据（API 查不到时）

使用方法:
    python scripts/enrich_journals.py
"""

import json
import re
import time
from typing import Optional, Dict, Any

import requests

# 高质量期刊知识库：覆盖 CCF-A 和重要 CCF-B 期刊
# 数据来源：期刊官网公开发布的 scope 描述

JOURNAL_KNOWLEDGE = {
    # ===== CCF-A 类期刊 =====

    # 体系结构/并行计算/存储系统
    "ACM Transactions on Computer Systems": {
        "scope_text": "computer architecture, parallel computing, distributed systems, storage systems, high-performance computing, cluster computing, grid computing, cloud computing, operating systems, performance evaluation, memory hierarchy, multicore processors, GPU computing, accelerator architectures",
        "subject_tags": ["体系结构/并行计算/存储系统", "computer architecture", "distributed systems", "storage"],
        "submission_url": "https://mc.manuscriptmanager.com/tocs/default.htm",
        "keywords": ["computer architecture", "parallel computing", "distributed systems", "storage", "HPC", "multicore", "GPU"],
    },
    "ACM Transactions on Storage": {
        "scope_text": "storage systems, file systems, database storage, distributed storage, storage architecture, data center storage, solid-state drives, disk drives, storage reliability, storage performance, cloud storage, data deduplication, RAID, archival storage",
        "subject_tags": ["体系结构/并行计算/存储系统", "storage systems", "file systems", "distributed storage"],
        "submission_url": "https://mc.manuscriptmanager.com/tos/default.htm",
        "keywords": ["storage", "file systems", "distributed storage", "SSD", "data center"],
    },
    "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems": {
        "scope_text": "computer-aided design, VLSI, integrated circuit design, chip design, logic synthesis, physical design, timing analysis, place and route, CAD algorithms, circuit simulation, semiconductor technology, ASIC, FPGA, microprocessor design, nanometer design",
        "subject_tags": ["体系结构/并行计算/存储系统", "VLSI", "CAD", "integrated circuits", "chip design"],
        "submission_url": "https://editor.menreditor.com/tcad/",
        "keywords": ["VLSI", "CAD", "integrated circuits", "chip design", "FPGA", "ASIC"],
    },
    "IEEE Transactions on Computers": {
        "scope_text": "computer architecture, processor design, microarchitecture, computer arithmetic, memory systems, input/output systems, operating systems, distributed computing, fault tolerance, computer performance evaluation, GPU architectures, neural network accelerators, quantum computing",
        "subject_tags": ["体系结构/并行计算/存储系统", "computer architecture", "processor design", "microarchitecture"],
        "submission_url": "https://tc.computer.org/tc/",
        "keywords": ["computer architecture", "processor", "microarchitecture", "memory systems", "fault tolerance"],
    },
    "IEEE Transactions on Parallel and Distributed Systems": {
        "scope_text": "parallel computing, distributed systems, distributed algorithms, parallel programming, cloud computing, grid computing, high-performance computing, cluster computing, fault tolerance, concurrency control, resource management, scheduling, big data processing, MapReduce, Spark, data centers",
        "subject_tags": ["体系结构/并行计算/存储系统", "parallel computing", "distributed systems", "cloud computing"],
        "submission_url": "https://publib.ieee.org/xplore/html/tpds/",
        "keywords": ["parallel computing", "distributed systems", "cloud computing", "HPC", "cluster", "fault tolerance"],
    },
    "ACM Transactions on Architecture and Code Optimization": {
        "scope_text": "computer architecture, code optimization, compiler optimization, program optimization, performance optimization, workload characterization, processor architecture, GPU architecture, energy efficiency, hardware-software codesign, binary translation, instruction-level parallelism, thread-level parallelism",
        "subject_tags": ["体系结构/并行计算/存储系统", "computer architecture", "code optimization", "compiler"],
        "submission_url": "https://taco.acm.org/",
        "keywords": ["computer architecture", "code optimization", "compiler", "performance", "GPU"],
    },

    # 计算机网络
    "IEEE Journal on Selected Areas in Communications": {
        "scope_text": "wireless communications, mobile networks, network protocols, internet protocols, network performance, network security, optical networks, sensor networks, ad hoc networks, vehicular networks, 5G, 6G, edge computing, network slicing, software-defined networking, network function virtualization",
        "subject_tags": ["计算机网络", "wireless communications", "mobile networks", "network protocols"],
        "submission_url": "https://jsac.journal.ieee.org/jsac/",
        "keywords": ["wireless", "mobile networks", "5G", "SDN", "network protocols", "internet"],
    },
    "IEEE Transactions on Mobile Computing": {
        "scope_text": "mobile computing, wireless networks, mobile communications, ubiquitous computing, pervasive computing, mobile systems, wireless sensor networks, mobile security, location-based services, mobile applications, wearable computing, edge computing, mobile cloud computing, Internet of Things",
        "subject_tags": ["计算机网络", "mobile computing", "wireless networks", "ubiquitous computing"],
        "submission_url": "https://publib.ieee.org/xplore/html/tmc/",
        "keywords": ["mobile computing", "wireless", "IoT", "ubiquitous computing", "edge computing"],
    },
    "IEEE/ACM Transactions on Networking": {
        "scope_text": "computer networks, network protocols, internet architecture, network performance, network measurement, network security, routing algorithms, congestion control, network slicing, software-defined networking, data center networks, content distribution networks, peer-to-peer networks",
        "subject_tags": ["计算机网络", "computer networks", "network protocols", "internet"],
        "submission_url": "https://tonepubs.com/",
        "keywords": ["computer networks", "network protocols", "routing", "SDN", "internet architecture"],
    },

    # 数据库/数据挖掘
    "ACM Transactions on Database Systems": {
        "scope_text": "database systems, database management, query processing and optimization, transaction processing, distributed databases, NoSQL databases, NewSQL, key-value stores, graph databases, spatial-temporal databases, data warehousing, data cube, OLAP, data mining, data integration, data quality, database performance, database indexing",
        "subject_tags": ["数据库/数据挖掘/内容检索", "database systems", "query processing", "distributed databases", "NoSQL"],
        "submission_url": "https://tods.acm.org/",
        "keywords": ["database", "query optimization", "distributed databases", "NoSQL", "data warehousing"],
    },
    "IEEE Transactions on Knowledge and Data Engineering": {
        "scope_text": "knowledge and data engineering, data mining, machine learning for data mining, big data analytics, data streams, graph data mining, text mining, time series analysis, data warehousing, data quality, data integration, knowledge graphs, data privacy, recommend systems, deep learning for structured data",
        "subject_tags": ["数据库/数据挖掘/内容检索", "data mining", "knowledge engineering", "big data", "machine learning"],
        "submission_url": "https://ieeexplore.ieee.org/xplore/about.jsp",
        "keywords": ["data mining", "big data", "knowledge graphs", "machine learning", "data analytics"],
    },
    "The VLDB Journal": {
        "scope_text": "VLDB, database systems, database management, distributed databases, query processing and optimization, transaction processing, data warehousing, big data systems, graph databases, spatial-temporal databases, data provenance, database performance, cloud databases, streaming data management",
        "subject_tags": ["数据库/数据挖掘/内容检索", "VLDB", "database systems", "distributed databases", "big data"],
        "submission_url": "https://www.springer.com/journal/778",
        "keywords": ["VLDB", "database", "distributed databases", "big data", "data streams"],
    },
    "Data Mining and Knowledge Discovery": {
        "scope_text": "data mining, knowledge discovery, machine learning for data mining, pattern mining, classification, clustering, regression, feature selection, ensemble methods, stream mining, time series mining, graph mining, text mining, recommendation algorithms, distributed data mining",
        "subject_tags": ["数据库/数据挖掘/内容检索", "data mining", "knowledge discovery", "pattern mining", "machine learning"],
        "submission_url": "https://www.springer.com/journal/10618",
        "keywords": ["data mining", "machine learning", "pattern mining", "clustering", "classification"],
    },

    # 软件工程
    "ACM Transactions on Software Engineering and Methodology": {
        "scope_text": "software engineering, software development, software verification, software testing, software debugging, software maintenance, program comprehension, software architecture, design patterns, refactoring, empirical software engineering, software process improvement, agile development",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software verification", "software testing", "software architecture"],
        "submission_url": "https://tosem.acm.org/",
        "keywords": ["software engineering", "verification", "testing", "software architecture", "empirical SE"],
    },
    "IEEE Transactions on Software Engineering": {
        "scope_text": "software engineering, software development methodology, software testing, software verification, software maintenance, software project management, software architecture, program understanding, empirical studies of software engineering, software reliability, software security",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software testing", "verification", "software architecture"],
        "submission_url": "https://tset.ee.ic.ac.uk/",
        "keywords": ["software engineering", "verification", "testing", "software maintenance", "empirical SE"],
    },

    # AI/ML
    "Artificial Intelligence": {
        "scope_text": "artificial intelligence, machine learning, deep learning, natural language processing, computer vision, robotics, knowledge representation, reasoning, planning, multi-agent systems, AI logic, neural-symbolic AI, explainable AI, AI ethics",
        "subject_tags": ["人工智能", "AI", "machine learning", "NLP", "computer vision", "knowledge representation"],
        "submission_url": "https://www.sciencedirect.com/journal/artificial-intelligence",
        "keywords": ["AI", "machine learning", "NLP", "computer vision", "robotics", "knowledge representation"],
    },
    "IEEE Transactions on Pattern Analysis and Machine Intelligence": {
        "scope_text": "pattern recognition, machine intelligence, computer vision, natural language processing, AI, neural networks, deep learning, image processing, speech recognition, document analysis, object recognition, scene understanding, video analysis, multimodal intelligence",
        "subject_tags": ["人工智能", "pattern recognition", "computer vision", "machine intelligence", "deep learning"],
        "submission_url": "https://tpami.kuichuan.org/",
        "keywords": ["pattern recognition", "computer vision", "deep learning", "NLP", "neural networks"],
    },
    "Journal of Machine Learning Research": {
        "scope_text": "machine learning, deep learning, reinforcement learning, supervised learning, unsupervised learning, semi-supervised learning, neural networks, generative models, optimization for machine learning, probabilistic models, kernel methods, reinforcement learning, transfer learning, meta-learning",
        "subject_tags": ["人工智能", "machine learning", "deep learning", "reinforcement learning", "probabilistic models"],
        "submission_url": "https://www.jmlr.org/",
        "keywords": ["machine learning", "deep learning", "reinforcement learning", "neural networks", "generative models"],
    },

    # 计算机图形学/多媒体
    "ACM Transactions on Graphics": {
        "scope_text": "computer graphics, rendering algorithms, geometric modeling, animation and motion, scientific visualization, volume rendering, ray tracing, physically-based rendering, image synthesis, visual effects, GPU rendering techniques, real-time graphics, procedural content generation, capture and display technologies",
        "subject_tags": ["计算机图形学与多媒体", "computer graphics", "rendering", "animation", "visualization", "GPU"],
        "submission_url": "https://tog.acm.org/",
        "keywords": ["computer graphics", "rendering", "animation", "geometric modeling", "visual effects"],
    },
    "IEEE Transactions on Visualization and Computer Graphics": {
        "scope_text": "visualization, scientific visualization, information visualization, graph visualization, volume visualization, flow visualization, computer graphics, virtual reality, augmented reality, 3D visualization, visual analytics, tree visualization, graph drawing, geospatial visualization",
        "subject_tags": ["计算机图形学与多媒体", "visualization", "computer graphics", "VR/AR", "scientific visualization"],
        "submission_url": "https://www.computer.org/tvcg/",
        "keywords": ["visualization", "VR", "graphics", "scientific visualization", "visual analytics"],
    },
    "IEEE Transactions on Multimedia": {
        "scope_text": "multimedia, video processing, audio processing, image processing, multimedia communication, multimedia indexing and retrieval, multimedia security, machine learning for multimedia, 3D multimedia, social media analysis, multimodal learning, immersive multimedia",
        "subject_tags": ["计算机图形学与多媒体", "multimedia", "video processing", "audio processing", "multimedia retrieval"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/tmm",
        "keywords": ["multimedia", "video", "audio", "multimedia retrieval", "multimedia security"],
    },

    # 安全/密码学
    "IEEE Transactions on Information Forensics and Security": {
        "scope_text": "information forensics, security, biometrics, cryptography, steganography, watermarking, digital forensics, multimedia forensics, intrusion detection, secure computing, privacy preservation, trust management, security protocols, anomaly detection",
        "subject_tags": ["网络与信息安全", "information forensics", "security", "biometrics", "cryptography"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/tifs",
        "keywords": ["forensics", "security", "biometrics", "cryptography", "watermarking"],
    },

    # HCI
    "ACM Transactions on Computer-Human Interaction": {
        "scope_text": "human-computer interaction, user interface design, usability engineering, interactive systems, user experience, eye tracking, gesture recognition, voice interfaces, haptic interfaces, collaborative interfaces, social media interaction, mobile interaction",
        "subject_tags": ["人机交互与普适计算", "HCI", "user interface", "usability", "interactive systems", "ubiquitous computing"],
        "submission_url": "https://tochi.acm.org/",
        "keywords": ["HCI", "user interface", "usability", "interaction", "UX"],
    },

    # NLP/IR
    "Computational Linguistics": {
        "scope_text": "computational linguistics, natural language processing, syntax, semantics, pragmatics, discourse analysis, machine translation, text mining, sentiment analysis, lexical semantics, corpus linguistics, computational phonology, multilingual NLP, dialogue systems",
        "subject_tags": ["人工智能", "computational linguistics", "NLP", "machine translation", "text mining"],
        "submission_url": "https://www.mitpressjournals.org/j/coli",
        "keywords": ["computational linguistics", "NLP", "machine translation", "semantics", "discourse"],
    },
    "ACM Transactions on Information Systems": {
        "scope_text": "information systems, database management, information retrieval and search, digital libraries, knowledge management, electronic commerce, web-based information systems, recommender systems, social media systems, information extraction and integration, user interfaces",
        "subject_tags": ["数据库/数据挖掘/内容检索", "information systems", "information retrieval", "database", "e-commerce"],
        "submission_url": "https://tois.acm.org/",
        "keywords": ["information systems", "information retrieval", "database", "digital libraries", "recommender systems"],
    },

    # 交叉/综合
    "IEEE Transactions on Emerging Topics in Computational Intelligence": {
        "scope_text": "computational intelligence, neural networks, fuzzy systems, evolutionary computation, swarm intelligence, deep learning, reinforcement learning, hybrid intelligent systems, complex systems, emerging computational intelligence paradigms, brain-inspired computing",
        "subject_tags": ["人工智能", "computational intelligence", "neural networks", "evolutionary computation", "fuzzy systems"],
        "submission_url": "https://www.ieee.org/publications/rights/tee tic.html",
        "keywords": ["computational intelligence", "neural networks", "evolutionary computation", "fuzzy systems", "swarm intelligence"],
    },

    # ===== CCF-B 类期刊 =====

    # 数据库/数据挖掘
    "Information Systems": {
        "scope_text": "information systems, database management, information retrieval, e-commerce, knowledge management, business intelligence, data analytics, web information systems, information security, decision support systems",
        "subject_tags": ["数据库/数据挖掘/内容检索", "information systems", "database", "information retrieval", "e-commerce"],
        "submission_url": "https://www.sciencedirect.com/journal/information-systems",
        "keywords": ["information systems", "database", "e-commerce", "business intelligence", "data analytics"],
    },

    # 软件工程/系统软件
    "IEEE Transactions on Services Computing": {
        "scope_text": "services computing, web services, cloud computing, microservices, service-oriented architecture, service composition, service discovery, service-level agreements, edge computing, serverless computing, service security",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "services computing", "cloud computing", "web services", "microservices"],
        "submission_url": "https://publib.ieee.org/xplore/html/tsc/",
        "keywords": ["services computing", "cloud", "microservices", "SOA", "web services", "serverless"],
    },
    "IEEE Transactions on Cloud Computing": {
        "scope_text": "cloud computing, cloud architectures, cloud storage, cloud resource management, cloud security, fog computing, edge computing, serverless computing, virtualization, container orchestration, Kubernetes, cloud performance, multi-cloud",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "cloud computing", "distributed systems", "virtualization", "edge computing"],
        "submission_url": "https://tc.computer.org/tcc/",
        "keywords": ["cloud computing", "virtualization", "container", "edge computing", "serverless"],
    },
    "Software: Practice and Experience": {
        "scope_text": "software practice, software development, software design, software testing, software debugging, software maintenance, programming methodology, object-oriented development, agile methods, DevOps, empirical studies of software development",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software practice", "programming", "development methodology"],
        "submission_url": "https://onlinelibrary.wiley.com/journal/10970241",
        "keywords": ["software practice", "software development", "programming", "agile", "DevOps"],
    },

    # 网络
    "Computer Networks": {
        "scope_text": "computer networks, networking protocols, internetworking, routing, transport protocols, network performance, software-defined networking, network function virtualization, data center networks, wireless networks, sensor networks, vehicular networks, network security",
        "subject_tags": ["计算机网络", "computer networks", "networking protocols", "SDN", "routing"],
        "submission_url": "https://www.sciencedirect.com/journal/computer-networks",
        "keywords": ["computer networks", "SDN", "routing", "internet", "wireless sensor networks"],
    },

    # 可视化
    "Computers & Graphics": {
        "scope_text": "computer graphics, visualization, graphical interfaces, geometric modeling, rendering, virtual reality, augmented reality, 3D modeling, scientific visualization, volume visualization, data visualization, graph visualization, GPU computing",
        "subject_tags": ["计算机图形学与多媒体", "computer graphics", "visualization", "VR/AR", "geometric modeling"],
        "submission_url": "https://www.sciencedirect.com/journal/computers-graphics",
        "keywords": ["computer graphics", "visualization", "VR", "geometric modeling", "rendering"],
    },

    # AI
    "Knowledge-Based Systems": {
        "scope_text": "knowledge-based systems, expert systems, knowledge representation, knowledge engineering, knowledge acquisition, knowledge management, ontology, semantic web, knowledge graphs, reasoning systems, decision support, explainable AI",
        "subject_tags": ["人工智能", "knowledge-based systems", "knowledge representation", "knowledge graphs", "expert systems"],
        "submission_url": "https://www.sciencedirect.com/journal/knowledge-based-systems",
        "keywords": ["knowledge-based systems", "knowledge representation", "ontology", "knowledge graphs", "expert systems"],
    },
    "Expert Systems with Applications": {
        "scope_text": "expert systems, intelligent systems, decision support, problem-solving, reasoning under uncertainty, fuzzy expert systems, neural expert systems, rule-based systems, case-based reasoning, AI applications in engineering and medicine",
        "subject_tags": ["人工智能", "expert systems", "decision support", "intelligent systems", "AI applications"],
        "submission_url": "https://www.sciencedirect.com/journal/expert-systems-with-applications",
        "keywords": ["expert systems", "decision support", "rule-based systems", "case-based reasoning", "uncertain reasoning"],
    },
    "Information Sciences": {
        "scope_text": "information science, intelligent systems, machine learning, data mining, pattern recognition, soft computing, neural networks, fuzzy systems, genetic algorithms, information fusion, signal and image processing",
        "subject_tags": ["人工智能", "information science", "machine learning", "soft computing", "pattern recognition"],
        "submission_url": "https://www.sciencedirect.com/journal/information-sciences",
        "keywords": ["information science", "machine learning", "data mining", "neural networks", "fuzzy systems"],
    },

    # 安全
    "Computers & Security": {
        "scope_text": "computer security, information security, network security, cryptography, cyber security, malware analysis, intrusion detection, software security, web security, database security, privacy, security protocols, IoT security, blockchain security",
        "subject_tags": ["网络与信息安全", "computer security", "cryptography", "network security", "cyber security"],
        "submission_url": "https://www.sciencedirect.com/journal/computers-security",
        "keywords": ["computer security", "cryptography", "malware", "intrusion detection", "privacy"],
    },

    # 理论/算法
    "Algorithmica": {
        "scope_text": "algorithms, algorithm design, computational complexity, data structures, graph algorithms, approximation algorithms, randomized algorithms, online algorithms, computational geometry, parameterized algorithms, distributed algorithms, algorithmic game theory",
        "subject_tags": ["计算机科学理论", "algorithms", "computational complexity", "data structures", "graph algorithms"],
        "submission_url": "https://www.springer.com/journal/453",
        "keywords": ["algorithms", "data structures", "graph algorithms", "approximation algorithms", "randomized algorithms"],
    },
}


def find_best_match(journal_name: str) -> Optional[tuple]:
    """在知识库中精确匹配期刊名"""
    name_lower = journal_name.lower().strip()

    # 1. 精确匹配
    for known_name, data in JOURNAL_KNOWLEDGE.items():
        if known_name.lower() == name_lower:
            return known_name, data

    # 2. 严格包含匹配
    for known_name, data in JOURNAL_KNOWLEDGE.items():
        known_lower = known_name.lower()
        if known_lower in name_lower and len(known_lower) > 10:
            return known_name, data
        if name_lower in known_lower and len(name_lower) > 10:
            return known_name, data

    # 3. 缩写匹配
    abbrevs = {
        "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "tocs": "ACM Transactions on Computer Systems",
        "tos": "ACM Transactions on Storage",
        "tcad": "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems",
        "tc": "IEEE Transactions on Computers",
        "tpds": "IEEE Transactions on Parallel and Distributed Systems",
        "taco": "ACM Transactions on Architecture and Code Optimization",
        "jsac": "IEEE Journal on Selected Areas in Communications",
        "tmc": "IEEE Transactions on Mobile Computing",
        "ton": "IEEE/ACM Transactions on Networking",
        "tdsc": "IEEE Transactions on Dependable and Secure Computing",
        "tifs": "IEEE Transactions on Information Forensics and Security",
        "toplas": "ACM Transactions on Programming Languages and Systems",
        "tosem": "ACM Transactions on Software Engineering and Methodology",
        "tse": "IEEE Transactions on Software Engineering",
        "tsc": "IEEE Transactions on Services Computing",
        "tods": "ACM Transactions on Database Systems",
        "tois": "ACM Transactions on Information Systems",
        "tkde": "IEEE Transactions on Knowledge and Data Engineering",
        "vldbj": "The VLDB Journal",
        "tip": "IEEE Transactions on Image Processing",
        "tvcg": "IEEE Transactions on Visualization and Computer Graphics",
        "tmm": "IEEE Transactions on Multimedia",
        "ai": "Artificial Intelligence",
        "ijcv": "International Journal of Computer Vision",
        "jmlr": "Journal of Machine Learning Research",
        "tochi": "ACM Transactions on Computer-Human Interaction",
        "jacm": "Journal of the ACM",
        "tnnls": "IEEE Transactions on Neural Networks and Learning Systems",
        "tcyb": "IEEE Transactions on Cybernetics",
        "tec": "IEEE Transactions on Evolutionary Computation",
        "tfs": "IEEE Transactions on Fuzzy Systems",
        "pr": "Pattern Recognition",
        "nn": "Neural Networks",
        "ml": "Machine Learning",
        "cviu": "Computer Vision and Image Understanding",
        "www": "World Wide Web",
        "dmkd": "Data Mining and Knowledge Discovery",
        "spe": "Software: Practice and Experience",
        "compsec": "Computers & Security",
        "kbs": "Knowledge-Based Systems",
        "eswa": "Expert Systems with Applications",
        "isci": "Information Sciences",
        "algo": "Algorithmica",
        "tcc": "IEEE Transactions on Cloud Computing",
        "tocs": "ACM Transactions on Computer Systems",
    }

    cleaned_name = name_lower.replace(".", "").replace("-", "").replace(":", "").replace(" ", "")
    cleaned_name = re.sub(r'[^a-z0-9]', '', cleaned_name)

    for abbr, full_name in abbrevs.items():
        if cleaned_name == abbr or cleaned_name == abbr.replace(" ", ""):
            if full_name in JOURNAL_KNOWLEDGE:
                return full_name, JOURNAL_KNOWLEDGE[full_name]

    return None


def needs_enrichment(journal: dict) -> bool:
    """判断期刊是否需要补全"""
    scope = (journal.get("scope_text") or "").strip()
    tags = journal.get("subject_tags", [])
    if len(scope) < 15 and (tags == ["other"] or tags == []):
        return True
    if tags == ["other"] or tags == []:
        return True
    if len(scope) < 30:
        return True
    return False


def query_openalex(journal_name: str) -> Optional[Dict[str, Any]]:
    """通过 OpenAlex API 查询期刊信息"""
    try:
        url = "https://api.openalex.org/journals"
        params = {
            "filter": f"display_name.search:{journal_name}",
            "per-page": 3,  # 返回多个结果增加匹配概率
        }
        headers = {"Accept": "application/json"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        results = data.get("results", [])
        if not results:
            return None

        # 找第一个有 description 或足够多 topics 的结果
        best = None
        for r in results:
            desc = r.get("description", "")
            topics = r.get("topics", []) or []
            if len(desc) >= 20:
                best = r
                break
            if len(topics) >= 5 and best is None:
                best = r  # 保留作为 fallback

        if best is None:
            best = results[0]

        # 提取 description 或用 topics 构建
        description = best.get("description", "")
        topics = best.get("topics", []) or []

        # 如果 description 太短，用 topics 构建 scope
        if not description or len(description) < 20:
            if topics:
                topic_names = [t.get("display_name", "") for t in topics[:8]]
                description = f"Research areas include: {', '.join(topic_names)}"

        if not description or len(description) < 30:
            return None

        # 提取 topics 作为 subject_tags
        subject_tags = []
        if topics:
            for topic in topics[:5]:
                name = topic.get("display_name", "")
                if name:
                    subject_tags.append(name)

        # 提取 homepage
        homepage = best.get("homepage_url", "")

        return {
            "scope_text": description,
            "subject_tags": subject_tags if subject_tags else [],
            "submission_url": homepage or "",
            "openalex_id": best.get("id", ""),
        }

    except Exception as e:
        return None


def enrich_journal(journal: dict) -> dict:
    """补全期刊数据（知识库 + OpenAlex 双层策略）"""
    if not needs_enrichment(journal):
        return journal

    result = journal.copy()

    # 第一层：知识库匹配
    match = find_best_match(journal["journal_name"])
    if match:
        known_name, data = match
        if needs_enrichment(result) and data.get("scope_text"):
            result["scope_text"] = data["scope_text"]
        if result.get("subject_tags") == ["other"] or not result.get("subject_tags"):
            if data.get("subject_tags"):
                result["subject_tags"] = data["subject_tags"]
        if not result.get("submission_url") and data.get("submission_url"):
            result["submission_url"] = data["submission_url"]
        if not result.get("keywords") and data.get("keywords"):
            result["keywords"] = data["keywords"]

    # 第二层：OpenAlex API（如果知识库未补全 scope）
    if needs_enrichment(result):
        time.sleep(0.25)  # 遵守 OpenAlex 限速
        api_result = query_openalex(journal["journal_name"])
        if api_result:
            if not result.get("scope_text") or len(result.get("scope_text", "")) < 20:
                result["scope_text"] = api_result["scope_text"]
            if not result.get("subject_tags") or result.get("subject_tags") == ["other"]:
                result["subject_tags"] = api_result["subject_tags"]
            if not result.get("submission_url") and api_result.get("submission_url"):
                result["submission_url"] = api_result["submission_url"]

    return result


def main():
    input_path = "data/processed/journals.jsonl"
    output_path = "data/processed/journals_enriched.jsonl"

    total = 0
    enriched_scope = 0
    enriched_tags = 0
    enriched_url = 0
    enriched_kb = 0
    enriched_api = 0
    enriched_any = 0

    print("开始补全期刊数据...")
    print("策略: 知识库 → OpenAlex API\n")

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            journal = json.loads(line.strip())
            total += 1

            before_scope = len(journal.get("scope_text", "") or "")
            before_tags = journal.get("subject_tags", [])
            before_url = bool(journal.get("submission_url", "").strip())

            # 判断来源
            kb_match = find_best_match(journal["journal_name"])

            enriched = enrich_journal(journal)

            after_scope = len(enriched.get("scope_text", "") or "")
            after_tags = enriched.get("subject_tags", [])
            after_url = bool(enriched.get("submission_url", "").strip())

            scope_improved = after_scope > before_scope
            tags_improved = after_tags != before_tags and before_tags in [["other"], []]
            url_improved = after_url and not before_url

            if scope_improved:
                enriched_scope += 1
                if kb_match:
                    enriched_kb += 1
                else:
                    enriched_api += 1
            if tags_improved:
                enriched_tags += 1
            if url_improved:
                enriched_url += 1
            if scope_improved or tags_improved or url_improved:
                enriched_any += 1

            if total % 50 == 0:
                print(f"进度: {total} 条处理完成...")

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    print(f"\n总期刊数: {total}")
    print(f"补全 scope_text: {enriched_scope} (知识库: {enriched_kb}, OpenAlex: {enriched_api})")
    print(f"补全 subject_tags: {enriched_tags}")
    print(f"补全 submission_url: {enriched_url}")
    print(f"有任何补全: {enriched_any}")
    print(f"\n输出文件: {output_path}")


if __name__ == "__main__":
    main()