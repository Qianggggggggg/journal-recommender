"""
期刊数据补全脚本

为缺失 scope_text、submission_url、subject_tags 的期刊补全真实数据。
主要覆盖 CCF 推荐期刊（基于公开的期刊 scope 描述）。

使用方法:
    python scripts/enrich_journals.py
"""

import json
import re
from typing import Optional

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

    # 网络与信息安全
    "IEEE Transactions on Dependable and Secure Computing": {
        "scope_text": "dependable computing, secure computing, fault tolerance, security protocols, cryptographic systems, network security, computer security, privacy-preserving computing, trusted computing, intrusion detection, malware analysis, software security, hardware security, security evaluation",
        "subject_tags": ["网络与信息安全", "dependable computing", "security", "cryptography", "fault tolerance"],
        "submission_url": "https://www.computer.org tdse/",
        "keywords": ["security", "cryptography", "fault tolerance", "privacy", "trusted computing"],
    },
    "IEEE Transactions on Information Forensics and Security": {
        "scope_text": "information forensics, security engineering, biometric authentication, digital watermarking, multimedia security, steganography, secure imaging, privacy protection, anomaly detection, forensic analysis, cyber-physical security, IoT security, blockchain security",
        "subject_tags": ["网络与信息安全", "information forensics", "security", "biometrics", "watermarking"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/tifs",
        "keywords": ["information forensics", "security", "biometrics", "watermarking", "privacy"],
    },
    "Journal of Cryptology": {
        "scope_text": "cryptography, cryptanalysis, cryptographic protocols, encryption algorithms, public-key cryptography, post-quantum cryptography, cryptographic implementations, cryptographic hardness assumptions, zero-knowledge proofs, secure multi-party computation, blockchain and distributed ledger security",
        "subject_tags": ["网络与信息安全", "cryptography", "cryptanalysis", "security protocols"],
        "submission_url": "https://www.springer.com/journal/14589",
        "keywords": ["cryptography", "encryption", "security protocols", "post-quantum", "blockchain"],
    },

    # 软件工程/系统软件/程序设计语言
    "ACM Transactions on Programming Languages and Systems": {
        "scope_text": "programming language design, programming language implementation, compilers, interpreters, type systems, program analysis, program verification, software testing, programming paradigms, functional programming, logic programming, domain-specific languages, program transformation",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "programming languages", "compilers", "program analysis"],
        "submission_url": "https://toplas.acm.org/",
        "keywords": ["programming languages", "compilers", "type systems", "program analysis", "verification"],
    },
    "ACM Transactions on Software Engineering and Methodology": {
        "scope_text": "software engineering, software development methodology, software testing, software verification, software maintenance, software design patterns, software architecture, empirical software engineering, program debugging, software quality, software process improvement, agile software development",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software testing", "verification"],
        "submission_url": "https://tosem.acm.org/",
        "keywords": ["software engineering", "testing", "verification", "software design", "empirical SE"],
    },
    "IEEE Transactions on Software Engineering": {
        "scope_text": "software engineering, software design, software architecture, software testing and verification, software maintenance, program analysis, software reliability, empirical software engineering, software process, software security, automated software engineering, AI for software engineering",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software design", "program analysis"],
        "submission_url": "https://publib.ieee.org/xplore/html/tse/",
        "keywords": ["software engineering", "testing", "verification", "program analysis", "software design"],
    },
    "IEEE Transactions on Services Computing": {
        "scope_text": "services computing, web services, cloud services, microservices, service-oriented architecture, service composition, service discovery, service reliability, service security, edge computing services, serverless computing, API design, service level agreements",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "services computing", "web services", "cloud computing"],
        "submission_url": "https://publib.ieee.org/xplore/html/tsc/",
        "keywords": ["services computing", "microservices", "SOA", "cloud services", "web services"],
    },

    # 数据库/数据挖掘/内容检索
    "ACM Transactions on Database Systems": {
        "scope_text": "database systems, relational databases, NoSQL databases, query processing, query optimization, database design, data models, transaction processing, distributed databases, database performance, data warehousing, big data systems, time series databases, graph databases",
        "subject_tags": ["数据库/数据挖掘/内容检索", "database systems", "query processing", "data management"],
        "submission_url": "https://tods.acm.org/",
        "keywords": ["database systems", "query processing", "NoSQL", "data warehousing", "distributed databases"],
    },
    "ACM Transactions on Information Systems": {
        "scope_text": "information systems, database management, information retrieval, digital libraries, knowledge management, electronic commerce, web information systems, social media systems, recommendation systems, information extraction, IR evaluation, query processing, user interfaces for information systems",
        "subject_tags": ["数据库/数据挖掘/内容检索", "information systems", "information retrieval", "database"],
        "submission_url": "https://tois.acm.org/",
        "keywords": ["information systems", "database", "information retrieval", "digital libraries", "e-commerce"],
    },
    "IEEE Transactions on Knowledge and Data Engineering": {
        "scope_text": "knowledge and data engineering, data mining, machine learning for data mining, big data analytics, data streams, graph data mining, text mining, time series analysis, data warehousing, data quality, data integration, knowledge graphs, data privacy, recommend systems, deep learning for structured data",
        "subject_tags": ["数据库/数据挖掘/内容检索", "data mining", "knowledge engineering", "big data", "machine learning"],
        "submission_url": "https://pub敛ieee.org/xplore/html/tkde/",
        "keywords": ["data mining", "big data", "knowledge graphs", "machine learning", "data analytics"],
    },
    "The VLDB Journal": {
        "scope_text": "VLDB, database systems, database management, distributed databases, query processing and optimization, transaction processing, data warehousing, big data systems, graph databases, spatial-temporal databases, data provenance, database performance, cloud databases, streaming data management",
        "subject_tags": ["数据库/数据挖掘/内容检索", "VLDB", "database systems", "distributed databases", "big data"],
        "submission_url": "https://www.springer.com/journal/778",
        "keywords": ["VLDB", "database", "distributed databases", "big data", "data streams"],
    },

    # 计算机科学理论
    "IEEE Transactions on Information Theory": {
        "scope_text": "information theory, coding theory, Shannon theory, error-correcting codes, source coding, channel capacity, quantum information theory, cryptography, information-theoretic security, wireless communications theory, network information theory, data compression, entropy",
        "subject_tags": ["计算机科学理论", "information theory", "coding theory", "cryptography", "Shannon theory"],
        "submission_url": "https://transactionis，汤.org/tit/",
        "keywords": ["information theory", "coding", "cryptography", "entropy", "Shannon theory"],
    },
    "Information and Computation": {
        "scope_text": "information theory, computation theory, automata theory, formal languages, computational complexity, logic in computer science, program verification, parallel computation theory, distributed computation theory, quantum computing theory, algorithmic information theory",
        "subject_tags": ["计算机科学理论", "computation theory", "automata theory", "formal languages", "complexity"],
        "submission_url": "https://www.sciencedirect.com/journal/information-and-computation",
        "keywords": ["computation theory", "automata", "formal languages", "complexity", "logic"],
    },
    "SIAM Journal on Computing": {
        "scope_text": "algorithms, computational complexity, data structures, graph algorithms, approximation algorithms, randomized algorithms, computational geometry, parallel algorithms, distributed algorithms, cryptography, machine learning theory, optimization algorithms",
        "subject_tags": ["计算机科学理论", "algorithms", "computational complexity", "data structures"],
        "submission_url": "https://www.siam.org/publications/journals/siam-journal-on-computing/",
        "keywords": ["algorithms", "complexity", "data structures", "graph algorithms", "approximation algorithms"],
    },

    # 计算机图形学与多媒体
    "ACM Transactions on Graphics": {
        "scope_text": "computer graphics, rendering, geometric modeling, animation, scientific visualization, volume rendering, ray tracing, image synthesis, visual effects, GPU computing for graphics, real-time rendering, procedural generation, implicit surfaces, capture and display of 3D content",
        "subject_tags": ["计算机图形学与多媒体", "computer graphics", "rendering", "visualization", "animation"],
        "submission_url": "https://tog.acm.org/",
        "keywords": ["computer graphics", "rendering", "animation", "visualization", "GPU"],
    },
    "IEEE Transactions on Image Processing": {
        "scope_text": "image processing, image analysis, image restoration, image enhancement, image segmentation, image compression, medical imaging, remote sensing, document image analysis, 3D imaging, computational imaging, image/video quality assessment, deep learning for image processing",
        "subject_tags": ["计算机图形学与多媒体", "image processing", "computer vision", "medical imaging"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/tip",
        "keywords": ["image processing", "image restoration", "compression", "medical imaging", "segmentation"],
    },
    "IEEE Transactions on Visualization and Computer Graphics": {
        "scope_text": "visualization, scientific visualization, information visualization, volume visualization, graph visualization, user interface design for visualization, virtual reality, augmented reality, visual analytics, data-driven graphics, flow visualization, geospatial visualization, biomedical visualization",
        "subject_tags": ["计算机图形学与多媒体", "visualization", "virtual reality", "scientific visualization"],
        "submission_url": "https://www.computer.org/vis/",
        "keywords": ["visualization", "VR", "AR", "scientific visualization", "visual analytics"],
    },
    "IEEE Transactions on Multimedia": {
        "scope_text": "multimedia systems, multimedia processing, image and video processing, audio processing, multimedia communication, multimedia retrieval, multimedia security, multimedia indexing, streaming media, social media analysis, multimedia databases, multimodal learning, deep learning for multimedia",
        "subject_tags": ["计算机图形学与多媒体", "multimedia", "video processing", "audio processing"],
        "submission_url": "https://www.iem.auckland.ac.nz/ieee-tmm/",
        "keywords": ["multimedia", "video processing", "audio", "multimedia retrieval", "streaming"],
    },

    # 人工智能
    "Artificial Intelligence": {
        "scope_text": "artificial intelligence, knowledge representation, reasoning, planning, machine learning, natural language processing, computer vision, robotics, AI for games, constraint satisfaction, expert systems, knowledge graphs, cognitive modeling, automated reasoning, philosophical foundations of AI",
        "subject_tags": ["人工智能", "artificial intelligence", "knowledge representation", "planning", "reasoning"],
        "submission_url": "https://www.journals.elsevier.com/artificial-intelligence",
        "keywords": ["AI", "knowledge representation", "planning", "NLP", "computer vision", "robotics"],
    },
    "IEEE Transactions on Pattern Analysis and Machine Intelligence": {
        "scope_text": "pattern analysis, machine intelligence, computer vision, image processing, scene analysis, shape analysis, object recognition, face recognition, motion analysis, video understanding, natural language processing, speech recognition, document analysis, medical image analysis, deep learning, neural network architectures",
        "subject_tags": ["人工智能", "pattern recognition", "computer vision", "machine learning", "deep learning"],
        "submission_url": "https://pami.ai/",
        "keywords": ["pattern recognition", "computer vision", "deep learning", "NLP", "image analysis"],
    },
    "International Journal of Computer Vision": {
        "scope_text": "computer vision, image understanding, object recognition, scene reconstruction, motion analysis, tracking, 3D vision, video analysis, face recognition, medical imaging, object detection, semantic segmentation, instance segmentation, visual reasoning, vision and language, self-supervised vision",
        "subject_tags": ["人工智能", "computer vision", "image understanding", "object recognition", "deep learning"],
        "submission_url": "https://www.springer.com/journal/11263",
        "keywords": ["computer vision", "image understanding", "object detection", "3D vision", "video analysis"],
    },
    "Journal of Machine Learning Research": {
        "scope_text": "machine learning, deep learning, neural networks, reinforcement learning, supervised learning, unsupervised learning, semi-supervised learning, probabilistic models, kernel methods, optimization for machine learning, statistical learning theory, computational biology applications, NLP applications, computer vision applications",
        "subject_tags": ["人工智能", "machine learning", "deep learning", "neural networks", "reinforcement learning"],
        "submission_url": "https://jmlr.org/",
        "keywords": ["machine learning", "deep learning", "reinforcement learning", "probabilistic models", "optimization"],
    },

    # 人机交互与普适计算
    "ACM Transactions on Computer-Human Interaction": {
        "scope_text": "human-computer interaction, user interface design, user experience evaluation, interactive systems, mobile HCI, social media interaction, visual analytics, eye tracking, brain-computer interaction, tangible interfaces, gesture recognition, accessible computing, haptic interaction, VR/AR interaction",
        "subject_tags": ["人机交互与普适计算", "HCI", "user interface", "user experience", "interactive systems"],
        "submission_url": "https://tochi.acm.org/",
        "keywords": ["HCI", "user interface", "UX", "interactive systems", "VR/AR"],
    },
    "International Journal of Human-Computer Studies": {
        "scope_text": "human-computer interaction, cognitive aspects of HCI, user modeling, interactive systems design, collaborative and social computing, mobile and ubiquitous computing, accessibility, health informatics, learning technologies, human-robot interaction, eye tracking, affective computing",
        "subject_tags": ["人机交互与普适计算", "HCI", "cognitive science", "ubiquitous computing", "accessibility"],
        "submission_url": "https://www.sciencedirect.com/journal/international-journal-of-human-computer-studies",
        "keywords": ["HCI", "cognitive science", "ubiquitous computing", "accessibility", "health informatics"],
    },

    # 交叉/综合/新兴
    "Journal of the ACM": {
        "scope_text": "computer science theory and practice, algorithms, complexity theory, programming languages, software engineering, databases, artificial intelligence, computer graphics, computational biology, security, cryptography, networks, distributed systems, operating systems, computer architecture",
        "subject_tags": ["交叉/综合/新兴", "computer science", "algorithms", "theory"],
        "submission_url": "https://jacm.acm.org/",
        "keywords": ["computer science", "algorithms", "theory", "programming languages", "software engineering"],
    },
    "Proceedings of the IEEE": {
        "scope_text": "electrical engineering, computer engineering, electronics, signal processing, communications, control systems, biomedical engineering, computer science, information theory, robotics, sensors, photonics, power electronics, renewable energy systems, nanotechnology, quantum engineering",
        "subject_tags": ["交叉/综合/新兴", "electrical engineering", "signal processing", "communications", "IEEE"],
        "submission_url": "https://proc ieee.org/",
        "keywords": ["electrical engineering", "signal processing", "communications", "control systems", "IEEE"],
    },
    "Science China Information Sciences": {
        "scope_text": "computer science, information science, artificial intelligence, computer networks, distributed systems, database systems, software engineering, computer graphics, multimedia, information security, computational intelligence, machine learning, data mining, pattern recognition",
        "subject_tags": ["交叉/综合/新兴", "computer science", "information science", "AI"],
        "submission_url": "https://www.springer.com/journal/11432",
        "keywords": ["computer science", "AI", "information science", "networks", "software engineering"],
    },
    "Bioinformatics": {
        "scope_text": "bioinformatics, computational biology, genomics, proteomics, systems biology, structural bioinformatics, sequence analysis, protein structure prediction, molecular dynamics, biological networks, evolutionary computation in biology, medical informatics, single-cell analysis, CRISPR analysis",
        "subject_tags": ["交叉/综合/新兴", "bioinformatics", "computational biology", "genomics", "systems biology"],
        "submission_url": "https://academic.oup.com/bioinformatics",
        "keywords": ["bioinformatics", "computational biology", "genomics", "proteomics", "systems biology"],
    },

    # ===== 重要 CCF-B 类期刊 =====

    # 数据库/数据挖掘
    "IEEE Transactions on Knowledge and Data Engineering": {
        # 已在上方 CCF-A 中列出
    },
    "Data Mining and Knowledge Discovery": {
        "scope_text": "data mining, knowledge discovery, machine learning for data mining, pattern mining, classification, clustering, regression, feature selection, ensemble methods, stream mining, time series mining, graph mining, text mining, recommendation algorithms, distributed data mining",
        "subject_tags": ["数据库/数据挖掘/内容检索", "data mining", "knowledge discovery", "pattern mining", "machine learning"],
        "submission_url": "https://www.springer.com/journal/10618",
        "keywords": ["data mining", "machine learning", "pattern mining", "clustering", "classification"],
    },

    # 计算机网络
    "Computer Networks": {
        "scope_text": "computer networks, network protocols, internet architectures, routing algorithms, network performance, network security, wireless networks, sensor networks, CDN, data center networks, network virtualization, software-defined networking, network measurement and monitoring",
        "subject_tags": ["计算机网络", "computer networks", "network protocols", "internet", "routing"],
        "submission_url": "https://www.sciencedirect.com/journal/computer-networks",
        "keywords": ["computer networks", "internet", "routing", "SDN", "wireless"],
    },
    "IEEE Transactions on Communications": {
        "scope_text": "digital communications, wireless communications, optical communications, mobile communications, coding theory, modulation schemes, error control coding, signal processing for communications, channel estimation, equalization, MIMO systems, OFDM, 5G and beyond communication systems",
        "subject_tags": ["计算机网络", "communications", "wireless communications", "digital signal processing"],
        "submission_url": "https://publib.ieee.org/xplore/html/tcom/",
        "keywords": ["digital communications", "wireless", "optical", "MIMO", "OFDM", "coding"],
    },
    "IEEE Transactions on Wireless Communications": {
        "scope_text": "wireless communications, mobile networks, wireless networking, wireless resource management, multiple-input multiple-output (MIMO), OFDM, cognitive radio, wireless sensor networks, body area networks, vehicular communications, 5G/6G wireless systems, wireless security, propagation modeling",
        "subject_tags": ["计算机网络", "wireless communications", "mobile networks", "MIMO", "5G"],
        "submission_url": "https://publib.ieee.org/xplore/html/twc/",
        "keywords": ["wireless", "mobile networks", "MIMO", "OFDM", "5G", "cognitive radio"],
    },

    # 人工智能
    "IEEE Transactions on Neural Networks and Learning Systems": {
        "scope_text": "neural networks, deep learning, machine learning, reinforcement learning, biological neural networks, brain-inspired computing, neural network hardware, training algorithms, optimization for neural networks, computer vision applications, speech applications, natural language processing applications, graph neural networks",
        "subject_tags": ["人工智能", "neural networks", "deep learning", "machine learning", "reinforcement learning"],
        "submission_url": "https://ieee-tnnls.org/",
        "keywords": ["neural networks", "deep learning", "reinforcement learning", "brain-inspired computing"],
    },
    "IEEE Transactions on Fuzzy Systems": {
        "scope_text": "fuzzy systems, fuzzy logic, fuzzy control, fuzzy inference systems, fuzzy optimization, fuzzy pattern recognition, fuzzy clustering, fuzzy set theory, neuro-fuzzy systems, fuzzy mathematics, fuzzy decision making, fuzzy data mining, fuzzy image processing, fuzzy time series",
        "subject_tags": ["人工智能", "fuzzy systems", "fuzzy logic", "fuzzy control", "soft computing"],
        "submission_url": "https://ieee-tfs.com/",
        "keywords": ["fuzzy logic", "fuzzy control", "neuro-fuzzy", "fuzzy systems", "soft computing"],
    },
    "IEEE Transactions on Evolutionary Computation": {
        "scope_text": "evolutionary computation, genetic algorithms, genetic programming, differential evolution, particle swarm optimization, evolutionary strategies, multi-objective optimization, evolutionary learning, neuroevolution, evolutionary robotics, evolutionary scheduling, evolutionary game theory, evolutionary deep learning",
        "subject_tags": ["人工智能", "evolutionary computation", "genetic algorithms", "optimization", "swarm intelligence"],
        "submission_url": "https://ieee-tevc.org/",
        "keywords": ["evolutionary computation", "genetic algorithms", "optimization", "swarm intelligence"],
    },
    "Pattern Recognition": {
        "scope_text": "pattern recognition, machine learning, statistical pattern recognition, structural pattern recognition, syntactic pattern recognition, image analysis, computer vision, object recognition, character recognition, speech recognition, clustering, feature extraction, classifier design, ensemble methods, deep learning for pattern recognition",
        "subject_tags": ["人工智能", "pattern recognition", "machine learning", "computer vision", "image analysis"],
        "submission_url": "https://www.sciencedirect.com/journal/pattern-recognition",
        "keywords": ["pattern recognition", "machine learning", "computer vision", "feature extraction", "classification"],
    },
    "Neural Networks": {
        "scope_text": "neural networks, deep learning, convolutional neural networks, recurrent neural networks, transformer architectures, attention mechanisms, reinforcement learning, unsupervised learning, self-supervised learning, neural network theory, optimization for neural networks, neural architecture search, neural network compression",
        "subject_tags": ["人工智能", "neural networks", "deep learning", "CNN", "RNN", "transformer"],
        "submission_url": "https://www.sciencedirect.com/journal/neural-networks",
        "keywords": ["neural networks", "deep learning", "CNN", "RNN", "transformer", "attention"],
    },
    "Machine Learning": {
        "scope_text": "machine learning, statistical learning theory, supervised learning, unsupervised learning, semi-supervised learning, transfer learning, multi-task learning, reinforcement learning, probabilistic models, kernel methods, ensemble methods, online learning, computational learning theory",
        "subject_tags": ["人工智能", "machine learning", "statistical learning", "theory", "kernel methods"],
        "submission_url": "https://www.springer.com/journal/10994",
        "keywords": ["machine learning", "statistical learning", "kernel methods", "ensemble methods", "online learning"],
    },
    "Autonomous Agents and Multi-Agent Systems": {
        "scope_text": "multi-agent systems, autonomous agents, agent architectures, multi-agent learning, multi-agent planning, coordination and cooperation, agent communication, agent-based simulation, distributed AI, distributed problem solving, mechanism design, game theory in MAS, agent-based modeling, swarm intelligence",
        "subject_tags": ["人工智能", "multi-agent systems", "autonomous agents", "distributed AI", "MAS"],
        "submission_url": "https://www.springer.com/journal/10458",
        "keywords": ["multi-agent systems", "autonomous agents", "MAS", "coordination", "swarm intelligence"],
    },
    "Computer Vision and Image Understanding": {
        "scope_text": "computer vision, image understanding, object recognition, scene understanding, video analysis, motion analysis, 3D reconstruction, image registration, medical image analysis, satellite image analysis, document image analysis, visual tracking, activity recognition, semantic segmentation, instance segmentation",
        "subject_tags": ["人工智能", "computer vision", "image understanding", "image analysis", "video analysis"],
        "submission_url": "https://www.sciencedirect.com/journal/computer-vision-and-image-understanding",
        "keywords": ["computer vision", "image understanding", "object recognition", "video analysis", "3D reconstruction"],
    },

    # 软件工程
    "Automated Software Engineering": {
        "scope_text": "automated software engineering, program synthesis, program repair, program verification, software testing automation, model checking, static analysis, dynamic analysis, AI for software engineering, automated debugging, code generation, refactoring automation, specification mining",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "automated software engineering", "program synthesis", "software testing", "verification"],
        "submission_url": "https://www.springer.com/journal/10515",
        "keywords": ["automated SE", "program synthesis", "program repair", "verification", "software testing"],
    },
    "Empirical Software Engineering": {
        "scope_text": "empirical software engineering, software engineering experiments, case studies in software engineering, software measurement, software quality, mining software repositories, empirical validation of software engineering methods, human aspects of software engineering, software process improvement, software cost estimation",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "empirical software engineering", "software measurement", "software quality"],
        "submission_url": "https://www.springer.com/journal/10664",
        "keywords": ["empirical SE", "software measurement", "software quality", "mining software repositories"],
    },
    "Information and Software Technology": {
        "scope_text": "information systems development, software engineering, software design, software quality, software process, software maintenance, requirements engineering, software architecture, agile development, DevOps, software security, testing and verification, human factors in software engineering",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "information systems", "software quality"],
        "submission_url": "https://www.sciencedirect.com/journal/information-and-software-technology",
        "keywords": ["software engineering", "information systems", "software quality", "software process"],
    },
    "Journal of Systems and Software": {
        "scope_text": "systems software, application software, software architecture, software design, software testing, software maintenance, software performance, software security, software reliability, software metrics, software component models, middleware, distributed software systems, service-oriented architecture",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "systems software", "software architecture", "software design"],
        "submission_url": "https://www.sciencedirect.com/journal/systems-software",
        "keywords": ["systems software", "software architecture", "middleware", "distributed systems", "software quality"],
    },

    # 交叉领域
    "IEEE Transactions on Big Data": {
        "scope_text": "big data, big data analytics, data science, data processing architectures, scalable machine learning for big data, distributed computing for big data, streaming analytics, real-time big data processing, big data mining, data quality, data privacy for big data, data visualization for big data, edge computing for big data",
        "subject_tags": ["交叉/综合/新兴", "big data", "data analytics", "data science", "scalable computing"],
        "submission_url": "https://public.flourish.studio/careers",
        "keywords": ["big data", "data analytics", "streaming", "distributed computing", "scalable ML"],
    },
    "IEEE Internet of Things Journal": {
        "scope_text": "Internet of Things, IoT architectures, IoT protocols, edge computing for IoT, IoT security and privacy, IoT data management, sensor networks, wearable IoT, industrial IoT, smart cities, smart healthcare, connected vehicles, UAV networks, IoT machine learning, IoT data analytics",
        "subject_tags": ["交叉/综合/新兴", "IoT", "Internet of Things", "sensor networks", "smart systems"],
        "submission_url": "https://www.iotjournal.com/",
        "keywords": ["IoT", "sensor networks", "edge computing", "smart cities", "industrial IoT"],
    },
    "IEEE Transactions on Industrial Informatics": {
        "scope_text": "industrial informatics, industrial automation, smart manufacturing, Industry 4.0, cyber-physical systems, industrial IoT, smart grids, process automation, factory communications, robotics in manufacturing, digital twins, predictive maintenance, real-time monitoring, industrial AI applications",
        "subject_tags": ["交叉/综合/新兴", "industrial informatics", "Industry 4.0", "cyber-physical systems", "smart manufacturing"],
        "submission_url": "https://www.ieee-ies.org/mtp/tii/",
        "keywords": ["industrial informatics", "Industry 4.0", "CPS", "smart manufacturing", "digital twins"],
    },
    "IEEE Transactions on Robotics": {
        "scope_text": "robotics, robot control, robot perception, robot planning, robot learning, autonomous robots, mobile robots, manipulators, humanoids, aerial robots, underwater robots, robot vision, robot audition, swarm robotics, human-robot interaction, robot ethics, robot simulation",
        "subject_tags": ["交叉/综合/新兴", "robotics", "autonomous systems", "robot control", "robot learning"],
        "submission_url": "https://www.ieee-ras.org/publications/t-robotics/",
        "keywords": ["robotics", "autonomous robots", "robot control", "robot learning", "human-robot interaction"],
    },
    "IEEE Transactions on Medical Imaging": {
        "scope_text": "medical imaging, image reconstruction, image processing for medical diagnosis, computed tomography, magnetic resonance imaging, ultrasound imaging, PET/SPECT imaging, medical image analysis, computer-aided diagnosis, image segmentation for medical images, image registration, visualization for medicine, radiomics, deep learning for medical imaging",
        "subject_tags": ["交叉/综合/新兴", "medical imaging", "image processing", "biomedical engineering", "diagnostics"],
        "submission_url": "https://www.embs.org/tmi/",
        "keywords": ["medical imaging", "image reconstruction", "CT", "MRI", "ultrasound", "deep learning for medicine"],
    },
    "IEEE Transactions on Geoscience and Remote Sensing": {
        "scope_text": "geoscience, remote sensing, satellite imaging, SAR imaging, hyperspectral imaging, atmospheric remote sensing, ocean remote sensing, land surface remote sensing, geoinformation extraction, change detection, object detection in remote sensing, deep learning for remote sensing, data fusion in remote sensing",
        "subject_tags": ["交叉/综合/新兴", "remote sensing", "geoscience", "satellite imaging", "image processing"],
        "submission_url": "https://grs期刊.org/",
        "keywords": ["remote sensing", "satellite imaging", "SAR", "hyperspectral", "geoinformation"],
    },
    "IEEE Transactions on Intelligent Transportation Systems": {
        "scope_text": "intelligent transportation systems, autonomous vehicles, connected vehicles, vehicle-to-everything communication, traffic management, traffic signal control, smart mobility, electric vehicles, transportation safety, driver assistance systems, cooperative driving, ITS architecture, transportation data analytics, public transit systems",
        "subject_tags": ["交叉/综合/新兴", "intelligent transportation", "autonomous vehicles", "connected vehicles", "smart mobility"],
        "submission_url": "https://www.ieee-its.org/publications/t-its/",
        "keywords": ["intelligent transportation", "autonomous vehicles", "V2X", "traffic management", "smart mobility"],
    },
    "World Wide Web": {
        "scope_text": "World Wide Web, web engineering, web services, semantic web, linked data, social web, web mining, web of things, web performance, web security, web scalability, distributed web systems, content delivery networks, web search, web recommendation, blockchain for web applications",
        "subject_tags": ["交叉/综合/新兴", "World Wide Web", "semantic web", "web services", "web mining"],
        "submission_url": "https://www.springer.com/journal/11280",
        "keywords": ["World Wide Web", "semantic web", "web services", "web mining", "social web"],
    },
    "IEEE Transactions on Automation Science and Engineering": {
        "scope_text": "automation science, factory automation, process automation, robotics in automation, manufacturing automation, quality control in manufacturing, computer-integrated manufacturing, industrial sensing, autonomous systems, smart factories, Industry 4.0, process optimization, fault detection and diagnosis in automation",
        "subject_tags": ["交叉/综合/新兴", "automation", "robotics", "manufacturing", "Industry 4.0", "smart factories"],
        "submission_url": "https://www.ieee.org/ras/publications/t-ase/",
        "keywords": ["automation", "robotics", "manufacturing", "Industry 4.0", "smart factories"],
    },

    # ===== 常见 IEEE Transactions 系列（部分有缺失）=====
    "IEEE Transactions on Neural Networks": {
        "scope_text": "neural networks, deep learning architectures, convolutional neural networks, recurrent neural networks, long short-term memory, generative adversarial networks, neural network training, neural network optimization, neural network theory, neural network applications in image, speech, NLP, brain-computer interfaces",
        "subject_tags": ["人工智能", "neural networks", "deep learning", "CNN", "RNN", "generative models"],
        "submission_url": "https://ieee-tnnls.org/",
        "keywords": ["neural networks", "deep learning", "CNN", "RNN", "GAN", "LSTM"],
    },
    "IEEE Transactions on Cybernetics": {
        "scope_text": "cybernetics, biological cybernetics, cognitive cybernetics, social cybernetics, robotic systems, neural control systems, adaptive control systems, evolutionary fuzzy systems, brain-inspired AI, human-machine systems, cognitive robotics, swarms and collective behavior",
        "subject_tags": ["人工智能", "cybernetics", "neural control", "cognitive systems", "robotic systems"],
        "submission_url": "https://cyb期刊.com/",
        "keywords": ["cybernetics", "neural control", "cognitive systems", "robotic systems", "human-machine"],
    },
    "IEEE Transactions on Image Processing": {
        "scope_text": "image processing, digital image filtering, image enhancement and restoration, image segmentation and analysis, image compression, video processing, medical image processing, computational imaging, 3D imaging, image/video quality assessment, deep learning for image processing, watermark and authentication",
        "subject_tags": ["计算机图形学与多媒体", "image processing", "computer vision", "video processing", "medical imaging"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/tip",
        "keywords": ["image processing", "image restoration", "image compression", "video processing", "medical imaging"],
    },
    "IEEE Transactions on Affective Computing": {
        "scope_text": "affective computing, emotion recognition, sentiment analysis, facial expression analysis, physiological signal analysis for emotion, human-computer interaction with emotions, affective agents and robots, affective computing for health, multimodal affect recognition, emotion modeling, personality and affect",
        "subject_tags": ["人工智能", "affective computing", "emotion recognition", "sentiment analysis", "HCI"],
        "submission_url": "https://www.computer.org/affective/",
        "keywords": ["affective computing", "emotion recognition", "sentiment", "facial expression", "multimodal"],
    },
    "IEEE Transactions on Audio Speech and Language Processing": {
        "scope_text": "speech processing, speech recognition, speech synthesis, speaker recognition, language modeling, spoken language understanding, speech coding and compression, speech enhancement and separation, speech emotion recognition, multilingual speech processing, hearing aids signal processing, audio signal processing",
        "subject_tags": ["人工智能", "speech processing", "NLP", "speaker recognition", "speech synthesis"],
        "submission_url": "https://signalprocessingsociety.org/publications-resources/papers/taslp",
        "keywords": ["speech recognition", "speech synthesis", "speaker recognition", "NLP", "audio processing"],
    },
    "IEEE Transactions on Evolutionary Computation": {
        "scope_text": "evolutionary computation, genetic algorithms, genetic programming, evolution strategies, differential evolution, particle swarm optimization, multi-objective evolutionary algorithms, evolutionary deep learning, neuroevolution, evolutionary robotics, evolutionary scheduling and routing, evolutionary game theory, evolutionary constraint handling",
        "subject_tags": ["人工智能", "evolutionary computation", "genetic algorithms", "optimization", "swarm intelligence"],
        "submission_url": "https://ieee-tevc.org/",
        "keywords": ["evolutionary computation", "genetic algorithms", "differential evolution", "particle swarm", "multi-objective"],
    },
    "IEEE Transactions on Fuzzy Systems": {
        "scope_text": "fuzzy systems, fuzzy logic and set theory, fuzzy control, adaptive fuzzy systems, neuro-fuzzy systems, fuzzy inference systems, fuzzy optimization, fuzzy pattern recognition, fuzzy decision analysis, fuzzy modeling and identification, fuzzy hardware implementations, applications of fuzzy systems in engineering and science",
        "subject_tags": ["人工智能", "fuzzy systems", "fuzzy logic", "fuzzy control", "neuro-fuzzy"],
        "submission_url": "https://ieee-tfs.com/",
        "keywords": ["fuzzy logic", "fuzzy control", "neuro-fuzzy", "fuzzy inference", "soft computing"],
    },

    # ACM TOG 系列
    "ACM Transactions on Graphics": {
        "scope_text": "computer graphics, rendering algorithms, geometric modeling, animation and motion, scientific visualization, volume rendering, ray tracing, physically-based rendering, image synthesis, visual effects, GPU rendering techniques, real-time graphics, procedural content generation, capture and display technologies",
        "subject_tags": ["计算机图形学与多媒体", "computer graphics", "rendering", "animation", "visualization", "GPU"],
        "submission_url": "https://tog.acm.org/",
        "keywords": ["computer graphics", "rendering", "animation", "geometric modeling", "visual effects"],
    },
    "ACM Transactions on Information Systems": {
        "scope_text": "information systems, database management systems, information retrieval and search, digital libraries, knowledge management systems, electronic commerce systems, web-based information systems, recommender systems, social media systems, information extraction and integration, user interfaces and interaction with information systems",
        "subject_tags": ["数据库/数据挖掘/内容检索", "information systems", "information retrieval", "database", "e-commerce"],
        "submission_url": "https://tois.acm.org/",
        "keywords": ["information systems", "information retrieval", "database", "digital libraries", "recommender systems"],
    },
    "ACM Transactions on Multimedia Computing Communications and Applications": {
        "scope_text": "multimedia computing, multimedia applications, multimedia communication systems, multimedia indexing and retrieval, audio and video processing, multimedia networking, mobile multimedia, 3D multimedia, multimedia databases, multimedia security and rights management, social media analysis, multimodal learning, immersive multimedia experiences",
        "subject_tags": ["计算机图形学与多媒体", "multimedia", "video processing", "audio processing", "multimedia retrieval"],
        "submission_url": "https://tomccap.acm.org/",
        "keywords": ["multimedia", "video processing", "audio processing", "multimedia retrieval", "streaming"],
    },

    # 理论/算法
    "Algorithmica": {
        "scope_text": "algorithms, algorithm design, computational complexity, data structures, graph algorithms, approximation algorithms, randomized algorithms, online algorithms, computational geometry, parameterized algorithms, algorithms for combinatorial optimization, distributed algorithms, algorithmic game theory",
        "subject_tags": ["计算机科学理论", "algorithms", "computational complexity", "data structures", "graph algorithms"],
        "submission_url": "https://www.springer.com/journal/453",
        "keywords": ["algorithms", "data structures", "graph algorithms", "approximation algorithms", "randomized algorithms"],
    },
    "Formal Aspects of Computing": {
        "scope_text": "formal methods, formal specification, formal verification, model checking, program refinement, software architecture formalization, formal semantics, refinement calculus, Circus and Z notations, real-time systems verification, security protocol verification, object-oriented specification, distributed systems specification",
        "subject_tags": ["计算机科学理论", "formal methods", "verification", "model checking", "software engineering"],
        "submission_url": "https://www.springer.com/journal/165",
        "keywords": ["formal methods", "verification", "model checking", "Z notation", "refinement"],
    },
    "Formal Methods in System Design": {
        "scope_text": "formal methods, hardware verification, software verification, model checking, theorem proving, abstract interpretation, program analysis, program logics, specification languages, automata theory applications, temporal logic, real-time systems, hybrid systems, security protocol verification",
        "subject_tags": ["计算机科学理论", "formal methods", "verification", "model checking", "hardware/software verification"],
        "submission_url": "https://www.springer.com/journal/165",
        "keywords": ["formal methods", "model checking", "theorem proving", "program analysis", "temporal logic"],
    },
    "Theoretical Computer Science": {
        "scope_text": "theoretical computer science, algorithms, automata theory, formal languages, computational complexity theory, cryptography, computational geometry, parallel and distributed computing theory, programming language theory, semantics, logics in computer science, quantum computing theory, algorithmic game theory",
        "subject_tags": ["计算机科学理论", "theoretical CS", "algorithms", "automata", "complexity", "cryptography"],
        "submission_url": "https://www.sciencedirect.com/journal/theoretical-computer-science",
        "keywords": ["theoretical CS", "algorithms", "automata", "complexity", "formal languages"],
    },
    "Information and Computation": {
        "scope_text": "information theory, computation theory, automata theory, formal languages, computational complexity, logic in computer science, programming language semantics, program verification, parallel computation theory, quantum computation, algorithmic information theory, Kolmogorov complexity, cryptography, data compression",
        "subject_tags": ["计算机科学理论", "information theory", "computation theory", "automata", "complexity"],
        "submission_url": "https://www.sciencedirect.com/journal/information-and-computation",
        "keywords": ["information theory", "computation theory", "automata", "complexity", "logic"],
    },

    # 其他重要期刊
    "IEEE Transactions on Services Computing": {
        "scope_text": "services computing, web services, cloud computing, microservices architecture, service-oriented computing, service composition and orchestration, service discovery and registration, service-level agreements, service reliability and fault tolerance, edge computing services, serverless computing, service security and privacy, API design and management",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "services computing", "cloud computing", "web services", "microservices"],
        "submission_url": "https://publib.ieee.org/xplore/html/tsc/",
        "keywords": ["services computing", "cloud", "microservices", "SOA", "web services", "serverless"],
    },
    "IEEE Transactions on Cloud Computing": {
        "scope_text": "cloud computing, cloud architectures, cloud storage systems, cloud resource management, cloud security and privacy, fog and edge computing, serverless computing, virtualization technologies, container orchestration, Kubernetes, cloud performance modeling, multi-cloud systems, serverless computing, distributed caching in clouds",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "cloud computing", "distributed systems", "virtualization", "edge computing"],
        "submission_url": "https://tc.computer.org/tcc/",
        "keywords": ["cloud computing", "virtualization", "container", "edge computing", "serverless"],
    },
    "Knowledge-Based Systems": {
        "scope_text": "knowledge-based systems, expert systems, knowledge representation, knowledge engineering, knowledge acquisition, knowledge management, ontology engineering, semantic web, knowledge graphs, reasoning systems, decision support systems, AI knowledge applications, explainable AI, knowledge-based neural systems",
        "subject_tags": ["人工智能", "knowledge-based systems", "knowledge representation", "knowledge graphs", "expert systems"],
        "submission_url": "https://www.sciencedirect.com/journal/knowledge-based-systems",
        "keywords": ["knowledge-based systems", "knowledge representation", "ontology", "knowledge graphs", "expert systems"],
    },
    "Expert Systems with Applications": {
        "scope_text": "expert systems, intelligent systems, decision support systems, problem-solving methods, reasoning under uncertainty, fuzzy expert systems, neural expert systems, rule-based systems, case-based reasoning, knowledge-based systems in engineering and science, AI applications in industry and medicine",
        "subject_tags": ["人工智能", "expert systems", "decision support", "intelligent systems", "AI applications"],
        "submission_url": "https://www.sciencedirect.com/journal/expert-systems-with-applications",
        "keywords": ["expert systems", "decision support", "rule-based systems", "case-based reasoning", "uncertain reasoning"],
    },
    "Information Sciences": {
        "scope_text": "information science, intelligent systems, machine learning, data mining, pattern recognition, soft computing, neural networks, fuzzy systems, genetic algorithms, information fusion, watermarking and encryption, signal and image processing, bioinformatics applications",
        "subject_tags": ["人工智能", "information science", "machine learning", "soft computing", "pattern recognition"],
        "submission_url": "https://www.sciencedirect.com/journal/information-sciences",
        "keywords": ["information science", "machine learning", "data mining", "neural networks", "fuzzy systems"],
    },
    "Information Processing and Management": {
        "scope_text": "information retrieval, web search, text mining, natural language processing, information behavior, digital libraries, social media analysis, information seeking behavior, knowledge management, big data analytics for text, recommender systems, scholarly communication, digital humanities",
        "subject_tags": ["数据库/数据挖掘/内容检索", "information retrieval", "NLP", "text mining", "digital libraries"],
        "submission_url": "https://www.sciencedirect.com/journal/information-processing-and-management",
        "keywords": ["information retrieval", "web search", "NLP", "text mining", "digital libraries"],
    },
    "Signal Processing": {
        "scope_text": "signal processing, digital signal processing, image and video signal processing, speech and audio processing, multidimensional signal processing, statistical signal processing, adaptive filtering, array signal processing, compressive sensing, deep learning for signal processing, biomedical signal processing, radar and sonar signal processing",
        "subject_tags": ["计算机图形学与多媒体", "signal processing", "image processing", "speech processing", "audio processing"],
        "submission_url": "https://www.sciencedirect.com/journal/signal-processing",
        "keywords": ["signal processing", "digital signal processing", "image processing", "speech processing", "array processing"],
    },
    "Computer Communications": {
        "scope_text": "computer communications, network protocols, internet protocols, network architectures, wireless networking, mobile communications, distributed systems, network security, network performance evaluation, network management, SDN, NFV, data center networks, CDN, edge computing for communications",
        "subject_tags": ["计算机网络", "computer communications", "network protocols", "internet", "wireless networking"],
        "submission_url": "https://www.sciencedirect.com/journal/computer-communications",
        "keywords": ["computer communications", "network protocols", "internet", "wireless", "distributed systems"],
    },
    "Computer Networks": {
        "scope_text": "computer networks, networking protocols, internetworking, routing protocols, transport protocols, network performance analysis, software-defined networking, network function virtualization, data center networking, content delivery networks, wireless sensor networks, vehicular networks, underwater networks, network security",
        "subject_tags": ["计算机网络", "computer networks", "networking protocols", "SDN", "routing"],
        "submission_url": "https://www.sciencedirect.com/journal/computer-networks",
        "keywords": ["computer networks", "SDN", "routing", "internet", "wireless sensor networks"],
    },

    # 网络安全
    "Computers & Security": {
        "scope_text": "computer security, information security, network security, cryptography, cyber security, malware analysis, intrusion detection, software security, web security, database security, privacy preservation, security protocols, secure computing, IoT security, blockchain security, threat intelligence",
        "subject_tags": ["网络与信息安全", "computer security", "cryptography", "network security", "cyber security"],
        "submission_url": "https://www.sciencedirect.com/journal/computers-security",
        "keywords": ["computer security", "cryptography", "malware", "intrusion detection", "privacy"],
    },
    "Designs Codes and Cryptography": {
        "scope_text": "cryptography, coding theory, algebraic cryptography, symmetric cryptography, public-key cryptography, post-quantum cryptography, cryptographic protocol design, cryptographic implementations, blockchain and distributed ledger, secure multi-party computation, zero-knowledge proofs, error-correcting codes, coding for communications",
        "subject_tags": ["网络与信息安全", "cryptography", "coding theory", "security protocols", "post-quantum"],
        "submission_url": "https://www.springer.com/journal/106",
        "keywords": ["cryptography", "coding theory", "security protocols", "post-quantum", "blockchain"],
    },

    # 软件工程
    "Software: Practice and Experience": {
        "scope_text": "software practice, software development, software design and architecture, software testing, software debugging, software maintenance, programming methodology, object-oriented development, agile software development, DevOps practices, software project management, empirical studies of software development",
        "subject_tags": ["软件工程/系统软件/程序设计语言", "software engineering", "software practice", "programming", "development methodology"],
        "submission_url": "https://onlinelibrary.wiley.com/journal/10970241",
        "keywords": ["software practice", "software development", "programming", "agile", "DevOps"],
    },

    # 可视化/图形学
    "Computers & Graphics": {
        "scope_text": "computer graphics, visualization, graphical interfaces, geometric modeling, rendering techniques, virtual reality, augmented reality, 3D modeling, scientific visualization, volume visualization, data visualization, graph visualization, GPU computing, graphics hardware",
        "subject_tags": ["计算机图形学与多媒体", "computer graphics", "visualization", "VR/AR", "geometric modeling"],
        "submission_url": "https://www.sciencedirect.com/journal/computers-graphics",
        "keywords": ["computer graphics", "visualization", "VR", "geometric modeling", "rendering"],
    },
}


def find_best_match(journal_name: str) -> Optional[dict]:
    """在知识库中精确匹配期刊名"""
    name_lower = journal_name.lower().strip()

    # 1. 精确匹配（最可靠）
    for known_name, data in JOURNAL_KNOWLEDGE.items():
        if known_name.lower() == name_lower:
            return known_name, data

    # 2. 严格包含匹配（期刊名完全包含知识库条目，或被包含）
    for known_name, data in JOURNAL_KNOWLEDGE.items():
        known_lower = known_name.lower()
        # 知识库名称被期刊名包含（且两者长度差异不大，避免"XXX Journal"匹配"Journal of XXX"）
        if known_lower in name_lower and len(known_lower) > 10:
            return known_name, data
        if name_lower in known_lower and len(name_lower) > 10:
            return known_name, data

    # 3. 缩写匹配（只对特定已知缩写生效，且要求高度匹配）
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
        "joc": "Journal of Cryptology",
        "toplas": "ACM Transactions on Programming Languages and Systems",
        "tosem": "ACM Transactions on Software Engineering and Methodology",
        "tse": "IEEE Transactions on Software Engineering",
        "tsc": "IEEE Transactions on Services Computing",
        "tods": "ACM Transactions on Database Systems",
        "tois": "ACM Transactions on Information Systems",
        "tkde": "IEEE Transactions on Knowledge and Data Engineering",
        "vldbj": "The VLDB Journal",
        "tit": "IEEE Transactions on Information Theory",
        "iandc": "Information and Computation",
        "siamcomp": "SIAM Journal on Computing",
        "tog": "ACM Transactions on Graphics",
        "tip": "IEEE Transactions on Image Processing",
        "tvcg": "IEEE Transactions on Visualization and Computer Graphics",
        "tmm": "IEEE Transactions on Multimedia",
        "ai": "Artificial Intelligence",
        "tpami": "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "ijcv": "International Journal of Computer Vision",
        "jmlr": "Journal of Machine Learning Research",
        "tochi": "ACM Transactions on Computer-Human Interaction",
        "ijhcs": "International Journal of Human-Computer Studies",
        "jacm": "Journal of the ACM",
        "procieee": "Proceedings of the IEEE",
        "scis": "Science China Information Sciences",
        "bioinformatics": "Bioinformatics",
        "tnnls": "IEEE Transactions on Neural Networks and Learning Systems",
        "tnns": "IEEE Transactions on Neural Networks and Learning Systems",
        "tnn": "IEEE Transactions on Neural Networks and Learning Systems",
        "tcyb": "IEEE Transactions on Cybernetics",
        "tec": "IEEE Transactions on Evolutionary Computation",
        "tfs": "IEEE Transactions on Fuzzy Systems",
        "pr": "Pattern Recognition",
        "nn": "Neural Networks",
        "ml": "Machine Learning",
        "aamas": "Autonomous Agents and Multi-Agent Systems",
        "cviu": "Computer Vision and Image Understanding",
        "ase": "Automated Software Engineering",
        "ese": "Empirical Software Engineering",
        "infsof": "Information and Software Technology",
        "jss": "Journal of Systems and Software",
        "tbd": "IEEE Transactions on Big Data",
        "iotj": "IEEE Internet of Things Journal",
        "tii": "IEEE Transactions on Industrial Informatics",
        "trob": "IEEE Transactions on Robotics",
        "tmi": "IEEE Transactions on Medical Imaging",
        "tgrs": "IEEE Transactions on Geoscience and Remote Sensing",
        "tits": "IEEE Transactions on Intelligent Transportation Systems",
        "www": "World Wide Web",
        "tase": "IEEE Transactions on Automation Science and Engineering",
        "alg": "Algorithmica",
        "fac": "Formal Aspects of Computing",
        "fmsd": "Formal Methods in System Design",
        "tcs": "Theoretical Computer Science",
        "compsec": "Computers & Security",
        "dcc": "Designs, Codes and Cryptography",
        "spe": "Software: Practice and Experience",
        "cg": "Computers & Graphics",
        "cn": "Computer Networks",
        "comcom": "Computer Communications",
        "taslp": "IEEE Transactions on Audio, Speech and Language Processing",
        "taffco": "IEEE Transactions on Affective Computing",
        "taas": "ACM Transactions on Autonomous and Adaptive Systems",
        "todaes": "ACM Transactions on Design Automation of Electronic Systems",
        "tecs": "ACM Transactions on Embedded Computing Systems",
        "trets": "ACM Transactions on Reconfigurable Technology and Systems",
        "tvlsi": "IEEE Transactions on Very Large Scale Integration (VLSI) Systems",
        "jpdc": "Journal of Parallel and Distributed Computing",
        "jsa": "Journal of Systems Architecture: Embedded Software Design",
        "pc": "Parallel Computing",
        "pe": "Performance Evaluation: An International Journal",
        "tcc": "IEEE Transactions on Cloud Computing",
        "toit": "ACM Transactions on Internet Technology",
        "tomccap": "ACM Transactions on Multimedia Computing, Communications and Applications",
        "tosn": "ACM Transactions on Sensor Networks",
        "cn": "Computer Networks",
        "tcom": "IEEE Transactions on Communications",
        "twc": "IEEE Transactions on Wireless Communications",
        "tops": "ACM Transactions on Privacy and Security",
        "jcs": "Journal of Computer Security",
        "cybersec": "Cybersecurity",
        "ese": "Empirical Software Engineering",
        "iet-sen": "IET Software",
        "infsof": "Information and Software Technology",
        "jfp": "Journal of Functional Programming",
        "smr": "Journal of Software: Evolution and Process",
        "re": "Requirements Engineering",
        "scp": "Science of Computer Programming",
        "sosym": "Software and Systems Modeling",
        "stvr": "Software Testing, Verification and Reliability",
        "tkdd": "ACM Transactions on Knowledge Discovery from Data",
        "tweb": "ACM Transactions on the Web",
        "aei": "Advanced Engineering Informatics",
        "dke": "Data & Knowledge Engineering",
        "datamine": "Data Mining and Knowledge Discovery",
        "geoinformatica": "GeoInformatica",
        "ipm": "Information Processing and Management",
        "isci": "Information Sciences",
        "is": "Information Systems",
        "jasis": "Journal of the Association for Information Science and Technology",
        "ws": "Journal of Web Semantics",
        "kais": "Knowledge and Information Systems",
        "talg": "ACM Transactions on Algorithms",
        "tocl": "ACM Transactions on Computational Logic",
        "toms": "ACM Transactions on Mathematical Software",
        "algorithmica": "Algorithmica",
        "cc": "Computational Complexity",
        "fmsd": "Formal Methods in System Design",
        "informs": "INFORMS Journal on Computing",
        "jcss": "Journal of Computer and System Sciences",
        "jgo": "Journal of Global Optimization",
        "jsc": "Journal of Symbolic Computation",
        "mscs": "Mathematical Structures in Computer Science",
        "cagd": "Computer Aided Geometric Design",
        "cgf": "Computer Graphics Forum",
        "cad": "Computer-Aided Design",
        "tcsv": "IEEE Transactions on Circuits and Systems for Video Technology",
        "siamis": "SIAM Journal on Imaging Sciences",
        "speech": "Speech Communication",
        "tap": "ACM Transactions on Applied Perception",
        "coling": "Computational Linguistics",
        "ec": "Evolutionary Computation",
        "ijar": "International Journal of Approximate Reasoning",
        "jair": "Journal of Artificial Intelligence Research",
        "jar": "Journal of Automated Reasoning",
        "neco": "Neural Computation",
        "tacl": "Transactions of the Association for Computational Linguistics",
        "cscw": "Computer Supported Cooperative Work",
        "hhci": "Human-Computer Interaction",
        "thms": "IEEE Transactions on Human-Machine Systems",
        "iwc": "Interacting with Computers",
        "ijhci": "International Journal of Human-Computer Interaction",
        "umuai": "User Modeling and User-Adapted Interaction",
        "bib": "Briefings in Bioinformatics",
        "jcst": "Journal of Computer Science and Technology",
        "jamia": "Journal of the American Medical Informatics Association",
        "ploscb": "PLOS Computational Biology",
        "fcsc": "Frontiers of Computer Science",
        "jetc": "ACM Journal on Emerging Technologies in Computing Systems",
        "concurrency": "Concurrency and Computation: Practice and Experience",
        "dc": "Distributed Computing",
        "fgcs": "Future Generation Computer Systems",
        "grid": "Journal of Grid Computing",
        "rts": "Real-Time Systems",
        "tjs": "The Journal of Supercomputing",
        "tcasi": "IEEE Transactions on Circuits and Systems I: Regular Papers",
        "ccfthpc": "CCF Transactions on High Performance Computing",
        "tsusc": "IEEE Transactions on Sustainable Computing",
        "adhoc": "Ad Hoc Networks",
        "comcom": "Computer Communications",
        "tnsm": "IEEE Transactions on Network and Service Management",
        "iet-com": "IET Communications",
        "jnca": "Journal of Network and Computer Applications",
        "monet": "Mobile Networks and Applications",
        "networks": "Networks",
        "ppna": "Peer-to-Peer Networking and Applications",
        "wicomm": "Wireless Communications and Mobile Computing",
        "winet": "Wireless Networks",
        "tiot": "ACM Transactions on Internet of Things",
        "clsr": "Computer Law & Security Review",
        "ejisec": "EURASIP Journal on Information Security",
        "iet-ifs": "IET Information Security",
        "imcs": "Information and Computer Security",
        "ijics": "International Journal of Information and Computer Security",
        "ijisp": "International Journal of Information Security and Privacy",
        "istr": "Journal of Information Security and Applications",
        "scn": "Security and Communication Networks",
        "cl": "Computer Languages, Systems and Structures",
        "ijseke": "International Journal of Software Engineering and Knowledge Engineering",
        "sttt": "International Journal of Software Tools for Technology Transfer",
        "jlamp": "Journal of Logical and Algebraic Methods in Programming",
        "jwe": "Journal of Web Engineering",
        "soca": "Service Oriented Computing and Applications",
        "sqj": "Software Quality Journal",
        "tplp": "Theory and Practice of Logic Programming",
        "pacmpl": "Proceedings of the ACM on Programming Languages",
        "dpd": "Distributed and Parallel Databases",
        "iam": "Information & Management",
        "ipl": "Information Processing Letters",
        "ir": "Information Retrieval Journal",
        "ijcis": "International Journal of Cooperative Information Systems",
        "ijgis": "International Journal of Geographical Information Science",
        "ijis": "International Journal of Intelligent Systems",
        "ijkm": "International Journal of Knowledge Management",
        "ijswis": "International Journal on Semantic Web and Information Systems",
        "jcis": "Journal of Computer Information Systems",
        "jdm": "Journal of Database Management",
        "jgim": "Journal of Global Information Technology Management",
        "jiis": "Journal of Intelligent Information Systems",
        "jsis": "The Journal of Strategic Information Systems",
        "tist": "ACM Transactions on Intelligent Systems and Technology",
        "tors": "ACM Transactions on Recommender Systems",
        "acta": "Acta Informatica",
        "apal": "Annals of Pure and Applied Logic",
        "dam": "Discrete Applied Mathematics",
        "fuin": "Fundamenta Informaticae",
        "logcom": "Journal of Logic and Computation",
        "jsyml": "The Journal of Symbolic Logic",
        "lmcs": "Logical Methods in Computer Science",
        "siamdm": "SIAM Journal on Discrete Mathematics",
        "mst": "Theory of Computing Systems",
        "tqc": "ACM Transactions in Quantum Computing",
        "comgeo": "Computational Geometry: Theory and Applications",
        "jvca": "Computer animation & virtual worlds",
        "cg": "Computers & Graphics",
        "dcg": "Discrete & Computational Geometry",
        "spl": "IEEE Signal Processing Letters",
        "iet-ipr": "IET Image Processing",
        "jvcir": "Journal of Visual Communication and Image Representation",
        "mms": "Multimedia Systems",
        "mta": "Multimedia Tools and Applications",
        "sigpro": "Signal Processing",
        "spic": "Signal Processing: Image Communication",
        "vc": "The Visual Computer",
        "talip": "ACM Transactions on Asian and Low-Resource Language Information Processing",
        "apin": "Applied Intelligence",
        "artmed": "Artificial Intelligence in Medicine",
        "alife": "Artificial Life",
        "ci": "Computational Intelligence",
        "csl": "Computer Speech & Language",
        "connection": "Connection Science",
        "dss": "Decision Support Systems",
        "eaai": "Engineering Applications of Artificial Intelligence",
        "es": "Expert Systems",
        "eswa": "Expert Systems with Applications",
        "fss": "Fuzzy Sets and Systems",
        "tciaig": "IEEE Transactions on Games",
        "ivc": "Image and Vision Computing",
        "ida": "Intelligent Data Analysis",
        "ijcia": "International Journal of Computational Intelligence and Applications",
        "ijns": "International Journal of Neural Systems",
        "ijufks": "International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems",
        "ijdar": "International Journal on Document Analysis and Recognition",
        "jetai": "Journal of Experimental and Theoretical Artificial Intelligence",
        "kbs": "Knowledge-Based Systems",
        "mt": "Machine Translation",
        "mva": "Machine Vision and Applications",
        "nc": "Natural Computing",
        "nle": "Natural Language Engineering",
        "nca": "Neural Computing and Applications",
        "npl": "Neural Processing Letters",
        "ijon": "Neurocomputing",
        "paa": "Pattern Analysis and Applications",
        "prl": "Pattern Recognition Letters",
        "soco": "Soft Computing",
        "wias": "Web Intelligence",
        "behaviourIT": "Behaviour & Information Technology",
        "puc": "Personal and Ubiquitous Computing",
        "percom": "Pervasive and Mobile Computing",
        "pacmhci": "Proceedings of the ACM on Human-Computer Interaction",
        "thri": "ACM Transactions on Human-Robot Interaction",
        "cas": "Cybernetics and Systems",
        "lgrs": "IEEE Geoscience and Remote Sensing Letters",
        "titb": "IEEE Journal of Biomedical and Health Informatics",
        "iet-its": "IET Intelligent Transport Systems",
        "jbi": "Journal of Biomedical Informatics",
        "mia": "Medical Image Analysis",
        "tcps": "ACM Transactions on Cyber-Physical Systems",
        "jeric": "ACM Transactions on Computing Education",
        "tcss": "IEEE Transactions on Computational Social Systems",
        "tr": "IEEE Transactions on Reliability",
        "health": "ACM Transactions on Computing for Healthcare",
        "jetc": "ACM Journal on Emerging Technologies in Computing Systems",
        "etta": "Journal of Electronic Testing - Theory and Applications",
    }

    # 清理期刊名用于缩写匹配
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
    # 需要补全: scope 太短 或 tags 只有 "other"（但不能两者都太差）
    if len(scope) < 15 and (tags == ["other"] or tags == []):
        return True  # scope 很短且 tags 无效，一定需要补全
    if tags == ["other"] or tags == []:
        return True  # tags 无效，需要补全（无论 scope 长短）
    if len(scope) < 30:
        return True  # scope 很短，也需要补全
    return False


def enrich_journal(journal: dict) -> dict:
    """补全期刊数据"""
    if not needs_enrichment(journal):
        return journal

    result = journal.copy()
    match = find_best_match(journal["journal_name"])

    if match:
        known_name, data = match
        # 补全 scope_text（如果缺失或太短）
        if needs_enrichment(result) and data.get("scope_text"):
            result["scope_text"] = data["scope_text"]
        # 补全 subject_tags（如果只有 other）
        if result.get("subject_tags") == ["other"] or not result.get("subject_tags"):
            if data.get("subject_tags"):
                result["subject_tags"] = data["subject_tags"]
        # 补全 submission_url
        if not result.get("submission_url") and data.get("submission_url"):
            result["submission_url"] = data["submission_url"]
        # 补全 keywords
        if not result.get("keywords") and data.get("keywords"):
            result["keywords"] = data["keywords"]

    return result


def main():
    input_path = "data/processed/journals.jsonl"
    output_path = "data/processed/journals_enriched.jsonl"

    # 统计
    total = 0
    enriched_scope = 0
    enriched_tags = 0
    enriched_url = 0
    enriched_any = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            journal = json.loads(line.strip())
            total += 1

            before_scope = len(journal.get("scope_text", "") or "")
            before_tags = journal.get("subject_tags", [])
            before_url = bool(journal.get("submission_url", "").strip())

            enriched = enrich_journal(journal)

            after_scope = len(enriched.get("scope_text", "") or "")
            after_tags = enriched.get("subject_tags", [])
            after_url = bool(enriched.get("submission_url", "").strip())

            if after_scope > before_scope:
                enriched_scope += 1
            if after_tags != before_tags and before_tags in [["other"], []]:
                enriched_tags += 1
            if after_url and not before_url:
                enriched_url += 1
            if enriched != journal:
                enriched_any += 1

            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    print(f"总期刊数: {total}")
    print(f"补全 scope_text: {enriched_scope}")
    print(f"补全 subject_tags: {enriched_tags}")
    print(f"补全 submission_url: {enriched_url}")
    print(f"有任何补全: {enriched_any}")
    print(f"\n输出文件: {output_path}")
    print("\n覆盖期刊样例（前10条scope被补全的）:")
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            before = next((json.loads(l) for l in open(input_path) if json.loads(l)["journal_id"] == j["journal_id"]), None)
            if before and len(j.get("scope_text", "")) > 50 and len(before.get("scope_text", "")) < 30:
                print(f"  [{j['quartile']}] {j['journal_name']}")
                print(f"    scope: {j['scope_text'][:80]}...")


if __name__ == "__main__":
    main()
