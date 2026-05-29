# 期刊数据库创建计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development

**Goal:** 根据 CSV 文件创建 295 个期刊的 JSON Lines 数据库，每个期刊需要查找真实 scope

**Architecture:** 将任务按专业领域分解为 10 个子任务，每个子agent负责一个领域（如"计算机体系结构/并行与分布计算/存储系统"），对领域内所有期刊进行 web 搜索查找真实 scope，写入统一的 journals_output.jsonl

**Tech Stack:** WebSearch, WebFetch, 手动搜索确认

---

## 期刊列表（按专业领域）

```
1. 计算机体系结构/并行与分布计算/存储系统 (28个)
2. 计算机网络 (21个)
3. 网络与信息安全 (17个)
4. 软件工程/系统软件/程序设计语言 (25个)
5. 数据库/数据挖掘/内容检索 (34个)
6. 计算机科学理论 (28个)
7. 计算机图形学与多媒体 (28个)
8. 人工智能 (64个)
9. 人机交互与普适计算 (15个)
10. 交叉/综合/新兴 (35个)
```

## 输出格式

每个期刊 JSON 对象：
```json
{
  "journal_id": "tocs",
  "journal_name": "ACM Transactions on Computer Systems",
  "publisher": "ACM",
  "subject_tags": ["计算机体系结构/并行与分布计算/存储系统"],
  "ccf_rating": "A",
  "scope_text": "The journal publishes...",
  "submission_url": "https://...",
  "homepage_url": "https://...",
  "keywords": [],
  "oa_type": "subscription",
  "impact_like_score": 0.0,
  "review_time": "",
  "apc": 0.0
}
```

---

## Task 1: 计算机体系结构/并行与分布计算/存储系统 (28个)

**输出文件**: `/Users/qian/PycharmProjects/paper/data/journals_output.jsonl`

**期刊列表**:
1. TOCS - ACM Transactions on Computer Systems - A
2. TOS - ACM Transactions on Storage - A
3. TCAD - IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems - A
4. TC - IEEE Transactions on Computers - A
5. TPDS - IEEE Transactions on Parallel and Distributed Systems - A
6. TACO - ACM Transactions on Architecture and Code Optimization - A
7. TAAS - ACM Transactions on Autonomous and Adaptive Systems - B
8. TODAES - ACM Transactions on Design Automation of Electronic Systems - B
9. TECS - ACM Transactions on Embedded Computing Systems - B
10. TRETS - ACM Transactions on Reconfigurable Technology and Systems - B
11. TVLSI - IEEE Transactions on Very Large Scale Integration (VLSI) Systems - B
12. JPDC - Journal of Parallel and Distributed Computing - B
13. JSA - Journal of Systems Architecture: Embedded Software Design - B
14. Parallel Computing - B (无缩写)
15. Performance Evaluation: An International Journal - B (无缩写)
16. TCC - IEEE Transactions on Cloud Computing - B
17. JETC - ACM Journal on Emerging Technologies in Computing Systems - C
18. Concurrency and Computation: Practice and Experience - C (无缩写)
19. DC - Distributed Computing - C
20. FGCS - Future Generation Computer Systems - C
21. Integration - Integration, the VLSI Journal - C
22. JETTA - Journal of Electronic Testing-Theory and Applications - C
23. JGC - The Journal of Grid computing - C
24. RTS - Real-Time Systems - C
25. TJSC - The Journal of Supercomputing - C
26. TCASI - IEEE Transactions on Circuits and Systems I: Regular Papers - C
27. CCF-THPC - CCF Transactions on High Performance Computing - C
28. TSUSC - IEEE Transactions on Sustainable Computing - C

**执行步骤**:
- [ ] 对每个期刊执行 WebSearch: `"{journal_name}" aims scope site:xxx.com` 或 DBLP
- [ ] 从搜索结果中找到期刊官网或 DBLP 页面
- [ ] 访问期刊官网，提取 "Aims & Scope" 或 "About" 描述
- [ ] 如3次搜索失败，scope_text 设为 "暂无scope"
- [ ] 将结果追加写入 journals_output.jsonl

---

## Task 2: 计算机网络 (21个)

**期刊列表**:
1. JSAC - IEEE Journal on Selected Areas in Communications - A
2. TMC - IEEE Transactions on Mobile Computing - A
3. TON - IEEE Transactions on Networking - A
4. TOIT - ACM Transactions on Internet Technology - B
5. TOMM - ACM Transactions on Multimedia Computing, Communications and Applications - B
6. TOSN - ACM Transactions on Sensor Networks - B
7. CN - Computer Networks - B
8. TCOM - IEEE Transactions on Communications - B
9. TWC - IEEE Transactions on Wireless Communications - B
10. Ad hoc Networks - C (无缩写)
11. CC - Computer Communications - C
12. TNSM - IEEE Transactions on Network and Service Management - C
13. IET Communications - C (无缩写)
14. JNCA - Journal of Network and Computer Applications - C
15. MONET - Mobile Networks and Applications - C
16. Networks - C (无缩写)
17. PPNA - Peer-to-Peer Networking and Applications - C
18. WCMC - Wireless Communications and Mobile Computing - C
19. Wireless Networks - C (无缩写)
20. IOT - IEEE Internet of Things Journal - C
21. TIOT - ACM Transactions on Internet of Things - C

**执行步骤**: 同 Task 1

---

## Task 3: 网络与信息安全 (17个)

**期刊列表**:
1. TDSC - IEEE Transactions on Dependable and Secure Computing - A
2. TIFS - IEEE Transactions on Information Forensics and Security - A
3. Journal of Cryptology - A (无缩写)
4. TOPS - ACM Transactions on Privacy and Security - B
5. Computers & Security - B (无缩写)
6. Designs, Codes and Cryptography - B (无缩写)
7. JCS - Journal of Computer Security - B
8. Cybersecurity - Cybersecurity - B
9. CLSR - Computer Law & Security Review - C
10. EURASIP Journal on Information Security - C (无缩写)
11. IET Information Security - C (无缩写)
12. IMCS - Information and Computer Security - C (无缩写)
13. IJICS - International Journal of Information and Computer Security - C
14. IJISP - International Journal of Information Security and Privacy - C
15. JISA - Journal of Information Security and Applications - C
16. SCN - Security and Communication Networks - C
17. HCC - High-Confidence Computing - C

**执行步骤**: 同 Task 1

---

## Task 4: 软件工程/系统软件/程序设计语言 (25个)

**期刊列表**:
1. TOPLAS - ACM Transactions on Programming Languages and Systems - A
2. TOSEM - ACM Transactions on Software Engineering and Methodology - A
3. TSE - IEEE Transactions on Software Engineering - A
4. TSC - IEEE Transactions on Services Computing - A
5. ASE - Automated Software Engineering - B
6. ESE - Empirical Software Engineering - B
7. IETS - IET Software - B
8. IST - Information and Software Technology - B
9. JFP - Journal of Functional Programming - B
10. Journal of Software: Evolution and Process - B (无缩写)
11. JSS - Journal of Systems and Software - B
12. RE - Requirements Engineering - B
13. SCP - Science of Computer Programming - B
14. SoSyM - Software and Systems Modeling - B
15. STVR - Software Testing, Verification and Reliability - B
16. SPE - Software: Practice and Experience - B
17. CL - Computer Languages, Systems and Structures - C
18. IJSEKE - International Journal of Software Engineering and Knowledge Engineering - C
19. STTT - International Journal of Software Tools for Technology Transfer - C
20. JLAMP - Journal of Logical and Algebraic Methods in Programming - C
21. JWE - Journal of Web Engineering - C
22. SOCA - Service Oriented Computing and Applications - C
23. SQJ - Software Quality Journal - C
24. TPLP - Theory and Practice of Logic Programming - C
25. PACM PL - Proceedings of the ACM on Programming Languages - C

**执行步骤**: 同 Task 1

---

## Task 5: 数据库/数据挖掘/内容检索 (34个)

**期刊列表**:
1. TODS - ACM Transactions on Database Systems - A
2. TOIS - ACM Transactions on Information Systems - A
3. TKDE - IEEE Transactions on Knowledge and Data Engineering - A
4. VLDBJ - The VLDB Journal - A
5. TKDD - ACM Transactions on Knowledge Discovery from Data - B
6. TWEB - ACM Transactions on the Web - B
7. AEI - Advanced Engineering Informatics - B
8. DKE - Data & Knowledge Engineering - B
9. DMKD - Data Mining and Knowledge Discovery - B
10. EJIS - European Journal of Information Systems - B
11. GeoInformatica - B (无缩写)
12. IPM - Information Processing and Management - B
13. Information Sciences - B (无缩写)
14. IS - Information Systems - B
15. JASIST - Journal of the Association for Information Science and Technology - B
16. JWS - Journal of Web Semantics - B
17. KAIS - Knowledge and Information Systems - B
18. DSE - Data Science and Engineering - B
19. DPD - Distributed and Parallel Databases - C
20. I&M - Information & Management - C
21. IPL - Information Processing Letters - C
22. Discover Computing - C (无缩写)
23. IJCIS - International Journal of Cooperative Information Systems - C
24. IJGIS - International Journal of Geographical Information Science - C
25. IJIS - International Journal of Intelligent Systems - C
26. IJKM - International Journal of Knowledge Management - C
27. IJSWIS - International Journal on Semantic Web and Information Systems - C
28. JCIS - Journal of Computer Information Systems - C
29. JDM - Journal of Database Management - C
30. JGITM - Journal of Global Information Technology Management - C
31. JIIS - Journal of Intelligent Information Systems - C
32. JSIS - The Journal of Strategic Information Systems - C
33. TIST - ACM Transactions on Intelligent Systems and Technology - C
34. TORS - ACM Transactions on Recommender Systems - C

**执行步骤**: 同 Task 1

---

## Task 6: 计算机科学理论 (28个)

**期刊列表**:
1. TIT - IEEE Transactions on Information Theory - A
2. IANDC - Information and Computation - A
3. SICOMP - SIAM Journal on Computing - A
4. TALG - ACM Transactions on Algorithms - B
5. TOCL - ACM Transactions on Computational Logic - B
6. TOMS - ACM Transactions on Mathematical Software - B
7. Algorithmica - Algorithmica - B
8. CC - Computational complexity - B
9. FAC - Formal Aspects of Computing - B
10. FMSD - Formal Methods in System Design - B
11. INFORMS - INFORMS Journal on Computing - B
12. JCSS - Journal of Computer and System Sciences - B
13. JGO - Journal of Global Optimization - B
14. JSC - Journal of Symbolic Computation - B
15. MSCS - Mathematical Structures in Computer Science - B
16. TCS - Theoretical Computer Science - B
17. ACTA - Acta Informatica - C
18. APAL - Annals of Pure and Applied Logic - C
19. DAM - Discrete Applied Mathematics - C
20. FUIN - Fundamenta Informaticae - C
21. IPL - Information Processing Letters - C
22. JCOMPLEXITY - Journal of Complexity - C
23. LOGCOM - Journal of Logic and Computation - C
24. JSL - The Journal of Symbolic Logic - C
25. LMCS - Logical Methods in Computer Science - C
26. SIDMA - SIAM Journal on Discrete Mathematics - C
27. Theory of Computing Systems - C (无缩写)
28. TQC - ACM Transactions in Quantum Computing - C

**执行步骤**: 同 Task 1

---

## Task 7: 计算机图形学与多媒体 (28个)

**期刊列表**:
1. TOG - ACM Transactions on Graphics - A
2. TIP - IEEE Transactions on Image Processing - A
3. TVCG - IEEE Transactions on Visualization and Computer Graphics - A
4. TMM - IEEE Transactions on Multimedia - A
5. TOMM - ACM Transactions on Multimedia Computing,Communications and Applications - B
6. CAGD - Computer Aided Geometric Design - B
7. CGF - Computer Graphics Forum - B
8. CAD - Computer-Aided Design - B
9. TCSVT - IEEE Transactions on Circuits and Systems for Video Technology - B
10. JASA - The Journal of the Acoustical Society of America - B
11. SIIMS - SIAM Journal on Imaging Sciences - B
12. SPECOM - Speech Communication - B
13. CVMJ - Computational Visual Media - B
14. CGTA - Computational Geometry: Theory and Applications - C
15. CAVW - computer animation & virtual worlds - C
16. C&G - Computers & Graphics - C
17. DCG - Discrete & Computational Geometry - C
18. SPL - IEEE Signal Processing Letters - C
19. IET-IPR - IET Image Processing - C
20. JVCIR - Journal of Visual Communication and Image Representation - C
21. MS - Multimedia Systems - C
22. MTA - Multimedia Tools and Applications - C
23. SIGPRO - Signal Processing - C
24. IMAGE - Signal Processing: Image Communication - C
25. TVC - The Visual Computer - C
26. VI - Visual Informatics - C
27. VRIH - Virtual Reality & Intelligent Hardware - C
28. GMOD - Graphical Models - C

**执行步骤**: 同 Task 1

---

## Task 8: 人工智能 (64个)

**期刊列表**:
1. AI - Artificial Intelligence - A
2. TPAMI - IEEE Transactions on Pattern Analysis and Machine Intelligence - A
3. IJCV - International Journal of Computer Vision - A
4. JMLR - Journal of Machine Learning Research - A
5. TAP - ACM Transactions on Applied Perception - B
6. AAMAS - Autonomous Agents and Multi-Agent Systems - B
7. Computational Linguistics - B (无缩写)
8. CVIU - Computer Vision and Image Understanding - B
9. DKE - Data & Knowledge Engineering - B
10. Evolutionary Computation - B (无缩写)
11. TAC - IEEE Transactions on Affective Computing - B
12. TASLP - IEEE Transactions on Audio, Speech and Language Processing - B
13. IEEE Transactions on Cybernetics - B (无缩写)
14. TEC - IEEE Transactions on Evolutionary Computation - B
15. TFS - IEEE Transactions on Fuzzy Systems - B
16. TNNLS - IEEE Transactions on Neural Networks and learning systems - B
17. IJAR - International Journal of Approximate Reasoning - B
18. JAIR - Journal of Artificial Intelligence Research - B
19. Journal of Automated Reasoning - B (无缩写)
20. JSLHR - Journal of Speech, Language, and Hearing Research - B
21. Machine Learning - B (无缩写)
22. Neural Computation - B (无缩写)
23. Neural Networks - B (无缩写)
24. PR - Pattern Recognition - B
25. TACL - Transactions of the Association for Computational Linguistics - B
26. TALLIP - ACM Transactions on Asian and Low-Resource Language Information Processing - C
27. Applied Intelligence - C (无缩写)
28. AIM - Artificial Intelligence in Medicine - C
29. Artificial Life - C (无缩写)
30. Computational Intelligence - C (无缩写)
31. Computer Speech & Language - C (无缩写)
32. Connection Science - C (无缩写)
33. DSS - Decision Support Systems - C
34. EAAI - Engineering Applications of Artificial Intelligence - C
35. Expert Systems - C (无缩写)
36. ESWA - Expert Systems with Applications - C
37. Fuzzy Sets and Systems - C (无缩写)
38. TG - IEEE Transactions on Games - C
39. IET-CVI - IET Computer Vision - C
40. IET Signal Processing - C (无缩写)
41. IVC - Image and Vision Computing - C
42. IDA - Intelligent Data Analysis - C
43. IJCIA - International Journal of Computational Intelligence and Applications - C
44. IJIS - International Journal of Intelligent Systems - C
45. IJNS - International Journal of Neural Systems - C
46. IJPRAI - International Journal of Pattern Recognition and Artificial Intelligence - C
47. IJUFKS - International Journal of Uncertainty,Fuzziness and Knowledge-Based Systems - C
48. IJDAR - International Journal on Document Analysis and Recognition - C
49. JETAI - Journal of Experimental and Theoretical Artificial Intelligence - C
50. KBS - Knowledge-Based Systems - C
51. Machine Translation - C (无缩写)
52. Machine Vision and Applications - C (无缩写)
53. Natural Computing - C (无缩写)
54. NLE - Natural Language Engineering - C
55. NCA - Neural Computing and Applications - C
56. NPL - Neural Processing Letters - C
57. Neurocomputing - C (无缩写)
58. PAA - Pattern Analysis and Applications - C
59. PRL - Pattern Recognition Letters - C
60. Soft Computing - C (无缩写)
61. WI - Web Intelligence - C
62. TIIS - ACM Transactions on Interactive Intelligent Systems - C
63. TELO - ACM Transactions on Evolutionary Learning and Optimization - C
64. JATS - ACM Journal on Autonomous Transportation Systems - C

**执行步骤**: 同 Task 1

---

## Task 9: 人机交互与普适计算 (15个)

**期刊列表**:
1. TOCHI - ACM Transactions on Computer-Human Interaction - A
2. IJHCS - International Journal of Human-Computer Studies - A
3. CSCW - Computer Supported Cooperative Work - B
4. HCI - Human-Computer Interaction - B
5. IEEE Transactions on Human-Machine Systems - B (无缩写)
6. IWC - Interacting with Computers - B
7. IJHCI - International Journal of Human-Computer Interaction - B
8. UMUAI - User Modeling and User-Adapted Interaction - B
9. TSMC - IEEE Transactions on Systems, Man, and Cybernetics: Systems - B
10. CCF TPCI - CCF Transactions on Pervasive Computing and Interaction - B
11. BIT - Behaviour & Information Technology - C
12. PUC - Personal and Ubiquitous Computing - C
13. PMC - Pervasive and Mobile Computing - C
14. PACMHCI - Proceedings of the ACM on Human-Computer Interaction - C
15. THRI - ACM Transactions on Human-Robot Interaction - C

**执行步骤**: 同 Task 1

---

## Task 10: 交叉/综合/新兴 (35个)

**期刊列表**:
1. JACM - Journal of the ACM - A
2. Proc. IEEE - Proceedings of the IEEE - A
3. SCIS - Science China Information Sciences - A
4. Bioinformatics - Bioinformatics - A
5. Briefings in Bioinformatics - B (无缩写)
6. Cognition - Cognition - B
7. TASAE - IEEE Transactions on Automation Science and Engineering - B
8. TGARS - IEEE Transactions on Geoscience and Remote Sensing - B
9. TITS - IEEE Transactions on Intelligent Transportation Systems - B
10. TMI - IEEE Transactions on Medical Imaging - B
11. TR - IEEE Transactions on Robotics - B
12. TCBB - IEEE/ACM Transactions on Computational Biology and Bioinformatics - B
13. JCST - Journal of Computer Science and Technology - B
14. JAMIA - Journal of the American Medical Informatics Association - B
15. PLOS Computational Biology - B (无缩写)
16. The Computer Journal - B (无缩写)
17. WWW - The Web Conference - B
18. FCS - Frontiers of Computer Science - B
19. BCRA - Blockchain: Research and Applications - B
20. BMC Bioinformatics - C (无缩写)
21. Cybernetics and Systems - C (无缩写)
22. IEEE Geoscience and Remote Sensing Letters - C (无缩写)
23. JBHI - IEEE Journal of Biomedical and Health Informatics - C
24. TBD - IEEE Transactions on Big Data - C
25. IET Intelligent Transport Systems - C (无缩写)
26. JBI - Journal of Biomedical Informatics - C
27. Medical Image Analysis - C (无缩写)
28. TII - IEEE Transactions on Industrial Informatics - C
29. TCPS - ACM Transactions on Cyber-Physical Systems - C
30. TOCE - ACM Transactions on Computing Education - C
31. EITEE - ENGINEERING Information Technology & Electronic Engineering - C (无缩写)
32. TCSS - IEEE Transactions on Computational Social Systems - C
33. IEEE Transactions on Reliability - C (无缩写)
34. HEALTH - ACM Transactions on Computing for Healthcare - C
35. ACM DLT - ACM Distributed Ledger Technologies: Research and Practice - C

**执行步骤**: 同 Task 1

---

## 输出要求

最终输出文件: `/Users/qian/PycharmProjects/paper/data/journals_output.jsonl`

格式: JSON Lines，每行一个期刊对象

质量要求:
- scope_text 必须是真实从官网获取的内容
- 如确实无法查到，写 "暂无scope"，绝不编造
- publisher 需与实际出版社匹配
- submission_url 和 homepage_url 尽量使用真实 URL