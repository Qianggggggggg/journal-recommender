"""CCF 推荐期刊目录数据 - 基于第七版 CCF 推荐国际学术会议和期刊目录"""
import json

# CCF A 类期刊（按领域）
CCF_A_JOURNALS = {
    "体系结构/并行计算/存储系统": [
        {"journal_id": "tocs", "journal_name": "ACM Transactions on Computer Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tocs/"},
        {"journal_id": "tos", "journal_name": "ACM Transactions on Storage", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tos/"},
        {"journal_id": "tcad", "journal_name": "IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tcad/"},
        {"journal_id": "tc", "journal_name": "IEEE Transactions on Computers", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tc/"},
        {"journal_id": "tpds", "journal_name": "IEEE Transactions on Parallel and Distributed Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tpds/"},
        {"journal_id": "taco", "journal_name": "ACM Transactions on Architecture and Code Optimization", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/taco/"},
    ],
    "计算机网络": [
        {"journal_id": "jsac", "journal_name": "IEEE Journal on Selected Areas in Communications", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/jsac/"},
        {"journal_id": "tmc", "journal_name": "IEEE Transactions on Mobile Computing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tmc/"},
        {"journal_id": "ton", "journal_name": "IEEE/ACM Transactions on Networking", "publisher": "IEEE/ACM", "url": "http://dblp.uni-trier.de/db/journals/ton/"},
    ],
    "网络与信息安全": [
        {"journal_id": "tdsc", "journal_name": "IEEE Transactions on Dependable and Secure Computing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tdsc/"},
        {"journal_id": "tifs", "journal_name": "IEEE Transactions on Information Forensics and Security", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tifs/"},
        {"journal_id": "joc", "journal_name": "Journal of Cryptology", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/joc/"},
    ],
    "软件工程/系统软件/程序设计语言": [
        {"journal_id": "toplas", "journal_name": "ACM Transactions on Programming Languages and Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/toplas/"},
        {"journal_id": "tosem", "journal_name": "ACM Transactions on Software Engineering and Methodology", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tosem/"},
        {"journal_id": "tse", "journal_name": "IEEE Transactions on Software Engineering", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tse/"},
        {"journal_id": "tsc", "journal_name": "IEEE Transactions on Services Computing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tsc/"},
    ],
    "数据库/数据挖掘/内容检索": [
        {"journal_id": "tods", "journal_name": "ACM Transactions on Database Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tods/"},
        {"journal_id": "tois", "journal_name": "ACM Transactions on Information Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tois/"},
        {"journal_id": "tkde", "journal_name": "IEEE Transactions on Knowledge and Data Engineering", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tkde/"},
        {"journal_id": "vldbj", "journal_name": "The VLDB Journal", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/vldb/"},
    ],
    "计算机科学理论": [
        {"journal_id": "tit", "journal_name": "IEEE Transactions on Information Theory", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tit/"},
        {"journal_id": "iandc", "journal_name": "Information and Computation", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/iandc/"},
        {"journal_id": "siamcomp", "journal_name": "SIAM Journal on Computing", "publisher": "SIAM", "url": "http://dblp.uni-trier.de/db/journals/siamcomp/"},
    ],
    "计算机图形学与多媒体": [
        {"journal_id": "tog", "journal_name": "ACM Transactions on Graphics", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tog/"},
        {"journal_id": "tip", "journal_name": "IEEE Transactions on Image Processing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tip/"},
        {"journal_id": "tvcg", "journal_name": "IEEE Transactions on Visualization and Computer Graphics", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tvcg/"},
        {"journal_id": "tmm", "journal_name": "IEEE Transactions on Multimedia", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tmm/"},
    ],
    "人工智能": [
        {"journal_id": "ai", "journal_name": "Artificial Intelligence", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ai/"},
        {"journal_id": "tpami", "journal_name": "IEEE Transactions on Pattern Analysis and Machine Intelligence", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/pami/"},
        {"journal_id": "ijcv", "journal_name": "International Journal of Computer Vision", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ijcv/"},
        {"journal_id": "jmlr", "journal_name": "Journal of Machine Learning Research", "publisher": "MIT Press", "url": "http://dblp.uni-trier.de/db/journals/jmlr/"},
    ],
    "人机交互与普适计算": [
        {"journal_id": "tochi", "journal_name": "ACM Transactions on Computer-Human Interaction", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tochi/"},
        {"journal_id": "ijhcs", "journal_name": "International Journal of Human-Computer Studies", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ijmms/"},
    ],
    "交叉/综合/新兴": [
        {"journal_id": "jacm", "journal_name": "Journal of the ACM", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/jacm/"},
        {"journal_id": "procieee", "journal_name": "Proceedings of the IEEE", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/pieee/"},
        {"journal_id": "scis", "journal_name": "Science China Information Sciences", "publisher": "Science China Press/Springer", "url": "http://dblp.uni-trier.de/db/journals/chinaf/"},
        {"journal_id": "bioinformatics", "journal_name": "Bioinformatics", "publisher": "Oxford University Press", "url": "http://dblp.uni-trier.de/db/journals/bioinformatics/"},
    ],
}

# CCF B 类期刊
CCF_B_JOURNALS = {
    "体系结构/并行计算/存储系统": [
        {"journal_id": "taas", "journal_name": "ACM Transactions on Autonomous and Adaptive Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/taas/"},
        {"journal_id": "todaes", "journal_name": "ACM Transactions on Design Automation of Electronic Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/todaes/"},
        {"journal_id": "tecs", "journal_name": "ACM Transactions on Embedded Computing Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tecs/"},
        {"journal_id": "trets", "journal_name": "ACM Transactions on Reconfigurable Technology and Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/trets/"},
        {"journal_id": "tvlsi", "journal_name": "IEEE Transactions on Very Large Scale Integration (VLSI) Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tvlsi/"},
        {"journal_id": "jpdc", "journal_name": "Journal of Parallel and Distributed Computing", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jpdc/"},
        {"journal_id": "jsa", "journal_name": "Journal of Systems Architecture: Embedded Software Design", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jsa/"},
        {"journal_id": "pc", "journal_name": "Parallel Computing", "publisher": "Elsevier", "url": "https://dblp.org/db/journals/pc/index.html"},
        {"journal_id": "pe", "journal_name": "Performance Evaluation: An International Journal", "publisher": "Elsevier", "url": "https://dblp.org/db/journals/pe/index.html"},
        {"journal_id": "tcc", "journal_name": "IEEE Transactions on Cloud Computing", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tcc/index.html"},
    ],
    "计算机网络": [
        {"journal_id": "toit", "journal_name": "ACM Transactions on Internet Technology", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/toit/"},
        {"journal_id": "tomccap", "journal_name": "ACM Transactions on Multimedia Computing, Communications and Applications", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tomccap/"},
        {"journal_id": "tosn", "journal_name": "ACM Transactions on Sensor Networks", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tosn/"},
        {"journal_id": "cn", "journal_name": "Computer Networks", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cn/"},
        {"journal_id": "tcom", "journal_name": "IEEE Transactions on Communications", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tcom/"},
        {"journal_id": "twc", "journal_name": "IEEE Transactions on Wireless Communications", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/twc/"},
    ],
    "网络与信息安全": [
        {"journal_id": "tops", "journal_name": "ACM Transactions on Privacy and Security", "publisher": "ACM", "url": "https://dblp.org/db/journals/tissec/index.html"},
        {"journal_id": "compsec", "journal_name": "Computers & Security", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/compsec/"},
        {"journal_id": "dcc", "journal_name": "Designs, Codes and Cryptography", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/dcc/"},
        {"journal_id": "jcs", "journal_name": "Journal of Computer Security", "publisher": "IOSPress", "url": "http://dblp.uni-trier.de/db/journals/jcs/"},
        {"journal_id": "cybersec", "journal_name": "Cybersecurity", "publisher": "Springer", "url": "https://dblp.uni-trier.de/db/journals/cybersec/index.html"},
    ],
    "软件工程/系统软件/程序设计语言": [
        {"journal_id": "ase", "journal_name": "Automated Software Engineering", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ase/"},
        {"journal_id": "ese", "journal_name": "Empirical Software Engineering", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ese/"},
        {"journal_id": "iet-sen", "journal_name": "IET Software", "publisher": "IET", "url": "https://dblp.uni-trier.de/db/journals/iet-sen/"},
        {"journal_id": "infsof", "journal_name": "Information and Software Technology", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/infsof/index.html"},
        {"journal_id": "jfp", "journal_name": "Journal of Functional Programming", "publisher": "Cambridge University Press", "url": "http://dblp.uni-trier.de/db/journals/jfp/"},
        {"journal_id": "smr", "journal_name": "Journal of Software: Evolution and Process", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/smr/"},
        {"journal_id": "jss", "journal_name": "Journal of Systems and Software", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jss/"},
        {"journal_id": "re", "journal_name": "Requirements Engineering", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/re/"},
        {"journal_id": "scp", "journal_name": "Science of Computer Programming", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/scp/"},
        {"journal_id": "sosym", "journal_name": "Software and Systems Modeling", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/sosym/"},
        {"journal_id": "stvr", "journal_name": "Software Testing, Verification and Reliability", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/stvr/index.html"},
        {"journal_id": "spe", "journal_name": "Software: Practice and Experience", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/spe/"},
    ],
    "数据库/数据挖掘/内容检索": [
        {"journal_id": "tkdd", "journal_name": "ACM Transactions on Knowledge Discovery from Data", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tkdd/"},
        {"journal_id": "tweb", "journal_name": "ACM Transactions on the Web", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tweb/"},
        {"journal_id": "aei", "journal_name": "Advanced Engineering Informa", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/aei/"},
        {"journal_id": "dke", "journal_name": "Data & Knowledge Engineering", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/dke/"},
        {"journal_id": "datamine", "journal_name": "Data Mining and Knowledge Discovery", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/datamine/"},
        {"journal_id": "ejis", "journal_name": "European Journal of Information Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ejis/"},
        {"journal_id": "geoinformatica", "journal_name": "GeoInformatica", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/geoinformatica/"},
        {"journal_id": "ipm", "journal_name": "Information Processing and Management", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ipm/"},
        {"journal_id": "isci", "journal_name": "Information Sciences", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/isci/"},
        {"journal_id": "is", "journal_name": "Information Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/is/"},
        {"journal_id": "jasis", "journal_name": "Journal of the Association for Information Science and Technology", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/jasis/"},
        {"journal_id": "ws", "journal_name": "Journal of Web Semantics", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ws/"},
        {"journal_id": "kais", "journal_name": "Knowledge and Information Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/kais/"},
    ],
    "计算机科学理论": [
        {"journal_id": "talg", "journal_name": "ACM Transactions on Algorithms", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/talg/"},
        {"journal_id": "tocl", "journal_name": "ACM Transactions on Computational Logic", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tocl/"},
        {"journal_id": "toms", "journal_name": "ACM Transactions on Mathematical Software", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/toms/"},
        {"journal_id": "algorithmica", "journal_name": "Algorithmica", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/algorithmica/"},
        {"journal_id": "cc", "journal_name": "Computational Complexity", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/cc/"},
        {"journal_id": "fac", "journal_name": "Formal Aspects of Computing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/fac/"},
        {"journal_id": "fmsd", "journal_name": "Formal Methods in System Design", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/fmsd/"},
        {"journal_id": "informs", "journal_name": "INFORMS Journal on Computing", "publisher": "INFORMS", "url": "http://dblp.uni-trier.de/db/journals/informs/"},
        {"journal_id": "jcss", "journal_name": "Journal of Computer and System Sciences", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jcss/"},
        {"journal_id": "jgo", "journal_name": "Journal of Global Optimization", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/jgo/"},
        {"journal_id": "jsc", "journal_name": "Journal of Symbolic Computation", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jsc/"},
        {"journal_id": "mscs", "journal_name": "Mathematical Structures in Computer Science", "publisher": "Cambridge University Press", "url": "http://dblp.uni-trier.de/db/journals/mscs/"},
        {"journal_id": "tcs", "journal_name": "Theoretical Computer Science", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/tcs/"},
    ],
    "计算机图形学与多媒体": [
        {"journal_id": "tomccap", "journal_name": "ACM Transactions on Multimedia Computing, Communications and Applications", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tomccap/"},
        {"journal_id": "cagd", "journal_name": "Computer Aided Geometric Design", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cagd/"},
        {"journal_id": "cgf", "journal_name": "Computer Graphics Forum", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/cgf/"},
        {"journal_id": "cad", "journal_name": "Computer-Aided Design", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cad/"},
        {"journal_id": "tcsv", "journal_name": "IEEE Transactions on Circuits and Systems for Video Technology", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tcsv/"},
        {"journal_id": "siamis", "journal_name": "SIAM Journal on Imaging Sciences", "publisher": "SIAM", "url": "http://dblp.uni-trier.de/db/journals/siamis/"},
        {"journal_id": "speech", "journal_name": "Speech Communication", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/speech/"},
    ],
    "人工智能": [
        {"journal_id": "tap", "journal_name": "ACM Transactions on Applied Perception", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/tap/"},
        {"journal_id": "aamas", "journal_name": "Autonomous Agents and Multi-Agent Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/aamas/"},
        {"journal_id": "coling", "journal_name": "Computational Linguistics", "publisher": "MIT Press", "url": "http://dblp.uni-trier.de/db/journals/coling/"},
        {"journal_id": "cviu", "journal_name": "Computer Vision and Image Understanding", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cviu/"},
        {"journal_id": "ec", "journal_name": "Evolutionary Computation", "publisher": "MIT Press", "url": "http://dblp.uni-trier.de/db/journals/ec/"},
        {"journal_id": "taffco", "journal_name": "IEEE Transactions on Affective Computing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/taffco/"},
        {"journal_id": "taslp", "journal_name": "IEEE Transactions on Audio, Speech and Language Processing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/taslp/"},
        {"journal_id": "tcyb", "journal_name": "IEEE Transactions on Cybernetics", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tcyb/"},
        {"journal_id": "tec", "journal_name": "IEEE Transactions on Evolutionary Computation", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tec/"},
        {"journal_id": "tfs", "journal_name": "IEEE Transactions on Fuzzy Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tfs/"},
        {"journal_id": "tnn", "journal_name": "IEEE Transactions on Neural Networks and Learning Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tnn/"},
        {"journal_id": "ijar", "journal_name": "International Journal of Approximate Reasoning", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ijar/"},
        {"journal_id": "jair", "journal_name": "Journal of Artificial Intelligence Research", "publisher": "AAAI", "url": "http://dblp.uni-trier.de/db/journals/jair/index.html"},
        {"journal_id": "jar", "journal_name": "Journal of Automated Reasoning", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/jar/"},
        {"journal_id": "ml", "journal_name": "Machine Learning", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ml/"},
        {"journal_id": "neco", "journal_name": "Neural Computation", "publisher": "MIT Press", "url": "http://dblp.uni-trier.de/db/journals/neco/"},
        {"journal_id": "nn", "journal_name": "Neural Networks", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/nn/"},
        {"journal_id": "pr", "journal_name": "Pattern Recognition", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/pr/"},
        {"journal_id": "tacl", "journal_name": "Transactions of the Association for Computational Linguistics", "publisher": "ACL", "url": "https://dblp.org/db/journals/tacl/index.html"},
    ],
    "人机交互与普适计算": [
        {"journal_id": "cscw", "journal_name": "Computer Supported Cooperative Work", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/cscw/"},
        {"journal_id": "hhci", "journal_name": "Human-Computer Interaction", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/hhci/"},
        {"journal_id": "thms", "journal_name": "IEEE Transactions on Human-Machine Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/thms/"},
        {"journal_id": "iwc", "journal_name": "Interacting with Computers", "publisher": "Oxford University Press", "url": "http://dblp.uni-trier.de/db/journals/iwc/"},
        {"journal_id": "ijhci", "journal_name": "International Journal of Human-Computer Interaction", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/ijhci/"},
        {"journal_id": "umuai", "journal_name": "User Modeling and User-Adapted Interaction", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/umuai/"},
        {"journal_id": "ccftpci", "journal_name": "CCF Transactions on Pervasive Computing and Interaction", "publisher": "CCF/Springer", "url": "https://dblp.org/db/journals/ccftpci/index.html"},
    ],
    "交叉/综合/新兴": [
        {"journal_id": "bib", "journal_name": "Briefings in Bioinformatics", "publisher": "Oxford University Press", "url": "http://dblp.uni-trier.de/db/journals/bib/"},
        {"journal_id": "tase", "journal_name": "IEEE Transactions on Automation Science and Engineering", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tase/"},
        {"journal_id": "tgrs", "journal_name": "IEEE Transactions on Geoscience and Remote Sensing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tgrs/"},
        {"journal_id": "tits", "journal_name": "IEEE Transactions on Intelligent Transportation Systems", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tits/"},
        {"journal_id": "tmi", "journal_name": "IEEE Transactions on Medical Imaging", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tmi/"},
        {"journal_id": "trob", "journal_name": "IEEE Transactions on Robotics", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/trob/"},
        {"journal_id": "tcbb", "journal_name": "IEEE/ACM Transactions on Computational Biology and Bioinformatics", "publisher": "IEEE/ACM", "url": "http://dblp.uni-trier.de/db/journals/tcbb/"},
        {"journal_id": "jcst", "journal_name": "Journal of Computer Science and Technology", "publisher": "Science Press/Springer", "url": "http://dblp.uni-trier.de/db/journals/jcst/"},
        {"journal_id": "jamia", "journal_name": "Journal of the American Medical Informatics Association", "publisher": "BMJ", "url": "http://dblp.uni-trier.de/db/journals/jamia/"},
        {"journal_id": "ploscb", "journal_name": "PLOS Computational Biology", "publisher": "PLOS", "url": "http://dblp.uni-trier.de/db/journals/ploscb/"},
        {"journal_id": "www", "journal_name": "World Wide Web", "publisher": "Springer", "url": "https://dblp.org/db/journals/www/index.html"},
        {"journal_id": "fcsc", "journal_name": "Frontiers of Computer Science", "publisher": "Higher Education Press", "url": "http://dblp.uni-trier.de/db/journals/fcsc/"},
    ],
}

# CCF C 类期刊
CCF_C_JOURNALS = {
    "体系结构/并行计算/存储系统": [
        {"journal_id": "jetc", "journal_name": "ACM Journal on Emerging Technologies in Computing Systems", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/jetc/"},
        {"journal_id": "concurrency", "journal_name": "Concurrency and Computation: Practice and Experience", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/concurrency/"},
        {"journal_id": "dc", "journal_name": "Distributed Computing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/dc/"},
        {"journal_id": "fgcs", "journal_name": "Future Generation Computer Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/fgcs"},
        {"journal_id": "integration", "journal_name": "Integration, the VLSI Journal", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/integration"},
        {"journal_id": "etta", "journal_name": "Journal of Electronic Testing - Theory and Applications", "publisher": "Springer", "url": "https://dblp.org/db/journals/et/index.html"},
        {"journal_id": "grid", "journal_name": "Journal of Grid computing", "publisher": "Springer", "url": "https://dblp.org/db/journals/grid/"},
        {"journal_id": "rts", "journal_name": "Real-Time Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/rts/"},
        {"journal_id": "tjs", "journal_name": "The Journal of Supercomputing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/tjs/"},
        {"journal_id": "tcasi", "journal_name": "IEEE Transactions on Circuits and Systems I: Regular Papers", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tcasI/index.html"},
        {"journal_id": "ccfthpc", "journal_name": "CCF Transactions on High Performance Computing", "publisher": "CCF", "url": "https://dblp.org/db/journals/ccfthpc/index.html"},
        {"journal_id": "tsusc", "journal_name": "IEEE Transactions on Sustainable Computing", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tsusc/"},
    ],
    "计算机网络": [
        {"journal_id": "adhoc", "journal_name": "Ad Hoc Networks", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/adhoc/"},
        {"journal_id": "comcom", "journal_name": "Computer Communications", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/comcom/"},
        {"journal_id": "tnsm", "journal_name": "IEEE Transactions on Network and Service Management", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tnsm/"},
        {"journal_id": "iet-com", "journal_name": "IET Communications", "publisher": "IET", "url": "http://dblp.uni-trier.de/db/journals/iet-com/"},
        {"journal_id": "jnca", "journal_name": "Journal of Network and Computer Applications", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jnca/"},
        {"journal_id": "monet", "journal_name": "Mobile Networks and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/monet/"},
        {"journal_id": "networks", "journal_name": "Networks", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/networks/"},
        {"journal_id": "ppna", "journal_name": "Peer-to-Peer Networking and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ppna/"},
        {"journal_id": "wicomm", "journal_name": "Wireless Communications and Mobile Computing", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/wicomm/"},
        {"journal_id": "winet", "journal_name": "Wireless Networks", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/winet/"},
        {"journal_id": "iotj", "journal_name": "IEEE Internet of Things Journal", "publisher": "IEEE", "url": "https://dblp.org/db/journals/iotj/index.html"},
        {"journal_id": "tiot", "journal_name": "ACM Transactions on Internet of Things", "publisher": "ACM", "url": "https://dblp.org/db/journals/tiot/index.html"},
    ],
    "网络与信息安全": [
        {"journal_id": "clsr", "journal_name": "Computer Law & Security Review", "publisher": "Elsevier", "url": "https://dblp.org/db/journals/clsr/index.html"},
        {"journal_id": "ejisec", "journal_name": "EURASIP Journal on Information Security", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ejisec/"},
        {"journal_id": "iet-ifs", "journal_name": "IET Information Security", "publisher": "IET", "url": "http://dblp.uni-trier.de/db/journals/iet-ifs/"},
        {"journal_id": "imcs", "journal_name": "Information and Computer Security", "publisher": "Emerald", "url": "http://dblp.uni-trier.de/db/journals/imcs/"},
        {"journal_id": "ijics", "journal_name": "International Journal of Information and Computer Security", "publisher": "Inderscience", "url": "http://dblp.uni-trier.de/db/journals/ijics/"},
        {"journal_id": "ijisp", "journal_name": "International Journal of Information Security and Privacy", "publisher": "IGI Global", "url": "http://dblp.uni-trier.de/db/journals/ijisp/"},
        {"journal_id": "istr", "journal_name": "Journal of Information Security and Applications", "publisher": "Elsevier", "url": "https://dblp.org/db/journals/istr/"},
        {"journal_id": "scn", "journal_name": "Security and Communication Networks", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/scn/"},
        {"journal_id": "hcc", "journal_name": "High-Confidence Computing", "publisher": "Elsevier", "url": "https://www.journals.elsevier.com/high-confidence-computing"},
    ],
    "软件工程/系统软件/程序设计语言": [
        {"journal_id": "cl", "journal_name": "Computer Languages, Systems and Structures", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cl/index.html"},
        {"journal_id": "ijseke", "journal_name": "International Journal of Software Engineering and Knowledge Engineering", "publisher": "World Scientific", "url": "http://dblp.uni-trier.de/db/journals/ijseke/index.html"},
        {"journal_id": "sttt", "journal_name": "International Journal of Software Tools for Technology Transfer", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/sttt/"},
        {"journal_id": "jlamp", "journal_name": "Journal of Logical and Algebraic Methods in Programming", "publisher": "Elsevier", "url": "https://dblp.org/db/journals/jlap/index.html"},
        {"journal_id": "jwe", "journal_name": "Journal of Web Engineering", "publisher": "Rinton Press", "url": "http://dblp.uni-trier.de/db/journals/jwe/"},
        {"journal_id": "soca", "journal_name": "Service Oriented Computing and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/soca/"},
        {"journal_id": "sqj", "journal_name": "Software Quality Journal", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/sqj/"},
        {"journal_id": "tplp", "journal_name": "Theory and Practice of Logic Programming", "publisher": "Cambridge University Press", "url": "http://dblp.uni-trier.de/db/journals/tplp/"},
        {"journal_id": "pacmpl", "journal_name": "Proceedings of the ACM on Programming Languages", "publisher": "ACM", "url": "https://dblp.org/db/journals/pacmpl/index.html"},
    ],
    "数据库/数据挖掘/内容检索": [
        {"journal_id": "dpd", "journal_name": "Distributed and Parallel Databases", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/dpd/"},
        {"journal_id": "iam", "journal_name": "Information & Management", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/iam/"},
        {"journal_id": "ipl", "journal_name": "Information Processing Letters", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ipl/"},
        {"journal_id": "ir", "journal_name": "Information Retrieval Journal", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ir/"},
        {"journal_id": "ijcis", "journal_name": "International Journal of Cooperative Information Systems", "publisher": "World Scientific", "url": "http://dblp.uni-trier.de/db/journals/ijcis/"},
        {"journal_id": "ijgis", "journal_name": "International Journal of Geographical Information Science", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/gis/"},
        {"journal_id": "ijis", "journal_name": "International Journal of Intelligent Systems", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/ijis/"},
        {"journal_id": "ijkm", "journal_name": "International Journal of Knowledge Management", "publisher": "IGI", "url": "http://dblp.uni-trier.de/db/journals/ijkm/"},
        {"journal_id": "ijswis", "journal_name": "International Journal on Semantic Web and Information Systems", "publisher": "IGI", "url": "http://dblp.uni-trier.de/db/journals/ijswis/"},
        {"journal_id": "jcis", "journal_name": "Journal of Computer Information Systems", "publisher": "IACIS", "url": "http://dblp.uni-trier.de/db/journals/jcis/"},
        {"journal_id": "jdm", "journal_name": "Journal of Database Management", "publisher": "IGI-Global", "url": "http://dblp.uni-trier.de/db/journals/jdm/"},
        {"journal_id": "jgim", "journal_name": "Journal of Global Information Technology Management", "publisher": "Ivy League Publishing", "url": "https://dblp.org/db/journals/jgim/index.html"},
        {"journal_id": "jiis", "journal_name": "Journal of Intelligent Information Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/jiis/"},
        {"journal_id": "jsis", "journal_name": "The Journal of Strategic Information Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jsis/"},
        {"journal_id": "tist", "journal_name": "ACM Transactions on Intelligent Systems and Technology", "publisher": "ACM", "url": "https://dblp.org/db/journals/tist/index.html"},
        {"journal_id": "tors", "journal_name": "ACM Transactions on Recommender Systems", "publisher": "ACM", "url": "https://dblp.org/db/journals/tors/index.html"},
    ],
    "计算机科学理论": [
        {"journal_id": "acta", "journal_name": "Acta Informatica", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/acta/"},
        {"journal_id": "apal", "journal_name": "Annals of Pure and Applied Logic", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/apal/"},
        {"journal_id": "dam", "journal_name": "Discrete Applied Mathematics", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/dam/"},
        {"journal_id": "fuin", "journal_name": "Fundamenta Informaticae", "publisher": "IOSPress", "url": "http://dblp.uni-trier.de/db/journals/fuin/"},
        {"journal_id": "ipl", "journal_name": "Information Processing Letters", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ipl/"},
        {"journal_id": "jc", "journal_name": "Journal of Complexity", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jc/"},
        {"journal_id": "logcom", "journal_name": "Journal of Logic and Computation", "publisher": "Oxford University Press", "url": "http://dblp.uni-trier.de/db/journals/logcom/"},
        {"journal_id": "jsyml", "journal_name": "The Journal of Symbolic Logic", "publisher": "Association for Symbolic Logic", "url": "http://dblp.uni-trier.de/db/journals/jsyml/"},
        {"journal_id": "lmcs", "journal_name": "Logical Methods in Computer Science", "publisher": "LMCS", "url": "http://dblp.uni-trier.de/db/journals/lmcs/"},
        {"journal_id": "siamdm", "journal_name": "SIAM Journal on Discrete Mathematics", "publisher": "SIAM", "url": "http://dblp.uni-trier.de/db/journals/siamdm/"},
        {"journal_id": "mst", "journal_name": "Theory of Computing Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/mst/"},
        {"journal_id": "tqc", "journal_name": "ACM Transactions in Quantum Computing", "publisher": "ACM", "url": "https://dblp.org/db/journals/tqc/index.html"},
    ],
    "计算机图形学与多媒体": [
        {"journal_id": "comgeo", "journal_name": "Computational Geometry: Theory and Applications", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/comgeo/"},
        {"journal_id": "jvca", "journal_name": "Computer animation & virtual worlds", "publisher": "Wiley", "url": "https://dblp.org/db/journals/jvca/index.html"},
        {"journal_id": "cg", "journal_name": "Computers & Graphics", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/cg/"},
        {"journal_id": "dcg", "journal_name": "Discrete & Computational Geometry", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/dcg/"},
        {"journal_id": "spl", "journal_name": "IEEE Signal Processing Letters", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/spl/"},
        {"journal_id": "iet-ipr", "journal_name": "IET Image Processing", "publisher": "IET", "url": "http://dblp.uni-trier.de/db/journals/iet-ipr/"},
        {"journal_id": "jvcir", "journal_name": "Journal of Visual Communication and Image Representation", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/jvcir/"},
        {"journal_id": "mms", "journal_name": "Multimedia Systems", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/mms/"},
        {"journal_id": "mta", "journal_name": "Multimedia Tools and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/mta/"},
        {"journal_id": "sigpro", "journal_name": "Signal Processing", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/sigpro/"},
        {"journal_id": "spic", "journal_name": "Signal Processing: Image Communication", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/spic/"},
        {"journal_id": "vc", "journal_name": "The Visual Computer", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/vc/"},
    ],
    "人工智能": [
        {"journal_id": "talip", "journal_name": "ACM Transactions on Asian and Low-Resource Language Information Processing", "publisher": "ACM", "url": "http://dblp.uni-trier.de/db/journals/talip/"},
        {"journal_id": "apin", "journal_name": "Applied Intelligence", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/apin/"},
        {"journal_id": "artmed", "journal_name": "Artificial Intelligence in Medicine", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/artmed/"},
        {"journal_id": "alife", "journal_name": "Artificial Life", "publisher": "MIT Press", "url": "http://dblp.uni-trier.de/db/journals/alife/"},
        {"journal_id": "ci", "journal_name": "Computational Intelligence", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/ci/"},
        {"journal_id": "csl", "journal_name": "Computer Speech & Language", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/csl/"},
        {"journal_id": "connection", "journal_name": "Connection Science", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/connection/"},
        {"journal_id": "dss", "journal_name": "Decision Support Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/dss/"},
        {"journal_id": "eaai", "journal_name": "Engineering Applications of Artificial Intelligence", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/eaai/"},
        {"journal_id": "es", "journal_name": "Expert Systems", "publisher": "Wiley", "url": "http://dblp.uni-trier.de/db/journals/es/"},
        {"journal_id": "eswa", "journal_name": "Expert Systems with Applications", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/eswa/"},
        {"journal_id": "fss", "journal_name": "Fuzzy Sets and Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/fss/"},
        {"journal_id": "tciaig", "journal_name": "IEEE Transactions on Games", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/tciaig/"},
        {"journal_id": "ivc", "journal_name": "Image and Vision Computing", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ivc/"},
        {"journal_id": "ida", "journal_name": "Intelligent Data Analysis", "publisher": "IOSPress", "url": "http://dblp.uni-trier.de/db/journals/ida/"},
        {"journal_id": "ijcia", "journal_name": "International Journal of Computational Intelligence and Applications", "publisher": "World Scientific", "url": "http://dblp.uni-trier.de/db/journals/ijcia/"},
        {"journal_id": "ijns", "journal_name": "International Journal of Neural Systems", "publisher": "World Scientific", "url": "http://dblp.uni-trier.de/db/journals/ijns/"},
        {"journal_id": "ijprai", "journal_name": "International Journal of Pattern Recognition and Artificial Intelligence", "publisher": "World Scientific", "url": "http://dblp.uni-trier.de/db/journals/ijprai/"},
        {"journal_id": "ijufks", "journal_name": "International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems", "publisher": "World Scientific", "url": "https://dblp.uni-trier.de/db/journals/ijufks/"},
        {"journal_id": "ijdar", "journal_name": "International Journal on Document Analysis and Recognition", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/ijdar/"},
        {"journal_id": "jetai", "journal_name": "Journal of Experimental and Theoretical Artificial Intelligence", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/jetai/"},
        {"journal_id": "kbs", "journal_name": "Knowledge-Based Systems", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/kbs/"},
        {"journal_id": "mt", "journal_name": "Machine Translation", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/mt/"},
        {"journal_id": "mva", "journal_name": "Machine Vision and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/mva/"},
        {"journal_id": "nc", "journal_name": "Natural Computing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/nc/"},
        {"journal_id": "nle", "journal_name": "Natural Language Engineering", "publisher": "Cambridge University Press", "url": "http://dblp.uni-trier.de/db/journals/nle/"},
        {"journal_id": "nca", "journal_name": "Neural Computing and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/nca/"},
        {"journal_id": "npl", "journal_name": "Neural Processing Letters", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/npl/"},
        {"journal_id": "ijon", "journal_name": "Neurocomputing", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/ijon/"},
        {"journal_id": "paa", "journal_name": "Pattern Analysis and Applications", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/paa/"},
        {"journal_id": "prl", "journal_name": "Pattern Recognition Letters", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/prl/"},
        {"journal_id": "soco", "journal_name": "Soft Computing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/soco/"},
        {"journal_id": "wias", "journal_name": "Web Intelligence", "publisher": "IOSPress", "url": "http://dblp.uni-trier.de/db/journals/wias/"},
    ],
    "人机交互与普适计算": [
        {"journal_id": "behaviourIT", "journal_name": "Behaviour & Information Technology", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/behaviourIT/"},
        {"journal_id": "puc", "journal_name": "Personal and Ubiquitous Computing", "publisher": "Springer", "url": "http://dblp.uni-trier.de/db/journals/puc/"},
        {"journal_id": "percom", "journal_name": "Pervasive and Mobile Computing", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/percom/"},
        {"journal_id": "pacmhci", "journal_name": "Proceedings of the ACM on Human-Computer Interaction", "publisher": "ACM", "url": "https://dblp.org/db/journals/pacmhci/index.html"},
        {"journal_id": "thri", "journal_name": "ACM Transactions on Human-Robot Interaction", "publisher": "ACM", "url": "https://dblp.org/db/journals/thri/index.html"},
    ],
    "交叉/综合/新兴": [
        {"journal_id": "bmcbi", "journal_name": "BMC Bioinformatics", "publisher": "BioMedCentral", "url": "http://dblp.uni-trier.de/db/journals/bmcbi/"},
        {"journal_id": "cas", "journal_name": "Cybernetics and Systems", "publisher": "Taylor & Francis", "url": "http://dblp.uni-trier.de/db/journals/cas/"},
        {"journal_id": "lgrs", "journal_name": "IEEE Geoscience and Remote Sensing Letters", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/lgrs/"},
        {"journal_id": "titb", "journal_name": "IEEE Journal of Biomedical and Health Informatics", "publisher": "IEEE", "url": "http://dblp.uni-trier.de/db/journals/titb/"},
        {"journal_id": "tbd", "journal_name": "IEEE Transactions on Big Data", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tbd/"},
        {"journal_id": "iet-its", "journal_name": "IET Intelligent Transport Systems", "publisher": "IET", "url": "http://digital-library.theiet.org/content/journals/iet-its"},
        {"journal_id": "jbi", "journal_name": "Journal of Biomedical Informatics", "publisher": "Elsevier", "url": "https://dblp.uni-trier.de/db/journals/jbi/"},
        {"journal_id": "mia", "journal_name": "Medical Image Analysis", "publisher": "Elsevier", "url": "http://dblp.uni-trier.de/db/journals/mia/"},
        {"journal_id": "tii", "journal_name": "IEEE Transactions on Industrial Informatics", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tii/index.html"},
        {"journal_id": "tcps", "journal_name": "ACM Transactions on Cyber-Physical Systems", "publisher": "ACM", "url": "https://dblp.org/db/journals/tcps/index.html"},
        {"journal_id": "jeric", "journal_name": "ACM Transactions on Computing Education", "publisher": "ACM", "url": "https://dblp.org/db/journals/jeric/index.html"},
        {"journal_id": "tcss", "journal_name": "IEEE Transactions on Computational Social Systems", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tcss/index.html"},
        {"journal_id": "tr", "journal_name": "IEEE Transactions on Reliability", "publisher": "IEEE", "url": "https://dblp.org/db/journals/tr/index.html"},
        {"journal_id": "health", "journal_name": "ACM Transactions on Computing for Healthcare", "publisher": "ACM", "url": "https://dblp.uni-trier.de/db/journals/health/index.html"},
    ],
}


def build_journal_list():
    """构建完整期刊列表，带 CCF 分级和领域标签"""
    journals = []

    for category, journal_list in CCF_A_JOURNALS.items():
        for j in journal_list:
            j["subject_tags"] = [category, "CCF-A", "CCF推荐"]
            j["ccf_rank"] = "A"
            j["quartile"] = "Q1"
            journals.append(j)

    for category, journal_list in CCF_B_JOURNALS.items():
        for j in journal_list:
            j["subject_tags"] = [category, "CCF-B", "CCF推荐"]
            j["ccf_rank"] = "B"
            j["quartile"] = "Q2"
            journals.append(j)

    for category, journal_list in CCF_C_JOURNALS.items():
        for j in journal_list:
            j["subject_tags"] = [category, "CCF-C", "CCF推荐"]
            j["ccf_rank"] = "C"
            j["quartile"] = "Q3"
            journals.append(j)

    return journals


if __name__ == "__main__":
    journals = build_journal_list()
    print(f"Total CCF journals: {len(journals)}")
    for j in journals[:10]:
        print(json.dumps(j, ensure_ascii=False, indent=2))