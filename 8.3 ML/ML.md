# ML


机器学习模块构建了涵盖算法理论、工程实践和应用场景的完整ML技术体系，为数据科学家提供系统化的机器学习解决方案。该模块系统性地整理了XGBoost、CatBoost、LightGBM等主流梯度提升算法，JioNLP、Rasa、RocketQA、Haystack等NLP算法工具，以及Optuna自动化参数调优、伪标签技术等前沿ML技术。技术栈涵盖了中文分词器、智能标注工具、信息抽取系统、文本分类算法、文本匹配模型、文本纠错技术等NLP核心组件，深入介绍了搜索推荐算法、语音识别模型、语音合成技术等多模态AI应用。

模块详细解析了PyMuPDF文档处理、HarvestText文本挖掘、funNLP中文工具包、AllenNLP深度学习框架、HanLP自然语言处理、NLTK文本分析等实用工具库的使用方法和最佳实践。内容包括数据预处理、特征工程、模型训练、效果评估等完整的ML工作流，以及数据集成、文本挖掘、预处理优化等数据科学核心技能。

此外，还提供了金融风控、医疗诊断、推荐系统、搜索引擎等典型应用场景的案例分析，以及分布式训练、模型压缩、推理优化等工程化部署技术，帮助开发者掌握从数据到模型的完整机器学习技术栈，实现高效的AI算法开发和产业化应用。

- [机器学习技术](https://github.com/alirezadir/machine-learning-interview-enlightener)
- [funNLP: 中英文敏感词、语言检测、中外手机/电话归属地/运营商查询、名字推断性别、手机号抽取、身份证抽取、邮箱抽取、中日文人名库、中文缩写库、拆字词典、词汇情感值、停用词、反动词表、暴恐词表、繁简体转换、英文模拟中文发音、汪峰歌词生成器、职业名称词库、同义词库、反义词库、否定词库、汽车品牌词库、汽车零件词库、连续英文切割、各种中文词向量、公司名字大全、古诗词库、IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库、中文聊天语料、中文谣言数据、百度中文问答数据集、句子相似度匹配算法集合、bert资源、文本生成&摘要相关工具、cocoNLP信息抽取工具、国内电话号码正则匹配、清华大学XLORE:中英文跨语言百科知识图谱、清华大学人工智能技术系列报](https://github.com/dingidng/funNLP)

# 数据集

- [!NLPDataSet: NLP数据集](https://github.com/liucongg/NLPDataSet)
- [【AdaSeq基础】30+NER数据汇总，涉及多行业、多模态命名实体识别数据集收集_NLP学徒的博客-CSDN博客](https://blog.csdn.net/JohnsonZh/article/details/128916348)
- [阿里天池助力医学领域AI新发展 发布医学信息处理CBLUE数据集 - 知乎](https://zhuanlan.zhihu.com/p/363717559?utm_id=0)
- [中文NER数据集整理 - 知乎](https://zhuanlan.zhihu.com/p/529541521)
- [中文医学NLP资源整理](https://github.com/GanjinZero/awesome_Chinese_medical_NLP)


# 0.机器学习算法

#### 简介
围绕人工智能在不同领域的应用展开。涉及国家电网人工智能竞赛公开赛调度赛题的Catboost模型训练、银行客户认购产品预测以及相关项目的代码实现与数据探索等内容。
#### 核心功能
- 运用Catboost模型对国家电网调度赛题进行训练，并进行5折交叉验证。
- 根据银行客户的多种信息，预测客户是否购买银行认购产品，使用了xgboost、lightgbm、catboost三个模型训练。
#### 技术原理
- Catboost模型训练：通过数据加载、特征提取、模型训练及交叉验证等步骤，利用Catboost算法进行回归预测，过程中涉及数据归一化等操作。
- 银行客户认购产品预测：读取客户相关数据，分离数值与分类变量，通过LabelEncoder对分类变量进行编码，使用lightgbm等模型进行训练和交叉验证。
#### 应用场景
- 电力调度领域：可用于国家电网的电力调度相关决策。
- 银行业务营销：帮助银行预测客户购买行为，制定营销策略。 


- [【数据挖掘入门】使用树模型快速搭建比赛基线模型及进阶分享 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2220613?channelType=0&channel=0)
- [国家电网有限公司2022年人工智能竞赛公开赛-调度赛题baseline - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/5104231?channelType=0&channel=0)
- [银行客户认购产品预测 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/5224269?channelType=0&channel=0)
- [2021科大讯飞-车辆贷款违约预测挑战赛 Top1方案 - 知乎](https://zhuanlan.zhihu.com/p/412337232)
- [飞桨常规赛：点击反欺诈预测 8月第1名方案 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2301871?channelType=0&channel=0)
- [机器学习算法（二）: 基于XGBoost的分类预测 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1566400?channelType=0&channel=0)
- [Catboost/LightGBM/XGBoost预测实现 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/438831?channelType=0&channel=0)
- [2021科大讯飞-车辆贷款违约预测赛事 Top1方案 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3795293?channelType=0&channel=0)

------------------------------------------------------------

## 数据挖掘-机器学习知识

#### 简介
围绕数据挖掘中的二手车价格预测相关内容展开。包含github项目中关于二手车价格预测的资料，以及一篇介绍模型融合技术在二手车价格预测比赛中应用的博客文章，涉及回归任务中的加权融合、分类任务中的Voting、Boosting和Bagging原理及对比、Stacking/Blending构建多层模型原理与实现等。
#### 核心功能
- 提供二手车价格预测相关的数据挖掘知识和模型融合技术。
- 介绍回归和分类任务中不同的模型融合方法，如回归任务中的加权融合、分类任务中的投票法。
- 阐述从样本集角度集成模型的Boosting和Bagging方法及两者区别。
- 讲解Stacking和Blending构建多层模型的原理、实现及比较。
#### 技术原理
- **回归任务中的加权融合**：根据各模型最终预测表现分配权重，改变其对最终结果影响大小，如正确率低的模型给予低权重，正确率高的模型给予高权重，通过加权平均等方式融合模型结果。
- **分类任务中的Voting**：选择所有机器学习算法中输出最多的类，将多个算法预测结果集中，少数服从多数，分为硬投票（直接输出类标签）和软投票（输出类概率）。
- **Boosting**：将弱分类器串联，后一个分类器依赖前一个分类器结果，对训练数据和连接方式操纵以减小误差，各弱分类器相对较弱，多用于集成学习。
- **Bagging**：对样本集采用随机有放回抽样构造分类器，基学习器间无强依赖关系，并行运行，通过投票或平均融合预测结果。
- **Stacking**：构建分层结构，用大量基分类器预测结果作为下一层输入特征，可增加特征，降低再训练过拟合性的方式有选择简单线性模型、利用交叉验证。
- **Blending**：与Stacking相似，建立Holdout集，训练集分两部分，用一部分训练第一层模型，在另一部分上预测结果作为第二层模型训练集特征，预测部分同理，有单纯Holdout和Holdout交叉两个版本。
#### 应用场景
- 二手车交易价格预测比赛。
- 其他需要进行数据挖掘和模型融合以提高预测准确性的相关领域。 


- [dingidng/team-learning-data-mining: 主要存储Datawhale组队学习中“数据挖掘/机器学习”方向的资料。](https://github.com/dingidng/team-learning-data-mining)
- [team-learning-data-mining/SecondHandCarPriceForecast at master · datawhalechina/team-learning-data-mining](https://github.com/datawhalechina/team-learning-data-mining/tree/master/SecondHandCarPriceForecast)
- [零基础数据挖掘入门系列(六) - 模型的融合技术大总结与结果部署_翻滚的小@强的博客-CSDN博客_数据挖掘与融合技术](https://blog.csdn.net/wuzhongqiang/article/details/105012739)
- [特征工程详解及实战项目（2） - 简书](https://www.jianshu.com/p/ae37e6c0e3f1)

-----------------------------------------------------------

## 伪标签

主要介绍了Kaggle知识点中的伪标签，包括其概念、在半监督学习中的作用、与软标签的区别、注意事项及竞赛案例等。
利用现有标注数据训练模型，用该模型对无标注数据预测，将预测标签和数据加入训练集迭代训练，不同竞赛场景有不同的伪标签使用机制。



- [解锁竞赛新姿势-伪标签技术](https://mp.weixin.qq.com/s?__biz=Mzk0NDE5Nzg1Ng==&mid=2247490079&idx=1&sn=473889b74f9805386acce480a98f9829&source=41#wechat_redirect)
- [Kaggle知识点：伪标签Pseudo Label](https://mp.weixin.qq.com/s?__biz=MzIwNDA5NDYzNA==&mid=2247485948&idx=1&sn=f9cfee33df3815891dc79acc17c4d21d&chksm=96c42439a1b3ad2f9ba748e07abdf649f0ee7cd207e925eae0a0add128f535b1a0b185b66f42&scene=178&cur_album_id=1364202321906941952#rd)

------------------------------------------------------------

## 模型参数自动调优


- [optuna/optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna)
- [席卷Kaggle的调参神器，NN和树模型通吃！](https://mp.weixin.qq.com/s/Gzl288KbqL785FwCZJWIew)

------------------------------------------------------------


# 0.NLP算法工具

#### 简介
本内容综合介绍了三个与自然语言处理（NLP）相关的开源项目和教程。`zero_nlp`致力于提供全面的中文NLP解决方案，涵盖大模型复现、数据处理、模型训练与推理部署。`nlp-tutorial`则是一个系统的NLP教程，旨在涵盖词向量、词法分析、预训练语言模型等基础及应用。`JioNLP`是一个功能丰富的中文NLP预处理和解析工具包，专注于提供高效、易用的文本处理功能。

#### 核心功能
*   **zero_nlp**: 中文文本分类、中文GPT2模型、中文CLIP模型、图像生成中文文本、以及ChatGLM-v2-6b等大模型的复现与使用。
*   **nlp-tutorial**: 词向量表示、词法分析、预训练语言模型、文本分类、文本语义匹配、信息抽取、文本生成、实体识别、机器翻译、对话系统等常见NLP任务的教学与实现。
*   **JioNLP**: 文本预处理（如分句、停用词过滤、繁简体转换、拼音转换）、文本数据增强（回译、同音词替换、噪声扰动）、大文件读写、正则抽取与解析（邮箱、身份证号、URL、时间、货币金额、地址等）、各类词典加载、命名实体识别（NER）工具集、文本分类分析、情感分析、时间语义解析、关键短语抽取、文本摘要。

#### 技术原理
这三个项目主要基于**深度学习**和**规则/词典**相结合的自然语言处理技术。`zero_nlp`和`nlp-tutorial`大量依赖于**预训练语言模型**（如GPT2、CLIP、ChatGLM）和**Transformer架构**，并以**PyTorch**作为主要的深度学习框架，实现模型的训练、微调和推理。`JioNLP`则侧重于通过**规则匹配**、**正则表达式解析**和**大规模词典**来实现高效、准确的中文文本预处理和信息抽取，例如其时间语义解析和地址解析等功能主要通过精确的规则集合实现。

#### 应用场景
*   **zero_nlp**: 面向需要快速复现和部署中文大型NLP模型的开发者，适用于构建中文文本分类系统、内容生成应用、图文理解等场景，可作为企业级中文NLP解决方案的参考。
*   **nlp-tutorial**: 适合NLP初学者进行系统性学习，也为研究人员和开发者提供常用NLP任务的基线实现，可应用于学术研究、教育培训以及快速开发原型系统。
*   **JioNLP**: 广泛应用于各种需要对中文文本进行预处理和信息提取的场景，如数据清洗、智能客服、舆情分析、智能推荐系统中的文本规范化、自动化报告生成中的关键信息提取，以及任何涉及中文文本数据准备和解析的NLP项目。


- [zero_nlp常规大模型复现使用](https://github.com/yuanzhoulvpi2017/zero_nlp/blob/main/README.md)
- [自然语言处理（NLP）教程，包括：词向量，词法分析，预训练语言模型，文本分类，文本语义匹配，信息抽取，翻译，对话。](https://github.com/shibing624/nlp-tutorial)
- [JioNLP: 功能大全中文 NLP 预处理、解析工具包，准确、高效、易用 A Chinese NLP Preprocessing & Parsing Package www.jionlp.com](https://github.com/dongrixinyu/JioNLP)
- [JioNLP 工具包：网页自动清洗](https://github.com/dongrixinyu/JioNLP/wiki)

------------------------------------------------------------

## 0.RASA开源机器学习框架，实现基于文本和语音的对话自动化


#### 简介
- 介绍了RasaHQ的rasa_core、rasa-demo项目，以及paulpierre的RasaGPT项目。RasaGPT是首个基于Rasa和Langchain构建的无头大语言模型（LLM）聊天机器人平台，具有多种功能和特性。
#### 核心功能
- **聊天机器人构建**：基于Rasa和Langchain构建聊天机器人平台，可实现文本和语音对话。
- **多渠道支持**：支持Telegram，也可轻松集成Slack、Whatsapp、Line、SMS等。
- **文档处理与训练**：能上传文档并通过FastAPI进行“训练”，实现文档版本控制和自动“重新训练”。
- **自定义端点与数据库模型**：可通过FastAPI和SQLModel自定义异步端点和数据库模型。
- **自动生成标签**：机器人能根据用户问题和回复自动生成标签。
- **API文档与工具**：提供完整的API文档，包含Swagger和Redoc，还集成了PGAdmin用于浏览数据库。
- **Ngrok端点生成**：启动时自动生成Ngrok端点，方便通过Telegram访问机器人。
- **嵌入相似性搜索**：通过pgvector和Postgres函数在Postgres中实现嵌入相似性搜索。
#### 技术原理
- **Rasa**：处理与通信渠道（如Telegram）的集成，通过多个yaml文件进行配置，包括NLU管道、策略、凭证、域、端点、意图、规则和动作等设置。其NLU模型需训练，核心和动作服务器需分别运行。
- **Telegram**：Rasa自动更新Telegram Bot API的回调webhook，借助Ngrok生成可公开访问的URL并反向隧道到docker容器，由rasa-credentials服务处理相关过程，webhook将消息发送到FastAPI服务器，再由其转发到Rasa webhook。
- **PGVector**：是Postgres的插件，用于存储和计算向量数据类型，项目有自己的实现以适应自定义模式。
- **Langchain**：加载训练数据到数据库，进行索引存储，使用LlamaIndex查找相关数据并注入提示，通过提示中的护栏保持对话聚焦。
#### 应用场景
- **客户支持**：用于解答客户问题，提供常见问题解答和技术支持。
- **问答系统**：实现智能问答，根据用户问题提供准确答案。
- **电子学习**：辅助学习过程，提供相关知识和信息。
- **游戏**：如在角色扮演游戏中作为NPC与玩家互动。 


- [RasaHQ/rasa_core: Rasa Core is now part of the Rasa repo: An open source machine learning framework to automate text-and voice-based conversations](https://github.com/RasaHQ/rasa_core)
- [RasaHQ/rasa-demo: :tiger: Sara - the Rasa Demo Bot: An example of a contextual AI assistant built with the open source Rasa Stack](https://github.com/RasaHQ/rasa-demo)
- [paulpierre/RasaGPT: 💬 RasaGPT is the first headless LLM chatbot platform built on top of Rasa and Langchain. Built w/ Rasa, FastAPI, Langchain, LlamaIndex, SQLModel, pgvector, ngrok, telegram](https://github.com/paulpierre/RasaGPT)

------------------------------------------------------------

## 0.RocketQA，用于信息检索和问题回答的密集检索，包括中文和英文的最先进模型。


#### 简介
- PaddlePaddle的RocketQA项目中research文件夹的README文件，介绍了相关研究成果、代码和模型，以及引用信息和许可协议。Facebook Research的fairseq项目，是一个序列建模工具包，提供多种功能和预训练模型。
#### 核心功能
- RocketQA用于开放域问答的密集段落检索研究，提供代码和模型。fairseq用于训练和生成多种文本生成任务的模型。
#### 技术原理
- RocketQA通过优化训练方法进行密集段落检索。fairseq基于PyTorch，支持多GPU训练、多种生成算法等，利用Hydra进行灵活配置。
#### 应用场景
- RocketQA应用于开放域问答系统。fairseq可用于机器翻译、文本摘要、语言建模等自然语言处理任务。 


- [RocketQA/research/README.md at main · PaddlePaddle/RocketQA](https://github.com/PaddlePaddle/RocketQA/blob/main/research/README.md)
- [Facebook AI Research ：fairseq](https://github.com/facebookresearch/fairseq)

------------------------------------------------------------

## 0.haystack开源的NLP框架


#### 简介
- `https://github.com/deepset-ai/haystack` 是一个用于构建可定制、生产就绪的大型语言模型（LLM）应用程序的人工智能编排框架，可连接模型、向量数据库、文件转换器等组件构建问答、语义搜索等应用。
- `https://github.com/django-haystack/django-haystack` 为Django提供模块化搜索，具有统一API，可插入不同搜索后端，支持多高级特性。

#### 核心功能
- **Haystack**：连接组件构建多种LLM应用，支持RAG、问答、语义搜索等，具有技术无关、灵活、可扩展等特性。
- **Django - Haystack**：为Django提供模块化搜索，统一API接入不同搜索后端，支持高级搜索特性。

#### 技术原理
- **Haystack**：通过特定架构和机制实现组件连接与功能编排，利用先进检索方法处理数据与模型交互。
- **Django - Haystack**：基于Django框架，通过特定代码结构和接口实现与不同搜索后端的集成。

#### 应用场景
- **Haystack**：适用于构建RAG应用、问答系统、语义搜索工具、对话代理聊天机器人等LLM相关应用。
- **Django - Haystack**：用于Django项目中实现高效的搜索功能，如文档搜索、信息检索等。 


- [deepset-ai/haystack: :mag: Haystack is an open source NLP framework to interact with your data using Transformer models and LLMs (GPT-4, ChatGPT and alike). Haystack offers production-ready tools to quickly build complex question answering, semantic search, text generation applications, and more.](https://github.com/deepset-ai/haystack)
- [django-haystack/django-haystack: Modular search for Django](https://github.com/django-haystack/django-haystack)

------------------------------------------------------------

## 0.分词


#### 简介
分别指向不同的项目。“dongrixinyu/jiojio”是基于CPU的高性能中文分词器；“shibing624/pinyin-tokenizer”用于将连续拼音切分为单字拼音列表；“shibing624/companynameparser”未获取到具体介绍；“lancopku/pkuseg-python”是多领域中文分词工具包。
#### 核心功能
- jiojio：中文分词，支持词性标注，可添加自定义词典，基于CRF算法优化。
- pinyin-tokenizer：将连续拼音切分为单字拼音列表。
- pkuseg-python：多领域中文分词，支持用户自训练模型，可进行词性标注。
#### 技术原理
- jiojio：基于C语言开发，通过Python接口调用，利用CRF算法及字符特征工程优化模型。
- pinyin-tokenizer：基于前缀树（PyTrie）实现拼音切分。
- pkuseg-python：基于论文[Luo et. al, 2019]，采用领域自适应方法训练不同领域的预训练模型。
#### 应用场景
- jiojio：适用于自然语言处理任务中的中文文本分词，如文本分析、信息检索等。
- pinyin-tokenizer：用于拼音相关处理，如拼音转汉字等。
- pkuseg-python：可应用于新闻、网络、医药、旅游等不同领域的文本分词。 


- [中文分词器](https://github.com/dongrixinyu/jiojio)
- [拼音分词器，将连续的拼音切分为单字拼音列表。](https://github.com/shibing624/pinyin-tokenizer)
- [中文公司名称分词工具，支持公司名称中的地名，品牌名（主词），行业词，公司名后缀提取。](https://github.com/shibing624/companynameparser)
- [lancopku/pkuseg-python: pkuseg多领域中文分词工具; The pkuseg toolkit for multi-domain Chinese word segmentation](https://github.com/lancopku/pkuseg-python)

------------------------------------------------------------

## 1.1 智能标注-主动学习


#### 简介
labelit是一个用于基于主动学习进行文本和图像标注实验的Python模块，涵盖实验脚本、数据集获取、多种主动学习方法及模型，还介绍了主动学习相关概念及应用场景。
#### 核心功能
提供主动学习实验框架，支持多种数据集下载、多种主动学习方法及模型选择，可进行实验参数设置、结果收集与分析。
#### 技术原理
通过不同的采样方法（如Uniform、Margin等）从数据集中选择样本，利用各种模型（如Linear SVM、Kernel SVM等）进行训练和评估，根据设定的指标不断优化模型。
#### 应用场景
适用于文档分类与信息提取、图像检索、入侵检测等需要进行数据标注和模型训练的场景。 


- [自动标注，基于主动学习](https://github.com/shibing624/labelit)
- [labelit/README.md at master · shibing624/labelit](https://github.com/shibing624/labelit/blob/master/README.md)

------------------------------------------------------------

## 1.信息抽取（关键词抽取）


#### 简介
- 这几个GitHub项目主要围绕中文文本处理相关功能展开。包括从中文自然语言文本中抽取关键短语、命名实体识别、地址解析等，部分项目已集成至功能更优的工具包。
#### 核心功能
- **关键短语抽取**：能从中文文本中提取关键短语，可用于生成词云、提供摘要阅读、关键信息检索等。
- **命名实体识别**：实现了多种命名实体识别模型，如BertSoftmax、BertCrf、BertSpan等，并在标准数据集上比较了各模型效果。
- **地址解析**：支持中国三级区划地址（省、市、区）提取和级联映射，还能绘制地址目的地热力图。
#### 技术原理
- **关键短语抽取**：基于北大分词器pkuseg工具进行分词和词性标注，利用tfidf计算关键词权重，通过关键词提取算法找出碎片化关键词，再融合相邻关键碎片词，重新计算权重，去除相似词汇得到关键短语。同时使用预训练好的LDA模型计算文本和候选短语的主题概率分布，得到最终权重。
- **命名实体识别**：如BertSoftmax基于BERT预训练模型实现实体识别，BertSpan基于BERT训练span边界的表示以适配实体边界识别，均基于PyTorch实现训练和预测。
- **地址解析**：利用爬取自国家统计局和中华人民共和国民政局全国行政区划查询平台的中国行政区划地名数据集，通过特定算法实现地址提取和级联映射等功能。
#### 应用场景
- **文本信息处理**：在新闻、文章等文本的关键信息提取、摘要生成、实体识别等场景中发挥作用。
- **数据分析**：对包含地址等信息的文本数据进行分析，如统计不同地区的数据量等。
- **可视化**：根据地址信息绘制热力图、分类散点图等进行可视化展示。 


- [一个快速确定文本（新闻）归属地的工具](https://github.com/dongrixinyu/location_detect)
- [pke_zh, 中文关键词或关键句提取工具。](https://github.com/shibing624/pke_zh)
- [中文里抽取关键短语的工具](https://github.com/dongrixinyu/chinese_keyphrase_extractor)
- [命名实体识别工具](https://github.com/shibing624/nerpy)
- [中文地址提取工具](https://github.com/shibing624/addressparser)

------------------------------------------------------------

## 2.文本分类


#### 简介
- **pytextclassifier**：Python文本分类工具，可用于情感极性分析、文本风险分类等，支持多种分类和聚类算法。
- **NeuralNLP-NeuralClassifier**：未获取到具体有效信息。
- **pysenti**：中文情感极性分析工具，基于规则词典进行情感极性分析，扩展性强。

#### 核心功能
- **pytextclassifier**：实现多种文本分类和聚类算法，支持句子和文档级分类任务，涵盖二分类、多分类、多标签分类、多层级分类及Kmeans聚类等。
- **NeuralNLP-NeuralClassifier**：信息缺失，无法明确核心功能。
- **pysenti**：基于规则词典对文本进行情感极性分析，通过整合多种情感词典，并结合句子结构给情感词语赋予权重，最终加权求和得到文本情感极性得分。

#### 技术原理
- **pytextclassifier**：内部模块低耦合，模型惰性加载，通过多种分类算法如逻辑回归、随机森林、决策树等实现文本分类功能，还支持深度分类模型如FastText、TextCNN、TextRNN、BERT等。
- **NeuralNLP-NeuralClassifier**：信息不足，难以阐述技术原理。
- **pysenti**：基于规则的情感分析方法，先将文本切分、切词，利用情感词标识词语情感极性，再结合句子结构中连词、否定词、副词、标点等给情感词语权重，最后加权求和得文本情感极性得分。

#### 应用场景
- **pytextclassifier**：可应用于情感分析、文本风险分类、新闻分类、行业分类、产品分类等自然语言处理相关场景。
- **NeuralNLP-NeuralClassifier**：因信息缺失，无法确定应用场景。
- **pysenti**：可用于商品评论情感分析、文本调研等需要进行中文情感极性分析的场景。 


- [文本分类，句子和文档级的文本分类任务，支持二分类、多分类、多标签分类、多层级分类和Kmeans聚类](https://github.com/shibing624/pytextclassifier)
- [层次多标签文本分类](https://github.com/shibing624/NeuralNLP-NeuralClassifier)
- [情感分类](https://github.com/shibing624/pysenti)

------------------------------------------------------------

## 3.文本匹配


#### 简介
- 提供了多个与文本处理相关的项目，包括计算文本相似度、文本向量化、图像相似度计算、语义匹配检索等功能，支持多种模型和算法，涉及Java、Python等语言。

#### 核心功能
- **文本相似度计算**：实现了多种文本相似度计算方法，如基于词林编码法、汉语语义法、知网词语相似度、字面编辑距离法等，可用于计算词语、短语、句子、段落的相似度。
- **文本向量化**：把文本表征为向量矩阵，实现了Word2Vec、RankBM25、BERT、Sentence-BERT、CoSENT等多种文本表征模型。
- **图像相似度计算**：支持CLIP、pHash、SIFT等算法的图像相似度计算和匹配搜索，中文CLIP模型支持图搜图，文搜图、还支持中英文图文互搜。
- **语义匹配检索**：基于多种算法实现语义匹配检索，可在文档候选集中找与query最相似的文本，常用于QA场景的问句相似匹配、文本搜索等任务。

#### 技术原理
- **文本相似度计算**：通过各种算法对文本的语义、词形、词序等进行分析和比较，计算出相似度得分。
- **文本向量化**：利用深度学习模型对文本进行编码，将其转换为向量表示，以便于后续的计算和处理。
- **图像相似度计算**：通过对图像的特征提取和匹配，计算出图像之间的相似度。
- **语义匹配检索**：基于语义理解和匹配算法，在文档集中搜索与查询语句最相似的文档。

#### 应用场景
- **信息检索**：在海量文本数据中快速找到与查询相关的信息。
- **文本分类**：根据文本的相似度对其进行分类。
- **问答系统**：匹配相似问题，提供准确答案。
- **图像搜索**：根据图像内容搜索相似图像。
- **情感分析**：分析文本的情感倾向。 


- [！similarities支持亿级数据文搜文语义相似度计算、匹配搜索工具包](https://github.com/shibing624/similarities)
- [text2vec 文本匹配-文本向量表征工具，把文本转化为向量矩阵](https://github.com/shibing624/text2vec)
- [shi文本相似度计算工具包，java编写，可用于文本相似度计算、情感分析等任务，开箱即用。](https://github.com/shibing624/similarity)
- [lyj157175/search-bm25: 使用bm25算法，实现在海量文档集中，利用关键字query搜索到最相似的文档](https://github.com/lyj157175/search-bm25)

------------------------------------------------------------

## 4.文本纠错


#### 简介
介绍了这是一个有用的Python文本纠错工具包，支持多种错误类型纠正，实现了多种模型的文本纠错及效果评估等内容。
#### 核心功能
提供多种模型实现中文文本纠错，包括Kenlm、DeepContext、Seq2Seq、T5、ERNIE_CSC、MacBERT、MuCGECBart、NaSGECBart、GPT等模型，可纠正音似、形似、语法、专名错误等，还能评估各模型效果。
#### 技术原理
基于多种技术实现，如基于Kenlm统计语言模型工具训练中文NGram语言模型结合规则方法和混淆集纠正拼写错误；基于PyTorch实现不同结构的模型如DeepContext、ConvSeq2Seq、T5、MacBERT、GPT等，通过在相应预训练模型上微调中文纠错数据集来适配任务；基于PaddlePaddle实现ERNIE_CSC模型，在ERNIE - 1.0上微调；基于ModelScope实现MuCGECBart模型等。
#### 应用场景
适用于需要进行中文文本纠错的场景，如拼音输入法、语音识别校对、五笔输入法、OCR校对、搜索引擎query纠错等业务场景下的文本纠错。 


- [pycorrector文本纠错](https://github.com/shibing624/pycorrector/blob/master/README.en.md)
- [pycorrector/README.md at master · shibing624/pycorrector](https://github.com/shibing624/pycorrector/blob/master/README.md)

------------------------------------------------------------

## 7.搜索推荐

#### 简介
这是一个用Java语言开发的短语搜索系统，专注于提供高效的短语检索服务，特别适用于公司名称、地址名称等特定短语的搜索场景。

#### 核心功能
*   **短语搜索：** 支持对公司名称、地址名称等结构化或半结构化短语进行精确或模糊搜索。
*   **自定义排序：** 允许用户根据特定需求对搜索结果进行定制化排序。
*   **拼音处理：** 内置拼音处理能力，支持基于中文拼音的搜索匹配。
*   **Web接口服务：** 通过内置的Jetty服务器提供HTTP/Web接口，便于集成和访问。

#### 技术原理
该系统基于Java语言开发，利用其强大的生态系统和跨平台特性。通过集成Jetty嵌入式Web服务器，系统能够提供轻量级的HTTP服务接口，实现客户端与搜索功能的交互。在核心搜索技术上，它可能采用倒排索引、Trie树或哈希表等数据结构对短语进行高效存储和检索。针对中文短语，系统实现了拼音转换和匹配算法，以支持用户通过拼音进行搜索，并能处理多音字及拼音变体。自定义排序功能则可能通过权重计算或外部配置实现。

#### 应用场景
*   **企业内部知识库检索：** 快速查找公司内部文档、项目名称、客户信息等短语。
*   **地址或地点信息查询：** 在地图应用、物流系统或地理信息系统中，高效搜索特定地址或地标。
*   **客户关系管理（CRM）系统：** 帮助销售或客服人员快速检索客户名称、公司名称或联系地址。
*   **行政管理系统：** 用于查找特定机构名称、部门名称或政策条款。
*   **电子商务平台：** 优化商品名称、品牌名称的搜索体验，尤其当用户输入拼音时。


- [phrase-search: 短语搜索，支持公司名称、地址名称等短语的搜索，支持自定义排序、拼音处理，内置jetty提供web接口。java编写。](https://github.com/shibing624/phrase-search)

------------------------------------------------------------

## 8.语音识别和语音合成模型


#### 简介
Parrots是一个语音识别和语音合成工具包，支持中文、英文、日文等多种语言。它基于distilwhisper实现了中文语音识别模型，基于GPT-SoVITS训练了语音合成模型。该工具包提供了多种使用方式，包括命令行模式和Python代码示例，还展示了一些预训练模型的信息。

#### 核心功能
- 实现语音识别（ASR）和语音合成（TTS）功能。
- 支持多种语言，如中文、英文、日文等。
- 提供命令行模式和Python代码示例，方便用户使用。

#### 技术原理
- ASR：基于distilwhisper实现中文语音识别模型。
- TTS：基于GPT-SoVITS训练语音合成模型。

#### 应用场景
- 语音识别：将语音转换为文本，可用于语音助手、语音转写等场景。
- 语音合成：将文本转换为语音，可用于有声读物、语音导航等场景。
- 多语言支持：适用于需要处理多种语言的场景，如跨国交流、多语言内容创作等。
- 命令行模式：方便在终端中快速执行语音识别和合成任务。
- Python代码示例：便于开发者在项目中集成语音功能。 


- [中文语音识别、文字转语音，基于语音库实现，易扩展。](https://github.com/shibing624/parrots)

------------------------------------------------------------


# 1.数据集成


#### 简介
分别介绍了渊亭科技的产品数据集成，以及RestCloud的数据传输平台，涉及数据处理、集成等相关内容。
#### 核心功能
- 渊亭科技提供涵盖决策中台、认知中台、数据中台等多领域产品，具备数据集成、标注探索、认知推理等功能。
- RestCloud的数据传输平台可实现业务系统数据集成与异构数据源数据传输，有数据抽取、转换、清洗、脱敏、加载等功能。
#### 技术原理
- 渊亭科技的产品基于微服务架构自主研发，涉及多种智能技术如认知智能、图数据库等。
- RestCloud的数据传输平台基于微服务架构，通过可视化操作构建数据集成流程，实现多种数据处理功能。
#### 应用场景
- 渊亭科技产品广泛应用于多个行业解决方案，如风险传导分析、反洗钱监管等。
- RestCloud的数据传输平台用于企业内业务系统数据集成及异构数据源间的数据交换。 


- [DataExa-Sentinel 数据集成](http://www.dataexa.com/product/data-integration)
- [RestCloud 新一代ETL数据集成平台](https://www.restcloud.cn/restcloud/mycms/product-etl.html?bd_vid=9759123659395352708)


# 1.文本挖掘和预处理工具


#### 简介
分别指向不同的自然语言处理相关项目，包括Jiagu工具包、FastTextRank算法实现、flair框架、fastNLP工具包以及nlpcda中的9simbert相关内容，涵盖了分词、词性标注、命名实体识别、文本摘要、关键词提取、情感分析等多种自然语言处理功能及相关算法和工具的实现。
#### 核心功能
- Jiagu提供中文分词、词性标注、命名实体识别、情感分析等多种自然语言处理功能。
- FastTextRank可进行快速文本摘要及关键词提取，并对算法时间复杂度进行了优化。
- flair是一个用于自然语言处理的简单框架，可进行命名实体识别、情感分析等任务。
- fastNLP是轻量级自然语言处理工具包，能减少工程型代码，具有便捷、高效、兼容的特性。
#### 技术原理
- Jiagu使用大规模语料训练，参考各大工具优缺点制作。
- FastTextRank通过改进的迭代算法降低计算图最大权节点的时间复杂度，并选择性使用词向量提高准确性。
- flair基于深度学习框架，通过预训练模型和特定算法实现各种自然语言处理任务。
- fastNLP通过重新设计架构实现对不同深度学习框架的兼容，内部包含数据处理、模型搭建、训练等相关模块和功能。
#### 应用场景
- 文本分析：如新闻、文章的内容理解、关键信息提取等。
- 情感分析：对评论、社交媒体等文本进行情感倾向判断。
- 信息抽取：从文本中提取命名实体、关系等信息。
- 自然语言处理相关研究和开发：为研究人员提供工具和基础框架。 


- [ownthink/Jiagu: Jiagu深度学习自然语言处理工具 知识图谱关系抽取 中文分词 词性标注 命名实体识别 情感分析 新词发现 关键词 文本摘要 文本聚类](https://github.com/ownthink/Jiagu)
- [FastTextRank: 中文文本摘要/关键词提取](https://github.com/ArtistScript/FastTextRank)
- [flair自然语言处理（NLP）的一个非常简单的框架](https://github.com/flairNLP/flair)
- [fastNLP是一款轻量级的自然语言处理（NLP）工具包](https://github.com/fastnlp/fastNLP)
- [nlpcda: 一键中文数据增强包 ； NLP数据增强、bert数据增强、EDA：pip install nlpcda](https://github.com/425776024/nlpcda#9simbert)

------------------------------------------------------------

## 0.PyMuPDF解析


#### 简介
内容围绕PyMuPDF展开，包括其在GitHub上的项目结构、安装使用方法、可选功能、相关文档以及许可协议等信息，同时还有一些博客文章从不同角度对其进行了介绍。
#### 核心功能
- 提供高性能Python库用于PDF及其他文档的数据提取、分析、转换与操作。
- 支持文档页面遍历、文本获取等基础操作。
#### 技术原理
- 为MuPDF添加Python绑定和抽象，利用MuPDF的功能实现文档处理。
- 依赖Python 3.9或更高版本，通过pip进行安装，部分可选功能需额外安装包支持。
#### 应用场景
- 文档处理相关领域，如数据提取、文本分析、格式转换等。
- 文字识别（结合Tesseract - OCR）、字体处理（结合fontTools）等特定功能应用场景。 


- [pymupdf/PyMuPDF: PyMuPDF is an enhanced Python binding for MuPDF – a lightweight PDF, XPS, and E-book viewer, renderer, and toolkit.](https://github.com/pymupdf/PyMuPDF)
- [Python处理PDF——PyMuPDF的安装与使用(1)_pip install pymupdf_冰__蓝的博客-CSDN博客](https://blog.csdn.net/ling620/article/details/120035699)
- [Python - PyMuPDF模块的简单使用 - 骑着螞蟻流浪 - 博客园](https://www.cnblogs.com/mayi0312/p/16591719.html)
- [基于pymupdf的PDF的文本、图片和表格信息提取_pymupdf提取表格_wxplol的博客-CSDN博客](https://blog.csdn.net/wxplol/article/details/109304946?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-109304946-blog-126873589.235^v38^pc_relevant_sort_base1&spm=1001.2101.3001.4242.3&utm_relevant_index=7)
- [Python 处理 PDF 的神器 -- PyMuPDF_宋宋讲编程的博客-CSDN博客](https://blog.csdn.net/qiqi1220/article/details/126873589)

------------------------------------------------------------

## 1.HarvestText:

#### 简介
HarvestText 是一个专注于文本挖掘与预处理的开源工具包，旨在为用户提供一套高效且灵活的文本处理解决方案。它尤其强调采用无监督或弱监督方法，降低对大规模标注数据的依赖，使得文本清洗、信息抽取和分析变得更加便捷。

#### 核心功能
*   **文本清洗与预处理:** 处理包括URL、email、微博等特定格式文本，进行字符规范化和标点符号去除。
*   **新词发现:** 自动识别并发现文本中未登录词或新出现的词汇。
*   **情感分析:** 评估文本的情感倾向性。
*   **命名实体识别 (NER):** 识别文本中的人名、地名、机构名等各类命名实体。
*   **实体链接:** 将识别出的实体别名或缩写与标准实体名称进行关联。
*   **关键词抽取:** 从文本中提取最具代表性的关键词。
*   **知识抽取:** 从非结构化文本中抽取出结构化的知识。
*   **句法分析:** 对句子的语法结构进行解析。
*   **精细分词分句:** 支持自定义分词规则，并能准确处理省略号、双引号等复杂标点的分句。

#### 技术原理
HarvestText 的核心技术理念在于其对**无监督 (Unsupervised)** 和 **弱监督 (Weakly Supervised)** 学习方法的运用。这意味着该工具包在执行文本清洗、新词发现、情感分析、实体识别和链接等任务时，不依赖于大量人工标注的数据集，而是可能通过以下方式实现：
*   **统计特征:** 例如，新词发现可能基于词语的凝固度、左右熵等统计指标。
*   **启发式规则:** 文本清洗和部分实体识别可能通过预定义的模式匹配和规则集实现。
*   **少量种子数据/外部知识:** 弱监督方法可能利用少量的初始标注数据或现有的知识库进行模型训练或引导。
*   **基于图的算法:** 实体链接可能涉及到构建实体关系图并进行推理。

#### 应用场景
*   **自然语言理解 (NLU) 数据预处理:** 作为下游NLP任务（如文本分类、机器翻译、问答系统）的数据清洗和特征工程前置工具。
*   **舆情监控与分析:** 对社交媒体、新闻评论等进行实时情感分析、热点追踪和关键人物识别。
*   **知识图谱构建:** 从海量非结构化文本中自动化抽取实体、关系和事件，为构建领域知识图谱提供基础数据。
*   **信息检索与推荐系统:** 提升搜索结果的相关性，通过关键词和实体分析优化用户推荐。
*   **学术研究与数据分析:** 为研究人员提供一套便捷的文本分析工具，支持语言学、社会科学等领域的数据挖掘。

- [HarvestText: 文本挖掘和预处理工具（文本清洗、新词发现、情感分析、实体识别链接、关键词抽取、知识抽取、句法分析等），无监督或弱监督方法](https://github.com/blmoistawinde/HarvestText)

------------------------------------------------------------

## 1.funNLP清华


#### 简介
该项目是一个几乎最全的中文NLP资源库，涵盖类ChatGPT的模型评测对比、资料、开源框架，LLM的训练、推理、低资源、高效训练，提示工程，文档问答，行业应用，课程资料，安全问题，多模态LLM，数据集，语料库，词库及词法工具，预训练语言模型，抽取，知识图谱，文本生成，文本摘要，智能问答，文本纠错，文档处理，表格处理，文本匹配，文本数据增强，文本检索，阅读理解，情感分析，事件抽取，机器翻译，数字转换，指代消解，文本聚类，文本分类，知识推理，可解释NLP，文本对抗攻击，文本可视化，文本标注工具，综合工具，有趣搞笑工具，课程报告面试等，比赛，金融NLP，医疗NLP，法律NLP，文本生成图像等方面的内容。

#### 核心功能
- 提供丰富的NLP资源，包括模型、工具、数据集等。
- 支持多种NLP任务，如文本生成、摘要、问答、分类等。
- 涵盖类ChatGPT相关的各种资源，如模型评测、开源框架等。

#### 技术原理
- 整合了众多开源的NLP技术和工具，如Transformer架构、预训练语言模型等。
- 通过收集和整理大量的NLP相关资源，为用户提供一站式的NLP资源平台。

#### 应用场景
- NLP研究：为研究人员提供丰富的资源和工具，促进NLP技术的发展。
- 自然语言处理应用开发：帮助开发者快速搭建和测试NLP应用。
- 教育和学习：作为NLP学习的资料和实践平台，辅助教学和学习。 


- [funNLP: NLP项目合集](https://github.com/fighting41love/funNLP)

------------------------------------------------------------

## AllenNLP


#### 简介
这是AllenNLP的GitHub仓库页面，介绍了该自然语言处理研究库，它基于PyTorch构建，用于开发多种语言任务的深度学习模型，目前处于维护模式，并提供了相关替代方案、安装指南、使用说明等内容。
#### 核心功能
- 提供自然语言处理工具和模型开发库，支持多种语言任务。
- 包含数据处理、模型训练、评估等功能模块。
#### 技术原理
基于PyTorch构建，通过定义各种模块和函数来实现自然语言处理任务，利用深度学习算法进行模型训练和优化。
#### 应用场景
- 自然语言处理相关的研究和开发。
- 各类语言任务的模型构建与训练，如文本分类、命名实体识别等。

#### 简介
该仓库是AllenNLP的模型库，包含了用于多种NLP任务的组件和预训练模型，介绍了任务类型、模型列表以及安装方式等内容。
#### 核心功能
- 提供多种自然语言处理任务的组件，如数据集读取器、模型和预测器等。
- 包含多个预训练模型，可用于各种NLP任务。
#### 技术原理
基于PyTorch构建，通过定义不同的类和模块来实现各种自然语言处理任务，利用深度学习技术进行模型训练和优化。
#### 应用场景
- 自然语言处理任务的模型开发与应用，如文本分类、情感分析、命名实体识别等。
- 利用预训练模型进行快速的任务求解和效果提升。

#### 简介
该仓库是AllenNLP演示项目的代码库，是一个展示AllenNLP功能的Web应用程序，现已不再维护，并介绍了其使用方法。
#### 核心功能
- 提供AllenNLP演示应用，展示该库在自然语言处理任务中的能力。
#### 技术原理
通过Docker和Python 3运行，利用相关命令启动本地实例、代理API请求等，实现对AllenNLP功能的展示。
#### 应用场景
- 直观展示AllenNLP库在各种自然语言处理任务上的应用效果。

#### 简介
该仓库是AllenNLP指南的代码库，涵盖贡献指南、运行应用、格式化、移动开发、依赖项等内容，介绍了参与方式及项目相关技术细节。
#### 核心功能
- 提供AllenNLP指南的相关代码和资料，方便用户学习和了解AllenNLP。
#### 技术原理
基于Gatsby构建前端，通过特定命令安装依赖并运行开发服务器，利用相关工具处理代码格式化、章节组织等。
#### 应用场景
- 作为学习AllenNLP的教程和参考文档，帮助开发者更好地使用该库。 


- [allenai/allennlp: An open-source NLP research library, built on PyTorch.](https://github.com/allenai/allennlp)
- [allenai/allennlp-models: Officially supported AllenNLP models](https://github.com/allenai/allennlp-models)
- [allenai/allennlp-demo: Code for the AllenNLP demo.](https://github.com/allenai/allennlp-demo)
- [allenai/allennlp-guide: Code and material for the AllenNLP Guide](https://github.com/allenai/allennlp-guide)

------------------------------------------------------------

## HanLP


#### 简介
该文章主要介绍了HanLP工具包，它支持多种语言的联合任务，提供了RESTful和native两种API，还介绍了安装方法和使用示例，包括查询预训练模型、选择子任务功能以及可视化输出等内容，并与其他工具的依存分析进行了对比。
#### 核心功能
HanLP是一个功能强大的自然语言处理工具包，支持多语言多任务处理，提供多种词性标注规范、命名实体识别规范、依存句法分析规范等，具备查询预训练模型、执行多种自然语言处理任务、选择子任务功能以及可视化输出等能力。
#### 技术原理
HanLP基于先进的自然语言处理算法和模型，如预训练模型技术，通过对大量文本数据的学习，能够准确地进行分词、词性标注、命名实体识别、依存句法分析、语义依存分析等任务。它采用了多种深度学习技术，如神经网络、循环神经网络等，以实现对自然语言的高效处理和理解。
#### 应用场景
- 文本分析：对各种文本进行分词、词性标注、命名实体识别、依存句法分析等，帮助理解文本结构和语义。
- 信息检索：用于提高信息检索的准确性和效率，通过对查询词和文档进行自然语言处理，更好地匹配用户需求。
- 机器翻译：辅助机器翻译系统进行语言理解和转换，提升翻译质量。
- 智能问答：为智能问答系统提供自然语言处理能力，准确理解用户问题并给出回答。 


- [HanLP的依存分析_Dawn_www的博客-CSDN博客_hanlp 依存句法分析](https://blog.csdn.net/sinat_36226553/article/details/116230574)

------------------------------------------------------------

## NLTK


#### 简介
这篇文章主要介绍了NLTK（Natural Language Toolkit）这个自然语言处理工具包，包括其能干的事情、设计目标、包含的语料库，以及分词、处理切词、词汇规范化、词性标注、获取近义词等模块及功能，还给出了一些示例代码。
#### 核心功能
NLTK是用于自然语言处理的Python库，提供了分词、词性标注、命名实体识别、句法分析等功能，可进行文本处理、分析语言结构等操作。
#### 技术原理
- **分词**：包括句子切分和单词切分，将文本段落分解为较小实体。
- **处理切词**：移除标点符号和停用词，并进行词汇规范化。
- **词汇规范化**：有词形还原和词干提取两种方法，分别根据词性获取词根和删除词缀返回词干。
- **词性标注**：识别给定单词的语法组并分配相应标签。
- **获取近义词**：通过synsets()方法查找指定词性的同义词集。
#### 应用场景
- 文本分析：如搜索文本、单词搜索、相似词搜索等。
- 自然语言处理任务：包括分词、词性标注、命名实体识别、句法分析等。 


- [NLTK简介及使用示例](https://blog.csdn.net/justlpf/article/details/121707391)

------------------------------------------------------------

## Synonyms-近义词


#### 简介
围绕自然语言处理中的中文近义词工具包及相关知识展开。介绍了Synonyms工具包的功能、使用方法、环境变量配置等，还涉及到OpenHowNet这个由清华大学自然语言处理实验室开发的义原知识库，包括其核心数据、基本功能、高级功能等内容。
#### 核心功能
- **Synonyms工具包**：支持文本对齐、推荐算法、相似度计算、语义偏移、关键字提取、概念提取、自动摘要、搜索引擎等自然语言理解任务。提供分词、获取词向量、查询近义词、比较句子相似度等功能。
- **OpenHowNet**：提供义原信息查询、义原树展示、基于义原的词相似度计算等功能。可获取HowNet中词语对应的概念、所有词语和义原、词语的义原标注，查询义原之间的关系等。
#### 技术原理
- **Synonyms工具包**：利用词向量模型和算法实现近义词查找、相似度计算等功能。通过加载预训练的词向量文件，对输入的词语进行匹配和计算，得出相应结果。
- **OpenHowNet**：基于知网（HowNet）构建，知网通过预定义的义原为词语所表示的概念进行标注。OpenHowNet在此基础上，开发了相关API，利用义原之间的关系和算法，实现各种查询和计算功能。
#### 应用场景
- **自然语言处理任务**：如文本对齐、推荐算法、自动摘要等。
- **智能问答系统**：帮助机器人更准确理解用户问题，提供更合适的回答。
- **信息检索**：提高检索的准确性和相关性，更好地满足用户需求。 


- [Synonyms 中文近义词工具包 -- 支持文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等_小金子的夏天的博客-CSDN博客_synonyms包](https://blog.csdn.net/WangYouJin321/article/details/123381626?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2~default~ESLANDING~default-2-123381626-blog-111469041.pc_relevant_landingrelevant&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~ESLANDING~default-2-123381626-blog-111469041.pc_relevant_landingrelevant&utm_relevant_index=5)
- [chatopera/Synonyms: :herb: 中文近义词：聊天机器人，智能问答工具包](https://github.com/chatopera/Synonyms)
- [Synonyms 中文近义词工具包 -- 支持文本对齐，推荐算法，相似度计算，语义偏移，关键字提取，概念提取，自动摘要，搜索引擎等_synonym关键字_星河_赵梓宇的博客-CSDN博客](https://blog.csdn.net/Aria_Miazzy/article/details/104812738?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-104812738-blog-107226349.235%5Ev38%5Epc_relevant_sort_base1&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~Rate-1-104812738-blog-107226349.235%5Ev38%5Epc_relevant_sort_base1&utm_relevant_index=2)
- [OpenHowNet/README_ZH.md at master · thunlp/OpenHowNet](https://github.com/thunlp/OpenHowNet/blob/master/README_ZH.md)
- [近义词抽取开源工具 - 知乎](https://zhuanlan.zhihu.com/p/500683549)
- [OpenHowNet基于义原知网近义词](https://openhownet.thunlp.org/)
- [OpenHowNet](https://openhownet.thunlp.org/search?keyword=%E6%80%A7%E5%88%AB)

------------------------------------------------------------

## spacy

#### 核心功能
- 提供Spacy相关模型的标签信息，展示不同版本模型及其特点。同时详细讲解Spacy进行依存分析的相关知识，包括其架构组成、模型选用、分词器、标注附录、操作方法以及可视化方式等。
#### 技术原理
- 利用自然语言处理技术，通过构建如Doc、Span、Token、Vocab等类来处理文本。Doc由Tokenizer构造后经管道组件修改，每个组件按顺序处理Doc。分词器将文本转为Doc，词汇表通过Lexeme对象和StringStore对象表示。模型基于不同算法和数据训练，以实现词性标注、依存分析等功能。
#### 应用场景
- 可用于自然语言处理任务，如文本分析、信息提取、问答系统构建、知识图谱搭建等。例如在处理中文文本时，根据不同需求选择合适的Spacy中文模型进行词性标注、依存句法分析等操作。 


- [各种库安装spacy-models](https://github.com/explosion/spacy-models/tags)
- [Spacy的依存分析_Dawn_www的博客-CSDN博客_spacy 依存句法分析](https://blog.csdn.net/sinat_36226553/article/details/115165857)
- [在Jupyter Notebook中使用spaCy可视化中英文依存句法分析结果 - 知乎](https://zhuanlan.zhihu.com/p/405071894)

------------------------------------------------------------

## 哈工大LTP


#### 简介
- 介绍了哈工大LTP平台的语义依存分析功能，包括模型安装配置、常见错误解决办法，对比了语义依存分析与句法依存分析的区别，探讨了语义依存图相较于传统依存树的优势。
#### 核心功能
- 提供自然语言处理技术，如中文分词、词性标注、命名实体识别、依存句法分析、语义角色标注等，重点介绍语义依存分析功能。
#### 技术原理
- 基于转移的方法，通过修改现有转移系统中的转移动作来直接生成语义依存图，突破传统基于转移的依存分析算法得到依存树的限制。
#### 应用场景
- 自然语言处理领域，可用于中文语义理解、回答问题等，帮助分析句子中实词之间的语义关系。 


- [哈工大LTP的依存分析_Dawn_www的博客-CSDN博客_ltp 依存句法分析](https://blog.csdn.net/sinat_36226553/article/details/115045943)
- [HIT-SCIR/ltp: Language Technology Platform](https://github.com/HIT-SCIR/ltp)
- [语言技术平台（ Language Technology Plantform | LTP ）](http://ltp.ai/index.html)

------------------------------------------------------------

## 百度DDParser


#### 简介
该文章主要介绍了百度DDParser的依存分析，包括安装过程中遇到的问题及解决方法，如版本不兼容导致的报错处理，还介绍了其操作方法、结果可视化方式、扩展功能以及与其他工具的对比等内容。
#### 核心功能
百度DDParser用于进行中文依存句法分析，支持使用其他工具的分词结果，还可通过指定参数实现不同功能扩展，如输出概率、词性标签等，并且能对分析结果进行可视化展示。
#### 技术原理
通过调用百度的相关库和工具，如paddlepaddle飞桨、DDParser等，利用其内部的算法和模型对输入的文本进行依存句法分析。在遇到版本不兼容等问题时，通过修改相关代码文件中的版本判断等逻辑来解决。
#### 应用场景
可应用于自然语言处理领域，如文本分析、信息抽取、语义理解等任务，帮助理解文本中词语之间的依存关系，为进一步的语言处理提供基础。 


- [百度DDParser的依存分析_Dawn_www的博客-CSDN博客_ddparser](https://blog.csdn.net/sinat_36226553/article/details/115370888)

------------------------------------------------------------


# 4.深度学习


#### 简介
该仓库提供了一系列深度学习示例，涵盖多种框架，如PyTorch、TensorFlow等，涉及计算机视觉、自然语言处理、推荐系统等多个领域，展示了如何利用NVIDIA CUDA-X软件栈在不同GPU上实现高效训练和部署，以达到最佳的可重复性精度和性能。
#### 核心功能
提供多种深度学习模型示例，支持不同框架，具备多GPU、多节点训练能力，可进行自动混合精度训练，能将模型导出为多种格式，并在NGC容器注册表中提供每月更新的Docker容器。
#### 技术原理
利用NVIDIA CUDA-X软件栈，结合各深度学习框架的特性，如PyTorch的JIT和TorchScript、TensorFlow的XLA等，实现模型的高效训练和部署。通过自动混合精度（AMP）技术，在支持的GPU架构上自动进行混合精度训练，利用TensorFloat-32（TF32）提升矩阵运算速度。
#### 应用场景
1. **计算机视觉领域**：用于图像分类、目标检测、语义分割等任务。
2. **自然语言处理领域**：可进行文本分类、机器翻译、问答系统等。
3. **推荐系统领域**：实现基于深度学习的推荐算法。
4. **语音处理领域**：包括语音识别和文本转语音等应用。 


- [NVIDIA/DeepLearningExamples: State-of-the-Art Deep Learning scripts organized by models - easy to train and deploy with reproducible accuracy and performance on enterprise-grade infrastructure.](https://github.com/NVIDIA/DeepLearningExamples)
- [【XGBoost 多分类】XGBoost解决多分类问题-CSDN博客](https://blog.csdn.net/u013421629/article/details/104952532?ops_request_misc=%257B%2522request%255Fid%2522%253A%252211E0167E-5438-4A28-95A4-B2ACAF5AB6F5%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=11E0167E-5438-4A28-95A4-B2ACAF5AB6F5&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-104952532-null-null.142^v100^pc_search_result_base9&utm_term=XGBoost%E5%A4%9A%E5%88%86%E7%B1%BB&spm=1018.2226.3001.4187)
- [LSTM-paddle文本分类](https://blog.csdn.net/Jakson_poor/article/details/113742920?spm=1001.2014.3001.5501)
- [“花朵分类“ 手把手搭建【卷积神经网络】_黎国溥-CSDN博客](https://blog.csdn.net/qq_41204464/article/details/116567051)
- [ResNet50网络结构_lsh呵呵-CSDN博客_resnet50网络结构](https://blog.csdn.net/nima1994/article/details/82686132)
- [ResNet50网络结构图及结构详解 - 知乎](https://zhuanlan.zhihu.com/p/353235794)
- [基于LSTM的沪深股票价格预测_qq_45528306的博客-CSDN博客](https://blog.csdn.net/qq_45528306/article/details/116722660)
- [RNN LSTM GRU 讲解_木东的博客-CSDN博客_gru lstm](https://blog.csdn.net/u010159842/article/details/113125279?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2~aggregatepage~first_rank_v2~rank_aggregation-7-113125279.pc_agg_rank_aggregation&utm_term=lstm%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BC%98%E7%BC%BA%E7%82%B9&spm=1000.2123.3001.4430)
- [LSTM入门例子：根据前9年的数据预测后3年的客流（PyTorch实现） - 知乎](https://zhuanlan.zhihu.com/p/94757947)
- [【时空序列预测第六篇】时空序列预测模型之EIDETIC 3D LSTM（结合3DConv与RNN，E3D-LSTM） - 知乎](https://zhuanlan.zhihu.com/p/111800002)
- [【时空序列预测第七篇】时空序列预测模型之GAN+LSTM - 知乎](https://zhuanlan.zhihu.com/p/118347562)

------------------------------------------------------------

## 飞桨lstm例子


#### 简介
主要围绕数据预测展开，涉及使用LSTM进行未来数据预测以及光伏数据预测等项目，包含数据处理、模型搭建、训练及结果展示等内容。
#### 核心功能
利用LSTM模型对未来数据进行预测，特别是针对光伏相关数据，通过数据预处理、模型构建、训练优化等步骤，实现对光伏功率等数据的准确预测。
#### 技术原理
运用LSTM（长短期记忆网络）技术，它能够处理和预测时间序列数据。通过对历史数据的学习，捕捉数据中的长期依赖关系。在光伏预测项目中，先对光伏相关的多变量数据进行预处理，包括数据清洗、归一化等操作，然后构建LSTM模型，将预处理后的数据输入模型进行训练，调整模型参数以最小化损失函数，从而得到最优的预测模型。
#### 应用场景
可应用于能源领域，如光伏电站的功率预测，帮助电站更好地规划发电、调度电力；也可用于其他涉及时间序列数据预测的场景，如经济趋势预测、气象数据预测等，为相关决策提供数据支持和依据。 


- [2020建模国赛C题初尝试——基于PaddlePaddle的LSTM - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/906384?channelType=0&channel=0)
- [使用LSTM完成负荷预测 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/2193892?channelType=0&channel=0)
- [光伏预测-基于paddle的multivariate-LSTM20210705 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/2209302?channelType=0&channel=0)
- [多变量多步LSTM实现光伏预测 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/2201431?channelType=0&channel=0)
- [基于PaddlePaddle2.0.0rc使用LSTM进行北京空气污染序列预测 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/2249207?channelType=0&channel=0)
- [用PaddlePaddle实现LSTM股票预测 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/127560?channelType=0&channel=0)
- [『NLP经典项目集』11：使用PaddleNLP预测新冠疫情病例数 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/1290873?channelType=0&channel=0)
- [“中国软件杯”大学生软件设计大赛二等奖开源代码 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/1188761?channelType=0&channel=0)
- [LSTM-京津冀城市PM2.5预测 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/projectdetail/1678970?channelType=0&channel=0)

------------------------------------------------------------


# 4.数学建模


#### 简介
- 中国研究生智慧城市技术与创意设计大赛官网，介绍了第十一届大赛的参赛信息、赛程安排等。数学建模网，包含研究生数学建模竞赛等相关资讯。是研究生竞赛论坛，有2024年研究生数学建模竞赛各题讨论区等内容。
#### 核心功能
- 提供研究生相关竞赛的信息发布、交流讨论平台，涵盖智慧城市技术与创意设计大赛及数学建模竞赛等。
#### 技术原理
- 基于网络平台技术，实现信息的展示与交互，通过网页界面呈现赛事详情、题目讨论等内容，方便用户浏览与参与交流。
#### 应用场景
- 研究生参加智慧城市技术与创意设计大赛、数学建模竞赛等相关赛事时获取信息、交流经验、解决疑问；高校教师和赛事组织者发布赛事通知、组织赛事活动等。 


- [中国研究生创新实践系列大赛管理平台](https://cpipc.acge.org.cn/cw/hp/1)
- [数学建模网—SHUMO.COM](https://www.shumo.com/home/)
- [研究生竞赛 - 数模论坛 - Powered by Discuz!](https://shumo.com/forum/forum.php?mod=forumdisplay&fid=113)
- [数据分析案例—共享单车影响因素分析 - 知乎](https://zhuanlan.zhihu.com/p/37966941)

------------------------------------------------------------

## 建模分析

- [Python使用DataFrame时减少内存使用的一个函数，亲测效果明显_无有-散人_新浪博客](http://blog.sina.com.cn/s/blog_727a8da80102xf0i.html)
- [Pandas读取并修改excel_Debris丶的博客-CSDN博客_pandas修改excel中某一个数据](https://blog.csdn.net/qq_34377830/article/details/81872568)
- [数据可视化实例（八）： 边缘直方图（matplotlib，pandas） - 秋华 - 博客园](https://www.cnblogs.com/qiu-hua/p/12877009.html)
- [边缘直方图](https://www.cnblogs.com/waws1314/p/12825203.html)
- [贝叶斯优化调参实战（随机森林，lgbm波士顿房价）_三年研究生能改变多少-CSDN博客_lgbmregressor参数](https://blog.csdn.net/ssswill/article/details/86564056?utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~default-1.no_search_link)
- [lightgbm回归模型使用方法（lgbm.LGBMRegressor）](https://blog.csdn.net/weixin_42813521/article/details/119054445)
- [numpy中的ndarray与pandas中的series、dataframe的转换 - 西瓜草莓甘蔗 - 博客园](https://www.cnblogs.com/tongtong123/p/10621184.html)
- [【Python-ML】SKlearn库L1正则化特征选择_医疗影像检索](https://blog.csdn.net/fjssharpsword/article/details/79085107)
- [使用ARMA做时间序列预测全流程（附MATLAB代码，ARIMA法） - 知乎](https://zhuanlan.zhihu.com/p/69630638)
- [2021年研究生数学建模竞赛-B题讨论区 - 数模论坛 - Powered by Discuz!](https://www.shumo.com/forum/forum.php?mod=forumdisplay&fid=144)

------------------------------------------------------------

## 数学建模


#### 简介
这几篇文章主要围绕数据分析与处理技术展开，涵盖SPSS中的主成分分析和因子分析、随机森林算法、多层感知器神经网络以及Origin绘制热力图等内容，介绍了这些技术的原理、操作过程及应用场景。
#### 核心功能
- **主成分分析和因子分析**：用于解决变量间多重共线性问题，前者通过线性变换减少变量，后者探讨变量内在结构。
- **随机森林算法**：在IBM SPSS Modeler中构建多棵树，通过Bagging技术随机选择样本数据和输入指标，最终综合投票得到预测结果。
- **多层感知器神经网络**：由输入层、输出层和隐藏层组成，可实现非线性分类，解决复杂问题。
- **Origin绘制热力图**：可将特定格式的数据绘制成热力图，用于直观展示数据特征。
#### 技术原理
- **主成分分析**：利用降维思想，通过线性变换将多个指标转化为不相关的综合指标，保留原始变量大部分信息。
- **因子分析**：从原始变量相关矩阵内部依赖关系出发，把变量表示成公共因子和特殊因子的线性组合。
- **随机森林算法**：构建每棵树时使用C&RT算法，通过Bagging技术随机选择样本数据和输入指标，最终综合投票得到预测结果。
- **多层感知器神经网络**：由多个感知器组合，每个神经元接收输入信号，通过累加器和激励函数计算输出信号，信号在网络中传递。
#### 应用场景
- **主成分分析和因子分析**：可用于多指标综合评价、解决多重共线性问题、探索变量潜在结构等。
- **随机森林算法**：可用于预测、分类等数据分析任务。
- **多层感知器神经网络**：广泛应用于人工智能领域，如模式识别、数据分类、回归分析等。
- **Origin绘制热力图**：常用于展示数据的分布和相关性，在科研论文、数据分析报告中广泛应用。 


- [SPSS信息浓缩技术--主成分分析、因子分析（图文+数据集）_可乐联盟-CSDN博客](https://blog.csdn.net/luyi_weilin/article/details/90452437?utm_medium=distribute.pc_relevant.none-task-blog-title-7&spm=1001.2101.3001.4242)
- [主成分分析和因子分析区别与联系_taojiea1014的博客-CSDN博客](https://blog.csdn.net/taojiea1014/article/details/79683826)
- [因子分析在SPSS中的操作过程及结果解读_GIS小学生的博客-CSDN博客](https://blog.csdn.net/sinat_36744986/article/details/86477963)
- [IBM SPSS Modeler随机森林算法介绍_数控小J 对大数据的探索与见解-CSDN博客](https://blog.csdn.net/chenjunji123456/article/details/53257045)
- [神经网络 - RBF神经网络与BP网络优缺点比较 - 机器学习基础知识_Not Found黄小包-CSDN博客](https://blog.csdn.net/weixin_40683253/article/details/80989682?utm_medium=distribute.pc_relevant.none-task-blog-title-4&spm=1001.2101.3001.4242)
- [MLP（多层感知器）神经网络_liuyukuan的专栏-CSDN博客](https://blog.csdn.net/liuyukuan/article/details/72934383)
- [Origin绘制热力图_qy20115549的博客-CSDN博客](https://blog.csdn.net/qy20115549/article/details/89156149?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242)

------------------------------------------------------------

## 特征筛选方法


#### 简介
主要围绕特征工程展开，介绍了特征工程的重要性、子问题、处理方法、选择方式、评估维度以及降维方法等内容，帮助读者全面了解特征工程这一概念及其在机器学习中的应用。
#### 核心功能
- 对原始数据进行清洗，排除脏数据，如过滤异常数据、检测异常点。
- 对不同类型特征进行处理，包括连续型特征的归一化、离散化，离散型特征的独热编码，时间型特征和文本型特征的特定处理方式，以及生成统计型特征、组合特征等。
- 从特征发散性和与目标相关性两方面选择有意义的特征，减少冗余和噪声。
- 对特征进行评估，考量特征自身质量和与目标值的相关性。
- 降低特征矩阵维度，如使用PCA和LDA等方法。
#### 技术原理
- **特征清洗**：结合业务过滤数据，采用偏差检测、基于统计、基于距离、基于密度等异常点检测算法。
- **特征处理**
    - **连续型特征**：线性归一化适用于数值集中情况，标准化归一化假设数据符合标准正态分布，非线性归一化通过数学函数映射数据；离散化采用等频、等距、树模型离散等方法。
    - **离散型特征**：常用独热编码（one - hot）方式转换为二元属性。
    - **时间型特征**：可看作连续型或离散型进行处理。
    - **文本型特征**：词袋模型去掉停用词后组成稀疏向量，在此基础上可加入ngram扩充、tfidf加权；word2vec将字词转为稠密向量。
    - **统计型特征**：计算当前样本集的均值、最大值、分位数等统计信息。
    - **组合特征**：简单特征组合如拼接型，模型特征组合如GBDT + LR方式。
    - **缺失值处理**：数值型特征用均值或中位数替换，通过关系链加权打分，或设默认值；冷启动问题通过协同过滤或网络模型解决。
- **特征选择**
    - **Filter过滤法**：通过卡方检验、互信息、皮尔森相关系数、最大信息系数等衡量单个特征与目标变量关联。
    - **Wrapper包装法**：根据目标函数每次选择或排除若干特征，如递归消除特征法、RFECV。
    - **Embedded嵌入法**：利用机器学习算法训练得到特征权值系数，按系数选择特征，如基于惩罚项的特征选择法、基于树模型的特征选择法。
- **特征评估**：从特征覆盖度、准确性、方差以及与目标值的相关系数、单特征AUC等维度衡量。
- **降维**：PCA是无监督降维方法，使映射后的样本具有最大发散性；LDA是有监督降维方法，让映射后的样本有最好分类性能。
#### 应用场景
- 机器学习领域，为模型提供优质特征，提升模型性能。
- 数据挖掘、数据分析工作中，处理原始数据，提取有效信息。
- 自然语言处理中，对文本型特征进行处理和分析。 


- [特征工程(Feature Engineering) - Allegro - 博客园](https://www.cnblogs.com/kukri/p/8566287.html)
- [机器学习中，有哪些特征选择的工程方法？ - 知乎](https://www.zhihu.com/question/28641663/answer/1223569051)
- [如何进行高维变量筛选和特征选择(一)？Lasso回归 - 知乎](https://zhuanlan.zhihu.com/p/161470286)
- [如何进行变量筛选和特征选择(二)？最优子集回归 - 知乎](https://zhuanlan.zhihu.com/p/161474117)
- [如何进行变量筛选和特征选择(三)？交叉验证 - 知乎](https://zhuanlan.zhihu.com/p/161476705)

------------------------------------------------------------

**[⬆ 返回README目录](../README.md#目录)**
**[⬆ Back to Contents](../README-EN.md#contents)**