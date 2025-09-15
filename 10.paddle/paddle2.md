# paddle2


# 0.FleetX 飞桨分布式


#### 简介
PaddleFleetX是百度飞桨（PaddlePaddle）深度学习框架开发的，专注于大规模模型训练的套件。它旨在为用户提供高性能、易用的全流程应用能力，涵盖大模型从开发、训练、精调、压推、推理到部署的端到端优化。FleetX是PaddlePaddle分布式训练的高级扩展包，支持参数服务器（Parameter Server）和Reduce模式等主流分布式训练架构，并提供丰富的并行能力，如数据并行、模型并行和流水线并行。

#### 核心功能
*   **大模型全流程支持：** 提供从模型开发、训练、精调到推理部署的完整解决方案，尤其针对大语言模型、跨模态大模型和生物计算大模型等领域。
*   **分布式训练：** 支持多种分布式训练架构，包括参数服务器（Parameter Server）和基于Reduce模式的训练，旨在提升大规模模型训练效率。
*   **并行策略：** 具备数据并行、模型并行和流水线并行等全面的并行能力，以优化训练性能。
*   **动静统一开发模式：** 基于飞桨的动静统一开发模式，全面采用动态图进行开发，并通过Generate API实现算子融合，兼顾调试便利性与静态图的性能。
*   **统一训练器（Trainer）：** 提供全场景统一的Trainer，可灵活配置4D混合并行策略，适用于预训练和精调阶段。
*   **分布式评估：** 在参数服务器训练模式下，支持分布式评估功能，通过`collective.evaluate`接口进行模型评估。
*   **启动工具：** 提供`paddle.distributed.launch`工具，支持通过单条命令启动多进程分布式训练脚本，自动设置环境变量。

#### 技术原理
PaddleFleetX基于飞桨深度学习框架，其核心技术原理主要体现在以下几个方面：
*   **分布式计算架构：** 采用Parameter Server和Collective (Reduce) 两种主流分布式训练架构。Parameter Server模式通过参数服务器管理和同步模型参数，而Collective模式则依赖通信原语（如AllReduce）实现多设备间的数据和梯度同步。
*   **混合并行策略（4D混合并行）：** 结合数据并行（Data Parallelism）、模型并行（Model Parallelism）、流水线并行（Pipeline Parallelism）以及Megatron-LM的张量并行（Tensor Parallelism）等多种并行技术，以适应超大规模模型的训练需求，有效解决单设备内存和计算能力的限制。
*   **动态图与静态图融合：** 利用飞桨的动静统一机制，在动态图模式下开发，同时通过底层优化和算子融合技术，在执行时获得接近静态图的性能优势，提升开发效率和调试便利性。
*   **高性能通信机制：** 底层依赖高效的通信原语，确保分布式训练过程中节点间的数据传输和同步具备低延迟和高吞吐量，例如使用NCCL等库进行多卡通信。
*   **优化器与调度策略：** 对数据读取、混合精度计算、高性能算子库以及并行策略自动寻优、流水线调度等全流程进行优化，旨在最大化计算资源利用率并加速模型收敛。

#### 应用场景
*   **大规模预训练模型开发：** 适用于训练千亿甚至万亿参数量的大语言模型（如GPT系列）、跨模态大模型以及生物计算大模型等。
*   **分布式深度学习研究：** 为研究人员提供一套完善的工具链，用于探索和优化大规模分布式训练算法和策略。
*   **企业级AI应用开发：** 在自然语言处理（NLP）、计算机视觉（CV）、推荐系统等领域，需要处理海量数据和复杂模型的工业级应用。
*   **模型训练加速：** 对于现有的大型深度学习模型，通过FleetX的分布式和并行能力，显著缩短训练周期，提升研发效率。
*   **高性能计算（HPC）：** 在多GPU、多节点集群环境下进行深度学习任务，充分利用集群算力。
*   **教育与科研：** 为高校和科研机构提供易于上手的分布式训练平台，用于教学和前沿研究。


- [FleetX 飞桨分布式](https://fleet-x.readthedocs.io/en/latest/)
- [PaddlePaddle/FleetX: Paddle Distributed Training Examples. 飞桨分布式训练示例 Resnet Bert GPT MOE DataParallel ModelParallel PipelineParallel HybridParallel AutoParallel Zero Sharding Recompute GradientMerge Offload AMP DGC LocalSGD Wide&Deep](https://github.com/PaddlePaddle/FleetX)
- [分布式训练快速开始-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/cluster_quick_start_cn.html)
- [使用FleetAPI进行分布式训练-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/06_distributed_training/fleet_api_howto_cn.html)
- [FleetX/examples/gpt at develop · PaddlePaddle/FleetX](https://github.com/PaddlePaddle/FleetX/tree/develop/examples/gpt)
- [启动分布式任务 — FleetX 0.1.0.beta documentation](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/launch.html)
- [7. 分布式预测 — PaddleFleetX 0.1.0.beta documentation](https://fleet-x.readthedocs.io/en/latest/paddle_fleet_rst/parameter_server/ps_distributed_evaluation.html)

-----------------------------------------------------------


# 0.PaddleX图形化客户端2.0


 #### 简介
飞桨（PaddlePaddle）是百度开发的开源深度学习平台，旨在提供全流程的深度学习开发工具。PaddleX是基于飞桨核心框架、开发套件和工具组件的全流程开发工具，其特点是“全流程打通、融合产业实践、易用易集成”。飞桨AI Studio星河社区则是一个面向AI学习者的人工智能学习与实训社区，集成了丰富的AI课程、大模型社区、深度学习项目、数据集、GPU算力等资源。AI Studio模型库是该社区的核心组成部分，提供模型存储、版本管理、体验和二次开发功能，覆盖了多种AI领域模型，并支持一站式在线开发部署。

#### 核心功能
*   **PaddleX**: 提供深度学习全流程开发能力，包括模型训练、评估、部署等，通过简明易懂的Python API，支持用户直接调用或二次开发，实现产业实践的快速落地。
*   **AI Studio星河社区**: 提供AI学习与实训环境，包括免费AI课程、大模型社区、深度学习样例项目、经典数据集和免费GPU算力。
*   **AI Studio模型库**: 具备模型存储、版本管理、模型共享（公开与非公开）、在线体验Demo效果、创建模型产线以及支持用户模型上传、托管和一键调用线上开发部署的功能。
*   **模型覆盖**: 涵盖计算机视觉（CV）、自然语言处理（NLP）、智能语音、文心大模型、科学计算、量子计算等多种AI任务方向。
*   **OCR能力**: PaddleOCR作为PaddlePaddle生态的一部分，提供文本检测和识别算法，支持多种模型和技术。

#### 技术原理
PaddleX构建于飞桨（PaddlePaddle）深度学习框架之上，利用其核心框架、开发套件和工具组件，实现深度学习模型的高效开发和部署。它通过封装复杂的底层实现，提供高层API接口，简化开发流程。AI Studio星河社区和模型库则基于云计算架构，提供云端GPU算力支持，使得用户无需本地配置即可进行模型训练、推理和部署。模型库采用分布式存储和版本管理系统，确保模型资产的安全与高效管理。其中，文心大模型体现了百度在预训练大模型领域的最新进展，通过海量数据训练，具备强大的泛化和迁移学习能力。平台还支持多种硬件环境的适配，如Docker容器化部署，增强了技术栈的兼容性和部署灵活性。

#### 应用场景
*   **工业级AI应用开发**: 开发者和企业可利用PaddleX快速构建和部署符合产业需求的深度学习模型，例如智能制造中的缺陷检测、智慧城市中的交通识别等。
*   **AI人才培养与学习**: AI Studio星河社区为初学者和进阶用户提供实践平台，通过课程、项目和比赛提升AI技能，加速AI人才的培养。
*   **模型共享与复用**: 模型库为开发者提供了一个集中管理、共享和发现优质AI模型的平台，促进模型资产的复用和创新，降低开发门槛。
*   **跨领域AI解决方案**: 覆盖CV、NLP、语音等多个领域的模型，可应用于智慧金融、智慧零售、医疗健康、智能客服、智能驾驶等多样化的行业解决方案中。
*   **研究与开发**: 科研人员和开发者可利用AI Studio的免费算力、数据集和模型资源，进行AI算法研究、模型调优和创新性应用的探索。
*   **文本识别与分析**: PaddleOCR的强大能力可应用于文档数字化、票据识别、智慧办公等需要从图像中提取文字信息的场景。


- [PaddleX图形化客户端](https://www.paddlepaddle.org.cn/paddle/paddleX)
- [paddleX快速开始](https://ai.baidu.com/ai-doc/AISTUDIO/Zlisojzjs)
- [模型库 - 飞桨AI Studio - 人工智能学习实训社区](https://aistudio.baidu.com/aistudio/modelsoverview)
- [模型库  - 人工智能学习实训社区](https://aistudio.baidu.com/modelsoverview?category=%E4%BA%A7%E4%B8%9A%E6%96%B9%E6%A1%88&supportPaddlex=1&sortBy=weight)

------------------------------------------------------------


# 0.paddleflow：基于云原生Kubernetes或K3sAI资源管理与调度工具



#### 简介
PaddleFlow是一个开源的AI资源管理与调度平台，旨在为AI开发和部署提供高效的资源管理和任务调度能力，确保AI任务的顺畅运行和资源的高效利用。

#### 核心功能
*   **作业生命周期管理**: 提供客户端 (`paddleflow-client`) 用于作业的准备和打包，服务端 (`paddleflow-server`) 进行作业分析、适应不同运行时环境以及管理。
*   **资源调度与编排**: 能够将作业调度到如Kubernetes或K3s等多种运行时环境中。
*   **存储兼容性与易用性增强**: 提供CSI (Container Storage Interface) 插件以接入PaddleFlowFS，并支持Fuse客户端管理能力，从而提升存储系统的兼容性和用户体验。

#### 技术原理
*   **分布式架构**: PaddleFlow采用客户端-服务端（Client-Server）架构，客户端负责作业的前期准备与封装，服务端则专注于作业的解析、运行时适配及全面管理。
*   **容器化集成**: 深度集成Kubernetes和K3s等容器编排平台，作为其主要的作业运行时环境，实现资源的弹性伸缩和高效隔离。
*   **存储接口标准化**: 通过CSI (Container Storage Interface) 插件机制，实现与各类存储系统（如PaddleFlowFS）的无缝对接，提供标准的存储卷管理能力。
*   **文件系统虚拟化**: 利用Fuse客户端技术，为用户提供灵活的文件系统访问和管理，增强数据处理的便捷性。

#### 应用场景
*   **AI训练与推理任务调度**: 在大规模深度学习训练和推理任务中，实现计算资源的智能分配和任务的高效执行。
*   **AI计算资源管理**: 对GPU、CPU等计算资源进行统一管理和分配，优化资源利用率。
*   **弹性AI开发平台**: 为AI开发者提供一个可弹性扩展、按需使用计算和存储资源的开发与实验环境。
*   **大数据AI工作流**: 结合大数据处理流程，管理和调度涉及大量数据I/O的AI作业。


- [PaddleFlow部署](https://github.com/PaddlePaddle/PaddleFlow/blob/develop/docs/zh_cn/deployment/how_to_install_paddleflow.md)
- [PaddleFlow：AI资源管理与调度工具](https://github.com/PaddlePaddle/PaddleFlow)

------------------------------------------------------------


# 0.paddle部署推理


#### 简介
本次分析涵盖了百度飞桨深度学习平台及其生态系统，包括其核心框架、一站式AI学习与开发平台AI Studio，以及Python开发中常见的包管理工具pip在可执行文件路径方面的技术问题。这些内容共同构建了一个AI技术从理论学习、模型开发到实际应用和问题解决的全链条视角。

#### 核心功能
*   **飞桨深度学习平台 (PaddlePaddle)**：提供易用、高效、灵活、可扩展的深度学习框架，支持各种预训练模型（如PaddleDetection、PaddleOCR、PaddleClas、PaddleNLP、PaddleRec）以及模型部署服务PaddleServing，致力于简化深度学习技术的创新与应用。
*   **飞桨AI Studio**：作为一个面向AI学习者的一站式开发实训平台，其核心功能包括提供丰富的AI课程（视频、项目、文档一体化）、深度学习样例工程、各领域的经典数据集、云端超强运算及存储资源，以及组织AI竞赛和搭建学习社区。
*   **pip包管理**：作为Python的包管理工具，其核心功能在于便捷地安装、升级和管理Python包及其依赖，但在特定情况下会遇到安装的包的可执行文件无法从命令行直接访问的问题。

#### 技术原理
*   **飞桨框架**：基于深度学习算法和模型架构，通过图执行、自动微分、优化器等机制实现高效的模型训练和推理。其“全链条”理念可能涉及到模型开发、训练、压缩、部署的端到端技术集成。
*   **AI Studio平台**：构建于云基础设施之上，利用云计算技术提供弹性计算和存储资源，为用户提供Jupyter Notebook等在线开发环境，集成预装的AI开发库和工具，实现数据的云端管理和模型训练的分布式计算。
*   **pip可执行文件路径问题**：该问题通常与操作系统的环境变量`PATH`相关。当pip安装带有可执行脚本的Python包时，默认会将这些脚本放置在Python环境（如`venv`或`bin`目录）下的特定路径中。如果该路径未被添加到系统的`PATH`环境变量中，命令行就无法找到并执行这些脚本。解决原理通常是手动将该路径加入`PATH`，或使用激活虚拟环境等方式让系统识别到这些可执行文件。

#### 应用场景
*   **飞桨深度学习平台**：广泛应用于计算机视觉（如图像识别、目标检测）、自然语言处理（如文本分类、机器翻译）、推荐系统、语音识别等各类AI模型的研发、训练和生产部署，为企业和开发者提供AI解决方案。
*   **飞桨AI Studio**：是AI爱好者、学生和开发者学习人工智能知识、进行深度学习项目实践、参与AI竞赛以及获取最新AI技术趋势的理想平台，尤其适合初学者入门和进阶。
*   **pip可执行文件路径问题解决**：主要应用于Python开发环境中，当开发者通过pip安装了带有命令行工具的库（如Flask、Django、Jupyter等），但无法直接在终端调用这些工具时，需要通过理解和调整系统环境变量`PATH`来解决，确保开发效率。


- [飞桨训推一体导航](https://www.paddlepaddle.org.cn/wholechain)
- [Ai部署课程](https://aistudio.baidu.com/aistudio/course/list/1)
- [部署相关问题：pip installs packages successfully, but executables not found from command line - Stack Overflow](https://stackoverflow.com/questions/35898734/pip-installs-packages-successfully-but-executables-not-found-from-command-line/65312196#65312196)

------------------------------------------------------------

## FastDeploy


#### 简介
FastDeploy是百度飞桨（PaddlePaddle）团队推出的一款高性能深度学习模型部署工具套件，旨在帮助开发者快速、便捷地将深度学习模型部署到云、边、端等多种硬件平台和应用场景。它提供了一站式、开箱即用的部署体验，覆盖图像、视频、文本和音频等20多个主流场景和150多种SOTA模型，尤其在大型语言模型和视觉语言模型的推理部署方面进行了重点优化。

#### 核心功能
*   **多硬件平台支持**：支持部署到云端、移动设备和边缘设备，兼容多种芯片（如英伟达、昆仑芯、海光、寒武纪、昇腾、燧原、太初、CPU等）和计算平台（CUDA、CANN、NeuWare SDK等）。
*   **广泛的模型覆盖**：支持图像、视频、文本、音频等20+主流AI场景的150+SOTA模型，并持续更新支持主流大模型。
*   **端到端优化**：提供从模型训练到部署的全流程优化，包括模型转换、图优化、量化、多线程推理等。
*   **易用API接口**：提供统一、简洁的API接口，降低部署门槛，方便开发者快速集成。
*   **高性能推理**：通过核心加速技术，如MLA、MTP、量化优化等，实现高吞吐和低延迟的推理性能，特别在大模型部署方面表现优异。
*   **动态图/静态图导出**：支持PaddlePaddle模型的动态图或静态图导出，方便适配不同的推理后端。

#### 技术原理
FastDeploy的核心技术原理在于其对深度学习推理过程的全面优化与加速。它通过集成多种推理引擎（如ONNX Runtime, TensorRT, OpenVINO, Paddle Inference等）和编译器技术，实现了模型在不同硬件上的高效运行。具体而言，它利用了**图优化（Graph Optimization）**技术，对模型结构进行剪枝、融合等操作，减少计算冗余；采用**量化（Quantization）**技术，将模型参数从浮点数转换为低精度整数，显著降低模型大小和计算量，同时保持精度；通过**内存优化（Memory Optimization）**和**多线程/多进程并行（Multi-threading/Multi-processing Parallelism）**策略，最大化硬件利用率；针对大型模型，引入**多卡部署（Multi-GPU Deployment）**和**混合精度推理（Mixed Precision Inference）**，实现跨设备的高效推理。此外，其还支持多种模型格式转换，确保模型在不同推理框架间的无缝衔接。

#### 应用场景
*   **智能安防**：在监控系统中部署行人检测、车辆识别、行为分析等模型，实现实时预警和事件追踪。
*   **智能制造**：应用于工业质检，部署缺陷检测、目标识别模型，提高产品合格率和生产效率。
*   **智慧交通**：部署车流量统计、交通信号灯优化、道路异常事件检测等模型，提升交通管理智能化水平。
*   **智慧零售**：用于客流分析、商品识别、行为分析等，优化门店运营和用户体验。
*   **智能教育**：在教学场景中应用图像识别进行作业批改、语音识别进行口语评测等。
*   **生物医疗**：部署医疗影像分析模型，辅助医生进行疾病诊断。
*   **大模型推理服务**：为大型语言模型（LLM）和视觉语言模型（VLM）提供高效、生产级的推理部署解决方案，支持如ERNIE等大模型在云端或边缘设备的部署。


- [FastDeploy是一款易用高效的推理部署开发套件](https://github.com/PaddlePaddle/FastDeploy/blob/develop/README_CN.md)
- [PaddlePaddle/FastDeploy: ⚡️An Easy-to-use and Fast Deep Learning Model Deployment Toolkit](https://github.com/PaddlePaddle/FastDeploy)
- [FastDeploy秒解模型部署难题，助力智慧农业应用快速落地](https://www.paddlepaddle.org.cn/support/news?action=detail&id=3155)

------------------------------------------------------------

## Paddle Lite端侧推理引擎


- [Profiler 工具 — Paddle-Lite 文档](https://paddle-lite.readthedocs.io/zh/develop/user_guides/profiler.html)
- [Python API — Paddle-Lite 文档](https://paddle-lite.readthedocs.io/zh/develop/api_reference/python_api_doc.html)
- [Welcome to Paddle-Lite's documentation! — Paddle-Lite 文档](https://paddle-lite.readthedocs.io/zh/latest/)
- [PaddlePaddle/Paddle-Lite: Multi-platform high performance deep learning inference engine (飞桨多端多平台高性能深度学习推理引擎）](https://github.com/PaddlePaddle/Paddle-Lite)
- [Paddle Lite_飞桨-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/paddle/paddlelite)

------------------------------------------------------------

## Paddle Serving


#### 简介
Paddle Serving 是基于飞桨（PaddlePaddle）深度学习框架构建的高性能、灵活易用的工业级在线推理服务框架。它旨在为机器学习开发者和企业提供稳定、高效的模型在线部署解决方案，支持多种部署环境和异构硬件，实现模型的快速服务化。

#### 核心功能
*   **高性能在线推理**: 针对深度学习模型提供优化过的推理服务，确保高吞吐和低延迟。
*   **多协议支持**: 支持 RESTful、gRPC、bRPC 等多种主流通信协议，方便客户端集成。
*   **异构硬件兼容**: 提供在 CPU、GPU、ARM CPU、昆仑 XPU 等多种异构硬件和多种操作系统环境下的推理解决方案。
*   **灵活部署方式**: 支持 Docker 容器化部署和 Kubernetes 集群部署，简化部署流程并实现弹性伸缩。
*   **运行时集成**: 通过 FastDeploy Runtime 集成 Paddle Inference、ONNX Runtime、TensorRT、OpenVINO 等多种推理引擎，实现多后端优化。
*   **训推一体化**: 与飞桨框架无缝衔接，支持训练和推理代码复用，提供统一的开发体验。
*   **自动化并行**: 框架具备动静统一自动并行能力，简化大模型训练和推理的并行策略配置。

#### 技术原理
Paddle Serving 的技术原理主要围绕高性能、高可用和易用性展开。其核心是基于飞桨深度学习框架，通过**模型序列化**（生成 .prototxt 等配置文件）和**服务化封装**，将训练好的模型转化为可对外提供推理服务的端点。它采用**多协议通信机制**（如 HTTP/1.1、HTTP/2、bRPC）实现高效的数据传输和请求响应。在底层，通过**FastDeploy Runtime**作为统一的部署接口，能够**抽象并调度不同的推理引擎**（如 Paddle Inference、ONNX Runtime、TensorRT、OpenVINO），这些引擎针对特定硬件（如 NVIDIA GPU、Intel CPU、昆仑 XPU）进行了高度优化，利用**硬件加速能力**（如 Tensor Core、SIMD 指令集）提升推理效率。部署方面，通过支持 **Docker 容器化**提供环境隔离和快速部署能力，结合 **Kubernetes 容器编排**技术实现服务的自动化部署、弹性伸缩和高可用性管理。飞桨框架自身的**自动并行技术**和**统一动静态图执行**机制，也为Serving层提供了高效的模型执行基础。

#### 应用场景
*   **智能图像识别**: 如图片分类、目标检测、图像分割、OCR（光学字符识别）等在线推理服务。
*   **自然语言处理**: 如文本分类、机器翻译、情感分析、智能问答等模型的线上部署。
*   **推荐系统**: 对用户行为数据进行实时预测，提供个性化推荐。
*   **语音识别与合成**: 将语音模型部署为服务，实现实时语音转文字或文字转语音。
*   **工业质检与安防监控**: 部署视觉模型进行缺陷检测、行为分析等。
*   **大模型推理**: 为大型预训练模型提供高性能、可扩展的在线推理服务。


- [Paddle Serving](https://github.com/PaddlePaddle/Serving)
- [Paddle Serving服务化部署框架](https://www.paddlepaddle.org.cn/tutorials/projectdetail/3946013)

------------------------------------------------------------

## PaddleFleetX

#### 简介
PaddleFleetX 是飞桨（PaddlePaddle）深度学习平台推出的大模型开发套件，旨在提供高效、易用的大语言模型和跨模态大模型分布式训练、开发与部署解决方案。它致力于简化大规模深度学习模型的训练过程，提升开发效率和模型性能。

#### 核心功能
*   **大模型训练支持：** 提供对大语言模型（LLM）和跨模态大模型的训练能力。
*   **分布式训练：** 支持数据并行、模型并行（包括流水线并行、张量并行）等多种分布式训练策略。
*   **优化与加速：** 包含梯度累积（Gradient Accumulation）等优化技术，以提升训练效率和显存利用率。
*   **灵活的部署：** 支持Docker镜像、物理机、K8S平台等多种环境部署。
*   **统一API：** 提供统一的Fleet API，简化分布式训练代码开发。

#### 技术原理
PaddleFleetX 的核心技术原理在于其对大规模分布式训练的支持。它基于PaddlePaddle深度学习框架，通过实现多种分布式策略来解决大模型训练面临的内存和计算资源挑战：
*   **数据并行 (Data Parallelism)：** 在多个设备上复制模型，每个设备处理不同批次的数据，梯度聚合后更新模型。
*   **模型并行 (Model Parallelism)：** 当模型过大无法放入单个设备内存时，将模型切分到多个设备上，包括：
    *   **流水线并行 (Pipeline Parallelism)：** 将模型的层级按顺序划分到不同设备，形成计算流水线。
    *   **张量并行 (Tensor Parallelism)：** 将模型内部的张量（如线性层权重）切分到不同设备。
*   **ParameterServer (PS) 架构：** 支持ParameterServer模式进行参数的分布式存储和更新，适用于大规模稀疏模型的训练。
*   **Collective通信：** 利用NCCL等高效的集体通信库，实现多设备间的数据交换和梯度同步。
*   **动态图训练：** 兼容动态图模式，提供更灵活的调试和开发体验。

#### 应用场景
*   **大规模预训练：** 适用于BERT、GPT等大语言模型，以及多模态模型的预训练任务。
*   **深度学习研究：** 为探索和验证新型大模型架构和分布式训练算法提供平台。
*   **企业级AI解决方案：** 支持企业在私有云或公有云上部署和训练大规模AI模型，应用于自然语言处理、计算机视觉等领域。
*   **高计算需求场景：** 针对需要利用集群算力进行模型训练和优化的场景，如科学计算、金融风控等。


- [PaddleFleetX部署](https://github.com/PaddlePaddle/PaddleFleetX)

------------------------------------------------------------

## 服务器部署 — Paddle Inference

#### 简介
Paddle Inference 是飞桨（PaddlePaddle）深度学习框架的原生推理库，专为服务器端和云端部署设计，旨在提供高性能、低时延的推理能力。它能够通用支持所有通过飞桨训练出的模型，并针对不同平台和应用场景进行了深度适配优化，确保模型的高效部署和即训即用。

#### 核心功能
*   **高性能推理：** 提供高吞吐、低时延的模型推理服务。
*   **模型兼容性：** 直接基于飞桨训练算子，支持所有飞桨训练出的模型。
*   **跨平台支持：** 提供Windows、Linux、MacOS等平台的预测库下载，并支持源码编译，适应多种硬件环境（如GPU、CPU、飞腾、鲲鹏、申威、兆芯、龙芯等）。
*   **部署工具链：** 包含模型导出、模型压缩（如PaddleSlim的剪枝、量化、知识蒸馏等）、模型转换（如X2Paddle）等功能，便于模型部署优化。
*   **易用性：** 提供Python和C++等多种语言的快速上手示例，简化开发者的使用流程。

#### 技术原理
Paddle Inference 采用 **Predictor** 作为其核心推理引擎。Predictor 通过对计算图的深入分析，执行一系列优化操作来显著提升推理性能，包括：
*   **算子融合：** 合并多个计算操作以减少计算开销。
*   **内存/显存优化：** 精心管理内存和显存，提高资源利用率。
*   **底层加速库集成：** 集成并支持 **MKLDNN**（针对CPU优化）、**TensorRT**（针对GPU优化）等高性能底层加速库。通过子图方式集成TensorRT，将可加速的算子组成子图交给TensorRT处理，同时保持飞桨的灵活性。
*   **模型格式：** 支持将训练好的模型导出为 `.pdmodel` 格式的推理模型，以便于部署。
*   **动转静：** 支持将动态图模型转换为静态图模型，以利于部署和优化。

#### 应用场景
*   **服务器端模型部署：** 在服务器环境中，为各类深度学习应用提供高性能的模型推理服务，如图像识别、自然语言处理等。
*   **云端服务：** 将训练好的模型部署到云平台，提供弹性、可扩展的AI推理API服务。
*   **工业级应用：** 适用于对模型推理性能、稳定性和兼容性有高要求的工业生产环境。
*   **AI推理服务搭建：** 开发者可以利用Paddle Inference 快速搭建和部署深度学习推理服务，实现从模型训练到线上部署的全流程。


- [服务器部署 — Paddle Inference-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/inference/inference_cn.html)
- [PaddlePaddle/Paddle-Inference-Demo](https://github.com/PaddlePaddle/Paddle-Inference-Demo)
- [下载安装Linux预测库 — Paddle-Inference documentation](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)
- [推理流程 — Paddle-Inference documentation](https://paddle-inference.readthedocs.io/en/latest/guides/introduction/workflow.html#)
- [飞桨框架模型导出 — Paddle-Inference documentation](https://paddle-inference.readthedocs.io/en/latest/guides/export_model/paddle_model_export.html#paddleslim)
- [快速上手Python推理 — Paddle-Inference documentation](https://paddle-inference.readthedocs.io/en/latest/guides/quick_start/python_demo.html)

------------------------------------------------------------


# 0.PaddleSlim模型压缩蒸馏

#### 简介
PaddleSlim 是百度飞桨（PaddlePaddle）开源的深度学习模型压缩和架构搜索库，旨在帮助开发者实现模型的小型化，从而提高模型在实际部署时的推理性能和效率。它作为 PaddlePaddle 框架的子模块，专注于提供一系列模型优化策略，广泛应用于计算机视觉和自然语言处理等领域。

#### 核心功能
*   **模型剪裁 (Pruning)**：通过移除模型中冗余的连接或神经元来减小模型大小。
*   **定点量化 (Quantization)**：支持低比特量化、定点量化、量化训练和离线量化，将模型参数从浮点数转换为低精度定点数，以减少模型体积和计算开销。
*   **知识蒸馏 (Knowledge Distillation)**：利用大型教师模型的知识来训练小型学生模型，使其在保持性能的同时减小模型规模。
*   **超参数搜索 (Hyperparameter Search)**：自动化寻找模型训练和压缩过程中的最优超参数配置。
*   **模型结构搜索 (Neural Architecture Search, NAS)**：自动化设计高效的神经网络结构。
*   **自动化压缩工具 (ACT - Auto Compression Toolkit)**：提供自动化模型压缩解决方案，支持解耦训练代码，基于推理模型和无监督数据进行压缩，性能等效人工压缩。
*   **跨框架支持**：兼容 PaddlePaddle、PyTorch 和 TensorFlow 产出的推理模型。

#### 技术原理
PaddleSlim 的技术原理涵盖了深度学习模型优化的多个关键方向：
*   **剪裁 (Pruning)**：基于冗余分析，通过移除权重、通道或层等对模型性能影响较小的部分，实现模型稀疏化，降低FLOPs和参数量。
*   **量化 (Quantization)**：核心在于将浮点型参数和激活值映射到低比特整数表示（如 INT8），这涉及到量化方案设计（如对称量化、非对称量化）、量化感知训练（QAT）和训练后量化（PTQ）等技术，旨在最小化量化误差并提升推理速度。
*   **知识蒸馏 (Knowledge Distillation)**：通过最小化学生模型输出与教师模型软标签（Soft Targets）之间的差异来训练学生模型，从而将教师模型的“知识”迁移到学生模型，实现模型瘦身的同时保持高精度。
*   **超参数优化 (Hyperparameter Optimization)**：采用网格搜索、随机搜索、贝叶斯优化等算法，自动探索并确定各种压缩策略中的最佳超参数组合，以获得最优压缩效果。
*   **神经网络结构搜索 (NAS)**：利用强化学习、进化算法或梯度下降等方法，在预定义的搜索空间中自动探索和评估不同的网络结构，从而找到更适合特定任务和部署环境的轻量级模型。
*   **自动化压缩 (ACT)**：将上述多种压缩策略进行集成和自动化，通常采用迭代或协同优化的方式，在给定约束下寻找最佳的压缩策略组合和参数，实现端到端的模型优化流程。

#### 应用场景
*   **云端部署**：加速深度学习模型在服务器端的推理速度，降低计算资源消耗。
*   **边缘设备部署**：将大型深度学习模型部署到计算能力和存储空间有限的移动设备、物联网设备、嵌入式系统和智能硬件（如 Nvidia GPU、ARM 设备）上，满足实时性要求。
*   **计算机视觉任务**：广泛应用于图像分类、目标检测、语义分割等视觉任务的模型压缩。
*   **自然语言处理任务**：优化大型语言模型（LLM）和预训练模型（如 ERNIE 3.0）的性能，实现无损压缩和高性能推理，降低大模型部署门槛。
*   **工业级应用**：为企业级应用提供高效、易用的模型小型化解决方案，加速AI技术在实际生产环境中的落地。


- [0.PaddleSlim模型压缩蒸馏](https://github.com/PaddlePaddle/PaddleSlim)
- [0. PaddleSlim 文档](https://paddleslim.readthedocs.io/zh_CN/develop/intro.html)
- [0.模型压缩 — PaddleSlim-](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/infer/paddleslim/paddle_slim_cn.html)
- [0.1模型自动化压缩工具ACT（Auto Compression Toolkit）](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression)
- [0.2自然语言处理模型自动压缩示例](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/nlp)
- [0.3ACT超参详细教程](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/hyperparameter_tutorial.md)
- [1.PaddleNLP 模型压缩 API](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/docs/compression.md)
- [ERNIE 3.0 轻量级模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-3.0/README.md#%E5%BE%AE%E8%B0%83)
- [ERNIE 3.0 Python部署指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-3.0/deploy/python)

------------------------------------------------------------

## 1.NLP蒸馏算法策略

#### 简介
百度AI Studio是一个基于飞桨深度学习平台的人工智能学习与实训社区，旨在提供一站式AI开发和教育解决方案，包括在线编程环境、免费算力、AI课程与资源等。ERNIE (文心大模型) 是百度开发的系列预训练大模型，具备多模态理解与生成能力，并通过开放API和开源策略，致力于推动AI技术的普及与应用。PaddleSlim则是飞桨生态下的模型压缩和架构搜索工具库，专注于优化AI模型，使其更高效地部署和运行，特别适用于大型模型的轻量化。

#### 核心功能
*   **AI Studio**: 提供在线AI开发与实训环境、海量AI课程和数据集、云端CPU/GPU算力支持、一站式教学管理系统及零代码/低代码AI应用构建能力，服务于AI学习、开发和教育。
*   **ERNIE大模型**: 支持文本、图像、音频和视频等多模态内容的理解与生成，具备强大的语义理解、知识推理和创作能力，并通过API和开源模型满足企业级应用和开发者需求。
*   **PaddleSlim**: 提供模型剪枝（Pruning）、量化（Quantization，包括定点量化、量化训练和离线量化）、知识蒸馏（Knowledge Distillation）和神经网络架构搜索（NAS）等多种模型压缩策略，旨在减小模型体积、提高推理速度并降低资源消耗。

#### 技术原理
ERNIE系列模型采用先进的Transformer架构和大规模预训练技术，通过海量数据学习通用知识和语言表示，实现多模态信息的统一建模和交叉理解。知识蒸馏是ERNIE模型压缩的关键技术之一，它通过训练一个小型学生模型来模仿大型教师模型的输出和中间表示，从而在保持性能的同时大幅减小模型规模。PaddleSlim则整合了多种模型优化算法：
*   **剪枝（Pruning）**: 识别并移除神经网络中不重要的连接或神经元，以减少模型参数量和计算量。
*   **量化（Quantization）**: 将模型参数和/或激活值从浮点数表示转换为低位宽（如8位整数）表示，以减少模型大小和计算复杂度。
*   **知识蒸馏（Knowledge Distillation）**: 通过软目标（Soft Targets）或特征图（Feature Maps）将教师模型的知识迁移到学生模型，使小型模型获得接近大型模型的性能。
*   **神经网络架构搜索（NAS）**: 自动化设计高效的模型网络结构，以满足特定硬件或性能需求。

#### 应用场景
*   **AI教育与科研**: AI Studio为高校、机构和个人提供AI学习、课程开发、实验实训和项目开发的平台。
*   **大模型应用开发与部署**: 利用ERNIE大模型构建智能对话系统、内容创作、智能问答、多模态检索等AI应用，并通过API集成到各类产品中。
*   **模型轻量化与端侧部署**: 运用PaddleSlim对ERNIE或其他飞桨模型进行压缩，使其能够在资源受限的设备（如移动设备、边缘设备）上高效部署，或在云端实现更低成本、更高吞吐的推理服务。
*   **企业级AI解决方案**: 结合ERNIE的强大能力和PaddleSlim的优化技术，为企业提供高效、可落地的AI解决方案，应用于智能客服、智能营销、智能制造、智慧城市等领域。


- [bert预训练模型蒸馏技术讲解](https://aistudio.baidu.com/aistudio/projectdetail/2258091)
- [0.模型蒸馏任务详细记录实现过程](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/Hl57xuabm)
- [BERT模型压缩教程——含蒸馏原理](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/nlp/paddlenlp_slim_ofa_tutorial.md)

------------------------------------------------------------

## 2.案例实现

#### 简介
主要围绕飞桨（PaddlePaddle）深度学习框架在自然语言处理（NLP）领域的应用，特别是大语言模型（LLM）的模型压缩技术。PaddleNLP作为一个强大且易用的LLM开发套件，旨在通过知识蒸馏、剪枝、量化等多种压缩策略，实现大模型的轻量化、高效训练、无损压缩以及高性能推理，从而降低部署难度，助力开发者实现产业级应用。其中，ERNIE系列模型作为核心基础，结合模型压缩技术，进一步提升了其在实际场景中的应用效率。

#### 核心功能
*   **大模型压缩：** 提供多种模型压缩策略，包括知识蒸馏（如BERT到Bi-LSTM的蒸馏、MiniLMv2、PP-MiniLM）、模型剪枝和量化，以显著减小模型尺寸、降低内存消耗和计算量，并提升推理速度。
*   **高效训练与推理：** 支持对大型模型进行高效训练，并优化其推理性能，确保在资源受限或对实时性要求高的场景下也能稳定运行。
*   **ERNIE模型支持：** 兼容并优化ERNIE系列预训练模型，使其能够通过压缩技术应用于各种下游任务。
*   **多任务NLP支持：** 涵盖广泛的NLP任务，如文本分类、神经网络搜索、问答、信息抽取、文档智能和情感分析等。
*   **工业级应用工具：** 提供工业级开发工具包（如ERNIEKit）和API，简化大模型从训练到部署的全流程。

#### 技术原理
*   **知识蒸馏 (Knowledge Distillation)：** 核心思想是将一个大型、高性能的“教师模型”（Teacher Model）的知识迁移到一个小型、高效的“学生模型”（Student Model）中。通过让学生模型学习教师模型的软目标（soft targets），即教师模型的输出概率分布或中间层表示，从而在模型尺寸大幅减小的同时，尽可能地保留教师模型的性能。“暗知识”（dark knowledge）的传递是其关键。
*   **模型剪枝 (Pruning)：** 通过移除模型中冗余或不重要的连接、神经元或层来减小模型大小。这可以是结构化剪枝（如移除整个通道或层）或非结构化剪枝（如移除单个权重），旨在不显著影响模型性能的前提下，减少参数数量和计算量。
*   **模型量化 (Quantization)：** 将模型的浮点数参数和激活值转换为低精度（如INT8）表示，从而减少模型大小、内存占用和计算复杂性。通常在训练后或训练过程中进行（如QAT，Quantization Aware Training），以最小化精度损失。
*   **统一框架与优化：** 基于PaddlePaddle深度学习框架，通过PaddleSlim等工具集成了上述压缩策略，并针对不同硬件（如XPU）进行优化，实现跨平台的高性能部署。

#### 应用场景
*   **轻量级部署：** 在移动设备、边缘计算设备或计算资源有限的环境中部署大型NLP模型，如智能手机上的语音助手、离线翻译应用。
*   **实时推理服务：** 需要快速响应的在线服务，如智能客服问答系统、实时内容审核、搜索引擎的语义匹配等。
*   **大规模数据处理：** 对海量文本数据进行高效分析和处理，如情感分析、文本分类、信息抽取等，降低运营成本。
*   **定制化模型开发：** 针对特定业务场景，快速训练和部署满足性能与效率平衡的定制化NLP模型。
*   **学术研究与工业实践：** 为研究人员提供模型压缩的实验平台，为企业提供加速AI应用落地的解决方案。


- [erniekit--蒸馏](https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/applications/tasks/data_distillation)
- [BERT模型的知识蒸馏到基于Bi-LSTM的小模型中](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/model_compression/distill_lstm/README.md)
- [MiniLMv2蒸馏策略](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/model_compression/minilmv2/README.md)
- [BERT Compression Based on PaddleSlim](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/model_compression/ofa/README.md)
- [PP-MiniLM 中文小模型蒸馏剪裁量化](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/model_compression/pp-minilm/README.md)

------------------------------------------------------------

## paddle2ONNX部署



#### 简介
Paddle2ONNX 是百度飞桨（PaddlePaddle）官方开发的一个模型转换工具。它专注于将基于 PaddlePaddle 框架训练和导出的深度学习模型，高效且准确地转换为开放神经网络交换（ONNX）格式的模型。这一转换旨在促进模型在不同深度学习框架、运行时环境以及各种硬件平台之间的互操作性和部署。

#### 核心功能
*   **模型格式转换：** 实现 PaddlePaddle 模型到 ONNX 格式的无缝转换。
*   **多版本 Opset 支持：** 稳定支持 ONNX Opset 版本 7 至 19 的导出，并兼容部分 Paddle 算子向更低 ONNX Opset 版本的转换。
*   **算子兼容性处理：** 提供对 PaddlePaddle 内部算子与 ONNX 算子之间映射关系的处理，确保转换后的模型在功能上的一致性。

#### 技术原理
Paddle2ONNX 的核心技术原理涉及模型解析、图结构转换和算子映射。它首先解析 PaddlePaddle 模型的计算图结构，识别其中的层和算子。随后，工具将这些 PaddlePaddle 特定的算子和数据流转换为 ONNX 定义的标准算子和图表示。这个过程需要精密的算子映射表和图优化策略，以确保转换后的 ONNX 模型在保持原始模型精度和性能的同时，符合 ONNX 规范并能够被支持 ONNX 的运行时高效执行。其对不同 ONNX Opset 版本的支持，表明其内部实现具备灵活的算子版本适配能力。

#### 应用场景
*   **跨平台部署：** 允许将 PaddlePaddle 训练的模型部署到各种支持 ONNX 运行时（如 ONNX Runtime）的平台和设备上，包括服务器、云端、边缘设备和移动端。
*   **框架互操作性：** 促进 PaddlePaddle 与其他深度学习框架（如 PyTorch, TensorFlow 等）之间模型共享和协作，因为 ONNX 作为一个通用格式，可以作为模型交换的桥梁。
*   **推理优化：** 利用 ONNX Runtime 或其他 ONNX 兼容推理引擎的优化能力，提升 PaddlePaddle 模型在推理阶段的性能和效率。
*   **硬件加速利用：** 便于模型在各类针对 ONNX 优化过的硬件加速器上进行部署和加速推理。


- [Paddle2ONNX: ONNX Model Exporter for PaddlePaddle](https://github.com/PaddlePaddle/Paddle2ONNX)

------------------------------------------------------------

# 0.可解释性模型

#### 简介
飞桨可信AI（TrustAI）和模型可解释性工具包（InterpretDL）是基于百度飞桨（PaddlePaddle）深度学习平台开发的工具集。TrustAI旨在提升深度学习模型的可靠性和可信度，通过可信分析和增强功能，解决训练数据缺陷并优化模型表现，尤其适用于自然语言处理（NLP）领域。InterpretDL则是一个专注于模型可解释性的算法库，集成了多种先进的解释方法，帮助开发者理解深度学习模型的决策过程，增强模型透明度和可信赖性。两者共同致力于推动AI模型在实际应用中的安全、可靠落地。

#### 核心功能
*   **TrustAI**:
    *   **可信分析**: 提供特征级和实例级证据分析，以解释模型预测的依据；识别训练数据覆盖不足和数据分布偏置。
    *   **可信增强**: 针对识别出的数据缺陷（如标注错误）进行数据清洗和优化，并提供基于证据指导的预测机制，以解决长文本理解等问题。
*   **InterpretDL**:
    *   **模型可解释性算法库**: 集成了LIME、Grad-CAM、Integrated Gradients等多种经典及SOTA可解释性算法，支持对模型输入特征、中间层特征和数据集层面进行解释。
    *   **模型评估**: 提供可解释性算法的评估能力。
*   **PaddleNLP集成**: TrustAI的能力已接入PaddleNLP，用于解决文本分类系统方案中的训练数据缺陷，提升NLP模型的可靠性。

#### 技术原理
*   **TrustAI**:
    *   **证据分析**: 通过分析模型预测结果，从输入数据（如文本）中提取关键特征或识别对预测有显著影响的数据实例，以揭示模型决策的关键线索。
    *   **数据缺陷识别**: 运用统计学方法或基于模型的技术来发现数据集中的偏置（如标签偏差、数据分布不均），评估其对模型鲁棒性的影响。
    *   **可信增强机制**: 通过自动化或半自动化的数据处理（如对错误标注数据的修正）和优化训练策略，增强模型在复杂或有缺陷数据下的泛化能力和准确性。
*   **InterpretDL**:
    *   **局部近似模型**: 如LIME，在局部通过训练一个简单的、可解释的模型来近似复杂深度学习模型的行为。
    *   **梯度可视化**: 如Grad-CAM，通过计算输出相对于中间特征图的梯度，来可视化模型在输入中关注的区域。
    *   **积分路径方法**: 如Integrated Gradients，沿着从基准输入到实际输入的路径对梯度进行积分，量化每个输入特征对预测的累积贡献。
    *   所有解释器遵循统一的API接口（如`interpret(kwargs)`），简化了多种算法的调用和集成。
*   **基础框架**: 所有工具均构建于PaddlePaddle深度学习框架之上，利用其高效的训练和推理能力。

#### 应用场景
*   **AI模型开发与部署**: 辅助开发者在模型训练和部署过程中，识别并解决潜在的数据缺陷和模型偏差，确保模型的准确性和泛化能力。
*   **高风险领域应用**: 在金融、医疗、法律等对AI模型可信度和透明度有严格要求的领域，提供模型决策依据，满足合规性要求。
*   **自然语言处理**: 提升文本分类、情感分析、问答系统等NLP任务中模型的可靠性，例如，解释模型为何给出特定文本分类结果，或识别导致模型错误判断的输入偏置。
*   **模型调试与优化**: 帮助数据科学家和工程师深入理解模型内部机制，发现并解决模型性能瓶颈或错误行为的根源。
*   **教育与研究**: 为研究人员和学生提供一个强大的平台，用于探索和开发新的模型可解释性及可信AI技术。


- [TrustAI可信评测](https://github.com/PaddlePaddle/TrustAI)
- [NLP可解释评估](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/model_interpretation/README.md)
- [InterpretDL: 基于『飞桨』的模型可解释性算法库](https://github.com/PaddlePaddle/InterpretDL/blob/master/README_CN.md)
- [InterpretDL](https://github.com/PaddlePaddle/InterpretDL/blob/master/tutorials/README.md)
- [InterpretDL Examples and Tutorials](https://github.com/PaddlePaddle/InterpretDL/blob/master/examples/README.md)

------------------------------------------------------------

# 0.模型性能分析

#### 简介
本总结综合了性能分析工具 Perfetto UI、Chrome tracing，以及深度学习框架 PaddlePaddle 在模型性能优化方面的多种技术，包括其 Profiler 工具、自动混合精度训练（AMP）和模型量化。这些技术和工具旨在帮助开发者识别性能瓶颈、优化资源利用，从而提升深度学习模型在训练和推理阶段的效率和速度。

#### 核心功能
*   **性能数据可视化与分析**: Perfetto UI 和 Chrome tracing 提供对系统、浏览器、应用以及深度学习模型运行时性能数据的详细可视化和分析能力，包括事件时间线、CPU使用、GPU活动等。
*   **深度学习模型性能分析**: PaddlePaddle Profiler 能够收集、统计并展示模型训练和推理过程中的性能数据，帮助用户定位性能瓶颈。
*   **模型训练优化**:
    *   **自动混合精度训练 (AMP)**: 通过使用 16 位浮点数（如 float16、bfloat16）替代传统的 32 位浮点数进行计算，以减少显存消耗和加速计算过程。
    *   **模型量化**: 将模型参数和计算从浮点数转换为低比特整数（如 INT8），以减小模型体积、加速推理速度并降低内存带宽需求。

#### 技术原理
*   **事件追踪与剖析**: Perfetto 和 Chrome tracing 基于事件追踪机制，通过在关键代码路径中插入埋点，记录时间戳、事件名称、进程/线程信息等，形成追踪数据。这些数据经过解析后，在可视化界面中以时间线、火焰图等形式展现，揭示系统和应用的运行时行为。
*   **数据类型优化**:
    *   **AMP (Automatic Mixed Precision)**: 利用支持 FP16 或 BF16 的硬件（如 GPU Tensor Cores）加速矩阵乘法和卷积运算。框架会自动识别并转换适合使用低精度的数据类型，同时保留关键部分的 FP32 精度以确保模型收敛性。
    *   **Quantization**: 通常采用后训练量化（Post-Training Quantization, PTQ）或训练时量化（Quantization Aware Training, QAT）。PTQ 直接将训练好的浮点模型转换为量化模型，而 QAT 则在训练过程中模拟量化误差，使模型对量化更鲁棒。量化减少了内存占用和计算量，因为整数运算通常比浮点运算更快、更节能。
*   **性能指标收集**: PaddlePaddle Profiler 通过集成框架底层的运行时信息收集机制，例如操作（Op）执行时间、内存分配、显存使用、I/O 操作等，并提供 API 供用户在模型代码中开启和配置性能分析会话。

#### 应用场景
*   **系统级性能调试**: 开发者可以使用 Perfetto UI 和 Chrome tracing 诊断操作系统、浏览器或应用程序的性能问题，例如 UI 渲染卡顿、启动速度慢、资源占用高等。
*   **深度学习模型训练优化**: 针对大型深度学习模型，利用 PaddlePaddle 的 AMP 和 Profiler 功能，可以有效减少训练时间和内存消耗，加速模型迭代和实验。
*   **模型部署与推理加速**: 模型量化技术广泛应用于移动设备、嵌入式系统和边缘计算设备上的深度学习模型部署，通过减小模型体积和加速推理，满足实时性、低功耗的需求。
*   **资源受限环境下的AI应用**: 在服务器或数据中心资源有限的场景，通过性能分析和优化，可以更高效地利用 GPU、CPU 等硬件资源，提升吞吐量。


- [性能分析插件：最新版本Perfetto UI](https://ui.perfetto.dev/)
- [性能分析插件：谷歌-tracing](chrome://tracing/)
- [Profiler模型性能分析-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/profiling_model.html#profiler)
- [paddle.profiler-API文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/profiler/Overview_cn.html)
- [自动混合精度训练（AMP）-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/amp_cn.html)
- [飞桨模型量化-使用文档-PaddlePaddle深度学习平台](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/performance_improving/quantization.html)
- [使用 FasterTokenizer 加速 · Issue #3141 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/3141)

------------------------------------------------------------


# 1.LLM-Paddle

 #### 简介
PaddleNLP 是一个基于飞桨（PaddlePaddle）深度学习框架的大语言模型 (LLM) 开发套件，旨在为开发者提供高效的大模型训练、无损压缩和高性能推理能力。它集成了丰富的模型库、简洁易用的API以及高性能分布式训练能力，致力于助力开发者实现高效的自然语言处理（NLP）和大规模语言模型（LLM）产业级应用。

#### 核心功能
*   **大语言模型开发与预训练:** 提供大语言模型（LLM）预训练的完整流程支持，包括数据准备、模型构建、训练启动及调优建议。
*   **模型训练与优化:** 支持在多种硬件上进行高效的大模型训练，并提供无损压缩（如模型蒸馏、量化）和高性能推理能力。
*   **丰富的模型库:** 拥有覆盖多场景的模型库，包含预训练模型、通用小模型（如ERNIE-Tiny、DynaBERT、TinyBERT、MiniLM等）供下游任务微调使用。
*   **开箱即用的NLP任务:** 提供产业级的NLP预置任务能力，涵盖自然语言理解与自然语言生成两大核心应用，支持一键预测。
*   **易用与高性能:** 具备简单易用、性能极致的特点，通过统一的应用范式 `paddlenlp.Taskflow` 提供调用。

#### 技术原理
PaddleNLP 基于百度飞桨深度学习框架构建，利用其动静统一的特性和分布式训练能力。其技术原理主要包括：
*   **飞桨深度学习框架:** 作为底层支持，提供高效的张量运算、自动微分和优化器等深度学习核心功能。
*   **分布式训练:** 支持在多设备、多节点上进行高效的大模型训练，通过并行化策略（如数据并行、模型并行）加速训练过程。
*   **模型压缩技术:** 采用蒸馏（Distillation）、量化（Quantization）等技术，在保证模型性能的同时，大幅减小模型体积，提升推理速度。
*   **高性能推理引擎:** 针对LLM的特点，优化推理计算图，实现低延迟、高吞吐的推理服务。
*   **模块化API设计:** 提供简洁、统一的API接口，如 `paddlenlp.Taskflow`，封装复杂的底层实现，便于开发者快速集成和使用。

#### 应用场景
*   **大语言模型开发与研究:** 为LLM的研究人员和开发者提供便捷的预训练、微调和部署工具链。
*   **自然语言理解（NLU）任务:** 应用于文本分类、命名实体识别、情感分析、问答系统等。
*   **自然语言生成（NLG）任务:** 用于文本摘要、机器翻译、对话系统、内容生成等。
*   **产业级NLP解决方案:** 助力企业在智能客服、智能营销、智能内容创作等领域构建高效的AI应用。
*   **资源受限环境下的模型部署:** 通过模型压缩技术，使得大型语言模型能够在边缘设备或移动端进行部署和推理。


- [大模型预训练介绍 — PaddleNLP 文档](https://paddlenlp.readthedocs.io/zh/latest/llm/pretraining/index.html)
- [飞桨大语言模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm)

------------------------------------------------------------

## 0.ERNIE SDK-含Agent


#### 简介
ERNIE Bot SDK 是百度飞桨推出的基于文心大模型（ERNIE Bot）的智能体（Agent）开发框架，旨在为开发者提供便捷的接口，轻松调用文心大模型的各项功能。该 SDK 包含 ERNIE Bot Agent 和 ERNIE Bot 两个主要项目，其中 ERNIE Bot Agent 是一个大模型智能体开发框架，结合了飞桨星河社区的丰富预置平台功能，而 ERNIE Bot 则提供对文心大模型基础能力的调用，如文本创作、通用对话、语义向量及AI作图等。同时，百度 AI Studio 是一个面向 AI 学习者和开发者的一站式开发实训平台，提供在线编程环境、免费算力及丰富的 AI 课程，支持 AI 学习、实训和项目开发。

#### 核心功能
*   **ERNIE Bot Agent 框架：** 提供基于文心大模型编排能力的智能体开发框架，支持快速构建和部署 AI 智能体。
*   **ERNIE Bot 能力调用：** 提供便捷接口，允许开发者调用文心大模型的多种核心功能，包括文本生成、通用对话、语义理解、向量嵌入以及图像生成等。
*   **开发与实训环境：** 百度 AI Studio 提供集成化的在线开发环境，包含 AI 课程、深度学习样例工程、数据集和云端计算资源，方便用户进行 AI 学习、实践和项目开发。
*   **模块化与可扩展性：** SDK 设计注重模块化，便于开发者根据需求进行功能扩展和集成。

#### 技术原理
ERNIE Bot SDK 的核心技术基于百度自研的文心大模型（ERNIE），这是一个通过知识增强实现表示学习的预训练语言模型。SDK 内部通过封装 RPC 或 RESTful API 调用，实现与后端 ERNIE 大模型服务的通信。ERNIE Bot Agent 框架则在此基础上，引入了智能体（Agent）的概念，通过编排（orchestration）和规划（planning）机制，使大模型能够理解复杂指令、调用外部工具或 API，并执行多步骤任务。其底层依赖于深度学习框架如 PaddlePaddle，利用大规模语料库进行预训练，并通过多任务学习、知识蒸馏等技术优化模型性能和效率。AI Studio 则提供云计算基础设施，支撑模型的训练、推理和部署，为用户提供容器化、分布式计算的开发环境。

#### 应用场景
*   **智能客服与对话系统：** 利用 ERNIE Bot 的通用对话和语义理解能力，开发高效智能客服、聊天机器人。
*   **内容创作与辅助：** 应用文本创作和AI作图功能，辅助文章撰写、广告文案生成、创意设计等。
*   **智能体开发与部署：** 基于 ERNIE Bot Agent 框架，快速构建可执行复杂任务的 AI 智能体，例如自动化工作流、智能助手。
*   **教育与科研：** 百度 AI Studio 作为一站式平台，为 AI 学习者和研究人员提供实践环境、数据集和算力支持，用于课程学习、算法验证和项目开发。
*   **企业级应用集成：** 开发者可将 ERNIE Bot SDK 集成到企业现有系统中，实现智能问答、数据分析、决策支持等智能化升级。


- [ERNIE Bot SDK](https://github.com/PaddlePaddle/ERNIE-Bot-SDK)
- [ERNIE Bot Agent](https://ernie-bot-agent.readthedocs.io/zh-cn/latest/)
- [文心一言插件开发课 -](https://aistudio.baidu.com/course/introduce/30125)

------------------------------------------------------------

## 1.大模型应用

#### 简介
共同展示了基于大语言模型（LLM）和自然语言处理（NLP）技术的多种应用与开发实践。核心内容涵盖了从构建智能对话系统（聊天机器人、代理）到搭建本地知识库问答系统，以及利用大模型进行创意内容生成（如小说续写）和模型效果评估等。其中，PaddleNLP作为核心框架，LangChain和ChatGLM作为关键技术组件，AI Studio则提供了便捷的开发与部署环境。

#### 核心功能
*   **智能对话系统构建：** 支持开发基于检索和生成模式的聊天机器人以及能够利用工具回答查询的智能代理。
*   **本地知识库问答：** 实现针对特定文档或本地知识进行问答，提供精准信息检索和生成回复。
*   **多模型效果评测：** 提供一个平台用于对比和评估不同大语言模型的生成效果。
*   **创意内容生成：** 例如中文小说续写功能，能够根据前文自动生成后续内容。
*   **NLP任务支持：** 涵盖文本分类、神经搜索、问答系统、信息抽取、情感分析等广泛的NLP应用。
*   **文档交互：** 允许用户通过对话方式与文档内容进行交互，实现对PDF等多模态文档的问答。

#### 技术原理
这些应用主要基于以下技术原理：
*   **大语言模型（LLM）：** 利用如ChatGLM-6B、ChatRWKV等预训练大模型作为核心，进行文本生成、理解和推理。
*   **自然语言处理（NLP）：** 依赖PaddleNLP提供的丰富NLP能力，包括词向量、Transformer模型、序列标注等，支撑各种文本任务。
*   **LangChain框架：** 运用LangChain连接LLM和外部数据源（如向量数据库）或工具，实现复杂工作流，如信息检索增强生成（RAG）和代理（Agent）的工具调用。
*   **向量检索与语义匹配：** 通过将文档内容转化为向量表示，结合向量数据库进行高效的语义搜索和多路召回，从而从海量文本中快速定位相关信息。
*   **Prompt Engineering：** 通过设计有效的Prompt引导LLM生成期望的回答或执行特定任务。
*   **Pipeline机制：** 在PaddleNLP中通过Pipeline串联不同NLP组件，简化复杂任务的开发和部署。


- [chatbot文档单多轮问答](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/chatbot)
- [基于ReACT的agents](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/agents)
- [Paddle-ChatDocuments -](https://aistudio.baidu.com/aistudio/projectdetail/6195067)
- [ChatLLM-EVALUATION - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/6145966)
- [Paddle-ChatDocuments](https://github.com/thomas-yanxin/Paddle-ChatDocuments)
- [LangChain-ChatGLM-Webui](https://github.com/thomas-yanxin/LangChain-ChatGLM-Webui)

------------------------------------------------------------

## 1.跨模态模型minigpt4, speecht5

#### 简介
SpeechT5是百度飞桨PaddleNLP中集成的多模态预训练模型，专注于语音和文本的统一处理。作为PaddleNLP强大且易用的自然语言处理库的一部分，SpeechT5旨在通过单一模型处理文本到语音、语音到文本、文本到文本以及语音到语音等多种序列转换任务，从而在研究和工业应用中提供高效的NLP能力。

#### 核心功能
*   **多模态统一处理：** 能够同时处理和学习文本与语音数据，实现跨模态的统一表示。
*   **语音合成 (TTS)：** 将文本转换为自然流畅的语音。
*   **语音识别 (ASR)：** 将语音转换为文本。
*   **语音到语音转换 (S2S)：** 直接实现语音间的转换，如语音风格迁移、语音翻译等。
*   **文本到文本转换 (T2T)：** 支持传统的文本序列到序列任务。

#### 技术原理
SpeechT5的核心架构是一个共享的编码器-解码器网络。在该网络之前和之后，分别设置了针对特定模态（语音/文本）的预处理网络 (pre-nets) 和后处理网络 (post-nets)。
1.  **预处理 (Pre-nets)：** 对输入的语音或文本数据进行特征提取和模态适应性处理。
2.  **共享编码器-解码器：** 处理预处理后的数据，执行序列到序列的转换任务。其关键在于通过在文本到语音、语音到文本、文本到文本和语音到语音等混合数据集上进行预训练，使得模型能够学习到文本和语音共享的统一隐空间表示。
3.  **后处理 (Post-nets)：** 根据共享编码器-解码器的输出，生成目标模态（语音或文本）的最终结果。这种设计使得模型能够在一个统一框架内灵活地处理不同模态间的转换，实现多任务学习和知识共享。


- [speecht5](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/speecht5)

------------------------------------------------------------

## 2.miniGPT4

#### 简介
MiniGPT-4是一个旨在增强视觉-语言理解能力的多模态大模型，它通过将冻结的视觉编码器与冻结的大型语言模型（如Vicuna）对齐，以实现类GPT-4的多模态能力。该项目在PaddlePaddle生态系统中的PaddleNLP和PaddleMIX中得到了集成和支持，提供了相关的实现、指令和教程。Large Model Systems Organization (LMSYS)也致力于大模型系统开源与可扩展性研究，为MiniGPT-4等提供了重要的研究背景和社区支持。

#### 核心功能
*   **详细图像描述生成：** 能够对输入的图像进行细致的描述和理解。
*   **手写文本生成网站：** 具备从手写草稿直接生成网站代码或结构的能力。
*   **故事创作：** 基于视觉输入进行富有逻辑和想象力的故事创作。
*   **问题解决：** 结合图像信息，提供问题解答。
*   **烹饪指导生成：** 从图像中理解烹饪场景并给出指导。
*   **多模态对话：** 实现流畅且语义连贯的视觉-语言交互对话。
*   **多任务学习统一接口：** MiniGPT-v2在此基础上进一步发展，将LLM作为统一接口进行视觉-语言多任务学习。

#### 技术原理
MiniGPT-4的核心技术原理在于其两阶段的训练方法：
1.  **架构对齐：** 将一个冻结的视觉编码器（如BLIP-2的视觉部分）与一个冻结的大型语言模型（如Vicuna）通过一个单一的投影层（linear layer）进行对齐。这使得LLM能够“理解”视觉特征。
2.  **第一阶段预训练：** 在约500万对齐的图像-文本数据上进行预训练，此阶段主要训练投影层，使LLM能够从视觉特征中获取图像信息。这一阶段计算效率高，仅需少量计算资源（例如，单张A100 GPU约7分钟）。
3.  **第二阶段微调：** 为了解决第一阶段可能导致的LLM生成能力受损问题，通过高质量、精心对齐的对话数据集进行微调，使用会话模板来提升模型的生成质量和连贯性。这确保了模型不仅能理解图像，还能生成高质量、符合人类偏好的文本响应。



- [minigpt4-paddlenlp](https://github.com/PaddlePaddle/PaddleNLP/blob/3f5737a1a63907513242c37b600ac981fa2e0419/examples/multimodal/minigpt4/README.md)
- [PaddleNLP/paddle_minigpt4_instrction.md at 3f5737a1a63907513242c37b600ac981fa2e0419 · PaddlePaddle/PaddleNLP · GitHub](https://github.com/PaddlePaddle/PaddleNLP/blob/3f5737a1a63907513242c37b600ac981fa2e0419/examples/multimodal/minigpt4/paddle_minigpt4_instrction.md)
- [Vision-CAIR/MiniGPT-4: MiniGPT-4: Enhancing Vision-language Understanding with Advanced Large Language Models](https://github.com/Vision-CAIR/MiniGPT-4)
- [minigpt4 at develop · PaddlePaddle/PaddleNLP · GitHub](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/minigpt4)
- [lmsys (Large Model Systems Organization)](https://huggingface.co/lmsys)
- [论文：MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592)

------------------------------------------------------------

## 1.飞桨大语言模型:ChatGLM, GLM, Llama


#### 简介
PaddleNLP是一个基于飞桨深度学习框架的全面自然语言处理（NLP）和大语言模型（LLM）开发套件。它集成了多种参数高效微调（PEFT）方法，如LoRA和Prefix-Tuning，旨在提供高效、易用且性能卓越的工具，帮助开发者快速将大型预训练模型应用于特定任务和场景，同时支持从模型训练到推理的完整LLM工作流。

#### 核心功能
*   **参数高效微调（PEFT）支持**：提供LoRA和Prefix-Tuning等先进的PEFT技术，允许用户仅通过微调少量参数即可高效适配大型语言模型到特定任务，支持单卡及分布式训练。
*   **高效训练与推理**：支持在多种硬件上对大模型进行高效训练、无损压缩以及高性能推理，优化资源利用。
*   **丰富的模型与任务覆盖**：内置一个强大的模型库（Awesome Model Zoo），支持文本分类、神经搜索、问答、信息抽取、文档智能、情感分析等广泛的NLP任务。
*   **端到端LLM工作流**：提供从模型训练到部署的全流程示例和工具，简化大语言模型应用的开发过程。
*   **易用性与灵活性**：通过用户友好的API，允许开发者轻松定义模型、数据集和配置，快速启动PEFT微调。

#### 技术原理
*   **基于参数高效微调（PEFT）技术**：核心在于通过引入少量可训练参数（如适配器层、低秩矩阵或前缀向量），或仅更新现有模型参数的一小部分，而非对整个巨型模型进行全量微调，从而大幅降低计算和存储开销。例如，LoRA通过分解更新矩阵为两个低秩矩阵进行增量学习，Prefix-Tuning通过在输入序列前添加可学习的前缀向量来引导模型行为。
*   **飞桨深度学习框架赋能**：底层依赖PaddlePaddle的核心框架，利用其自动混合精度优化策略和分布式Fleet API，实现高效的并行计算和内存管理，支持4D混合并行策略，以处理大规模预训练模型的训练需求。
*   **模块化与可扩展架构**：PEFT模块作为PaddleNLP的一部分，其设计允许灵活集成不同的PEFT算法，并通过统一的接口进行调用，方便开发者扩展和定制。



- [微调技术：低参数微调能力PEFT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/peft)
- [Big Bird-Google](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/language_model/bigbird/README.md)

------------------------------------------------------------

## GLM

#### 简介
PaddleNLP是一个基于飞桨（PaddlePaddle）深度学习框架的自然语言处理（NLP）开发套件。它集成了易用性与强大功能，旨在帮助开发者高效地进行大语言模型（LLM）的训练、无损压缩和高性能推理，从而加速大模型在工业界的落地应用。该库涵盖了从学术研究到实际生产的广泛NLP任务支持，并特别提供了如GLM等主流语言模型的实现与应用示例。

#### 核心功能
*   **大语言模型（LLM）支持：** 提供全面的LLM开发、训练、压缩与推理解决方案。
*   **高效训练能力：** 支持大规模语言模型的高效训练，包括分布式训练优化。
*   **模型优化与压缩：** 具备模型无损压缩功能，以减小模型体积并提升运行效率。
*   **高性能推理：** 在多种硬件设备上（包括GPU、XPU）实现低延迟、高吞吐的推理性能。
*   **丰富模型库：** 内置“Awesome Model Zoo”，包含大量预训练模型。
*   **易用API与示例：** 提供用户友好的文本领域API和多场景应用示例（如GLM语言模型示例），简化开发流程。

#### 技术原理
*   **基于飞桨（PaddlePaddle）框架：** 深度集成PaddlePaddle的底层算子、并行计算和优化能力，确保高性能和稳定性。
*   **分布式训练技术：** 针对大模型训练需求，采用数据并行、模型并行等分布式策略，实现高效的模型扩展与训练。
*   **模型量化与剪枝：** 通过无损压缩技术（如量化、剪枝、知识蒸馏等）优化模型结构，降低内存占用和计算复杂度，同时保持模型性能。
*   **推理引擎优化：** 底层推理引擎针对不同硬件平台进行定制优化，可能利用算子融合、计算图优化等技术，最大化推理效率。
*   **Transformer架构支持：** 实现并优化了GLM等基于Transformer架构的语言模型，涉及多头自注意力机制、前馈网络、位置编码等关键组件。



- [GLM](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/glm)
- [General Language Model (GLM) 是以自回归填空作为训练目标的通用语言模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/glm)

------------------------------------------------------------

## LLaMA


#### 简介
PaddleNLP是一个基于百度飞桨（PaddlePaddle）深度学习框架的自然语言处理（NLP）和大语言模型（LLM）开发套件。它旨在提供用户友好的文本域API、多场景应用示例以及高性能的分布式训练能力。该库支持从训练到部署的完整LLM工作流程，特别是对LLaMA系列模型提供了全面的支持。

#### 核心功能
*   **多模型支持与推理优化：** 支持LLaMA、Qwen、DeepSeek、Mistral、ChatGLM、Bloom和Baichuan等主流LLM模型系列，并支持Weight Only INT8/INT4推理以及WAC（权重、激活、Cache KV）的INT8/FP8量化推理。
*   **模型训练与微调：** 提供大模型的预训练、精调（SFT）、参数高效微调（PEFT，如LoRA、Prefix Tuning等）、对齐（DPO/SimPO/ORPO/KTO）以及强化学习从人类反馈中学习（RLHF）等功能。
*   **灵活的量化支持：** 为LLaMA等模型提供多种量化类型支持，包括FP16/BF16、WINT8、WINT4、INT8-A8W8、FP8-A8W8和INT8-A8W8C8。
*   **端到端LLM工作流：** 覆盖从模型训练、调优到部署的整个生命周期，提供实用示例。

#### 技术原理
PaddleNLP的核心技术原理围绕高性能和高效的大模型处理展开。
*   **基于飞桨框架：** 利用PaddlePaddle的分布式训练能力和优化特性，实现对大规模模型的有效训练和推理。
*   **混合精度训练与量化：** 支持FP16/BF16等混合精度训练，以及INT8、INT4等多种量化技术（如Weight Only Quantization, WAC Quantization），显著降低模型内存占用和计算需求，提升推理速度。
*   **参数高效微调（PEFT）技术：** 集成LoRA、Prefix Tuning等PEFT方法，通过只训练少量参数来适应特定任务，大幅减少微调成本和计算资源。
*   **对齐与强化学习：** 支持DPO、RLHF等对齐方法，使模型输出更符合人类偏好和指令。
*   **模型并行与分布式训练：** 结合飞桨的分布式策略，实现超大模型的训练，突破单设备内存限制。


- [LLaMA inplementation](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/llama)
- [LLaMA](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/llama)

------------------------------------------------------------

## bloom

#### 简介
PaddleNLP 是百度飞桨（PaddlePaddle）深度学习框架下的自然语言处理（NLP）开发库，旨在提供易用且功能强大的文本领域API、多场景应用示例以及高性能分布式训练能力。它是一个大型语言模型（LLM）开发套件，支持在多种硬件上高效地进行大模型训练、无损压缩和高性能推理。BLOOM 是一个拥有 1760 亿参数的自回归大型语言模型，由 BigScience 协作项目开发，能够生成 46 种自然语言和 13 种编程语言的文本。PaddleNLP 对 BLOOM 模型提供了全面的支持，使其在飞桨生态中得以应用和优化。

#### 核心功能
*   **大模型推理支持：** PaddleNLP 支持对包括 BLOOM 在内的多种主流大语言模型（如 LLaMA、Qwen、ChatGLM、Baichuan等）进行高效推理。
*   **模型量化与优化：** 提供 Weight Only INT8/INT4 推理、WAC（权重、激活、Cache KV）INT8/FP8 量化等多种优化技术，以降低模型显存占用和提升推理速度。
*   **大模型训练与精调：** 支持大模型的预训练、精调（包括 SFT、PEFT 技术）和对齐，使得开发者可以根据特定任务需求对模型进行定制化训练。
*   **丰富的模型库与示例：** 包含 Transformer 模型库和广泛的 NLP 应用示例，方便开发者快速上手和解决问题。
*   **多语言和编程语言文本生成：** BLOOM 模型本身能够生成 46 种自然语言和 13 种编程语言的连贯文本。

#### 技术原理
*   **自回归（Autoregressive）模型：** BLOOM 是一种自回归语言模型，通过预测序列中的下一个词来生成文本，这种机制使其能够生成连贯且上下文相关的长文本。
*   **Transformer 架构：** BLOOM 基于 Transformer 神经网络架构，利用自注意力机制（Self-Attention Mechanism）有效捕捉文本中的长距离依赖关系。
*   **大规模参数与训练：** BLOOM 拥有 1760 亿参数，在大量文本数据上进行训练，使其具备强大的语言理解和生成能力。
*   **字节对编码（BPE）分词：** BLOOM 使用字节对编码（Byte Pair Encoding, BPE）作为其分词器，能够处理多种语言并有效压缩词汇表。
*   **量化技术：** PaddleNLP 采用 INT8、INT4 等低精度量化技术对 BLOOM 等大模型进行压缩，减少模型大小和计算量，同时通过 WAC（Weight, Activation, Cache KV）等策略保持精度。
*   **高性能分布式训练：** PaddleNLP 利用飞桨框架的分布式训练能力，支持在大规模计算资源上高效训练和部署大模型。



- [BLOOM是一种自回归大型语言模型(LLM)](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bloom)
- [BLOOM是一种自回归大型语言模型(LLM)](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/bloom)

------------------------------------------------------------

## chatglm




#### 简介
向百度飞桨（PaddlePaddle）深度学习框架下的自然语言处理开发库PaddleNLP中关于ChatGLM模型的示例和LLM实现。PaddleNLP是一个大型语言模型（LLM）开发套件，旨在提供高效的大模型训练、无损压缩以及在各种硬件设备上的高性能推理能力，致力于推动大模型在工业应用中的高效落地。因此，涉及的是如何在PaddleNLP框架下实现和应用ChatGLM系列大型语言模型。

#### 核心功能
*   **ChatGLM模型支持与集成：** 提供在PaddleNLP框架下对ChatGLM模型的完整支持，包括模型定义、权重加载、推理接口等。
*   **高效大模型训练：** 支持ChatGLM模型的高效分布式训练，利用PaddlePaddle的优势进行大规模模型迭代。
*   **模型无损压缩：** 针对ChatGLM模型提供多种压缩技术，如量化、剪枝等，以在保持性能的同时减小模型体积，实现无损或低损耗压缩。
*   **高性能推理：** 优化ChatGLM模型在多种硬件（如GPU、XPU等）上的推理速度和效率。
*   **端到端工作流示例：** 提供从训练到部署的完整工作流示例，帮助开发者快速上手ChatGLM模型的开发和应用。
*   **多场景应用支持：** 针对ChatGLM模型在不同自然语言处理任务中的应用提供指导和实现。

#### 技术原理
PaddleNLP对ChatGLM等LLM的支持，主要基于以下技术原理：
*   **PaddlePaddle深度学习框架：** 利用PaddlePaddle的动态图和静态图混合编程模式，以及其在分布式训练、优化器、算子库等方面的底层优势。
*   **分布式训练技术：** 采用模型并行（如Megatron-LM的张量并行、流水线并行）、数据并行、ZeRO等技术，以支持超大规模ChatGLM模型的训练。
*   **模型量化（Quantization）：** 通过将模型参数从浮点数转换为低精度整数（如FP16、Int8甚至FP8），显著减少模型大小和计算量，加速推理。
*   **模型剪枝（Pruning）：** 识别并移除ChatGLM模型中冗余的连接或神经元，在不显著降低性能的前提下精简模型。
*   **高效推理引擎：** 集成或优化推理引擎，如Paddle Inference，以实现ChatGLM模型在不同硬件上的极致推理性能，包括FlashAttention等注意力机制优化。
*   **Transformer架构优化：** 对ChatGLM所基于的Transformer架构进行深度优化，包括自注意力机制、前馈网络等部分的计算优化。

#### 应用场景
*   **智能对话系统：** 作为核心语言模型，构建具备多轮对话、情感理解、知识问答能力的智能客服、聊天机器人。
*   **内容生成：** 用于文章撰写、摘要生成、广告文案、剧本创作、代码生成等创意性文本生成任务。
*   **智能助手：** 赋能语音助手、编程助手等，提供更自然、准确的交互和信息服务。
*   **多语言处理：** 利用ChatGLM的多语言能力，应用于跨语言沟通、机器翻译、多语言信息检索等。
*   **教育与研究：** 作为大型语言模型学习和研究的平台，支持学术实验和新算法验证。
*   **定制化场景：** 通过微调，为特定行业或企业（如金融、医疗、法律）构建专业领域的ChatGLM应用，实现垂直领域的智能化。


- [ChatGLM-6B 是一个开源的、支持中英双语问答的对话语言模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/chatglm)
- [ChatGLM-6B](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/chatglm)

------------------------------------------------------------

## chatglm2

#### 简介
PaddleNLP是一个基于百度PaddlePaddle深度学习框架的自然语言处理开发库，旨在提高文本领域建模效率，支持大模型训练、推理、精调、对齐和量化。ChatGLM2-6B是由清华大学智源人工智能研究院（THUDM）开发的开源中英双语对话模型ChatGLM-6B的第二代版本，它在继承前代模型流畅对话和低部署门槛等优点的基础上，进一步提升了性能。

#### 核心功能
*   **PaddleNLP**:
    *   提供用户友好的文本领域API。
    *   支持多场景应用示例。
    *   具备高性能分布式训练能力。
    *   支持多种LLM模型的推理（如LLaMA、Qwen、ChatGLM系列等）、预训练、精调（SFT、PEFT）、对齐和量化（INT8、INT4、FP8）。
*   **ChatGLM2-6B**:
    *   开源中英双语对话能力。
    *   支持流畅的对话交互。
    *   实现较低的部署门槛。
    *   作为ChatGLM系列的第二代，在性能上有所增强。

#### 技术原理
*   **PaddleNLP**:
    *   基于PaddlePaddle深度学习框架。
    *   涉及大模型的预训练、精调（如SFT、PEFT）、对齐技术。
    *   支持多种量化技术，包括Weight Only INT8/INT4推理和WAC（权重、激活、Cache KV）INT8/FP8量化推理，以优化模型大小和推理速度。
    *   利用分布式训练技术加速大型模型的训练过程。
*   **ChatGLM2-6B**:
    *   作为Transformer架构的对话模型，专注于生成连贯且有逻辑的对话内容。
    *   其“6B”表示模型拥有60亿参数，具备强大的语言理解和生成能力。
    *   通过对前代模型ChatGLM-6B的优化，可能在模型结构、训练数据、训练策略等方面进行了改进，以提升其性能和稳定性。

#### 应用场景
*   **PaddleNLP**:
    *   大型语言模型（LLM）的开发、训练、部署和优化。
    *   自然语言处理领域的各种应用开发，如文本分类、情感分析、问答系统、机器翻译等。
    *   学术研究和工业界的大模型应用实践。
*   **ChatGLM2-6B**:
    *   构建智能客服、虚拟助手、聊天机器人等对话式AI产品。
    *   进行多语言（中英）对话交互。
    *   低成本部署的本地化大模型应用。
    *   作为基础模型进行二次开发和微调，应用于特定领域的对话任务。


- [ChatGLM2-6B](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/chatglm2)
- [THUDM/ChatGLM2-6B: ChatGLM2-6B: An Open Bilingual Chat LLM | 开源双语对话语言模型](https://github.com/THUDM/ChatGLM2-6B)

------------------------------------------------------------

## ernie-3.5

#### 简介
PaddleNLP 是一个基于百度飞桨（PaddlePaddle）深度学习框架的大语言模型（LLM）开发套件。它专注于支持大模型的训练、无损压缩和高性能推理，旨在助力开发者实现高效的工业级应用。ERNIE 3.5 是百度在该框架下推出的一款知识增强型基础大模型，在前代ERNIE 3.0的基础上，在模型效能、功能和性能上进行了大幅提升，尤其在创意写作、问答、推理和代码生成等核心能力方面表现突出，并显著优化了推理成本。

#### 核心功能
1.  **高效的大模型训练与推理**: 支持在多种硬件设备上进行高效、大规模的模型训练，实现模型无损压缩，并提供高性能推理能力。
2.  **知识增强与检索增强**: 核心功能之一是其知识增强（Knowledge Enhancement）和检索增强（Retrieval Enhancement）技术，特别是引入了“知识片段增强”（Knowledge Snippet Enhancement），通过识别用户查询中的关键知识片段，并结合知识图谱和搜索引擎进行信息检索，从而生成更准确、更丰富的回复。
3.  **多模态与多任务支持**: ERNIE系列模型不仅涵盖自然语言处理，还扩展到视觉和跨模态领域。PaddleNLP作为一个整体套件，支持广泛的NLP任务，包括文本分类、神经搜索、问答、信息抽取、文档智能、情感分析等。
4.  **性能与成本优化**: 通过自适应混合并行训练技术和混合精度计算，大幅提升了模型训练速度和推理性能，显著降低了运行成本，使大模型更具经济性。

#### 技术原理
1.  **基于Transformer的编码器架构**: ERNIE 3.5 构建于先进的Transformer编码器架构之上，该架构包含多层隐藏层、多头注意力机制，并支持多种非线性激活函数（如GELU、ReLU、SiLU），使其能够有效捕捉文本中的复杂语义关系。
2.  **知识驱动的预训练范式**: 模型通过深度学习整合海量数据和知识，实现知识增强的预训练，使其具备强大的解释性和知识利用能力。其独特的“知识片段增强”技术是关键，它允许模型在处理用户查询时动态地识别、检索并整合外部知识片段，以提升模型的准确性和知识深度。
3.  **自适应混合并行训练**: 为应对大模型训练的巨大计算需求，ERNIE 3.5 采用了自适应混合并行训练技术，优化了模型在分布式计算环境下的训练效率，确保了模型的快速迭代。
4.  **混合精度计算**: 结合混合精度计算策略，在保证模型精度的同时，有效降低了计算资源消耗和训练时间，进一步提升了训练的经济性和效率。

#### 应用场景
1.  **智能内容创作**: 在创意写作、文章生成、摘要总结等领域展现出强大的能力。
2.  **智能问答系统**: 可用于开发高度智能的问答机器人，提供准确、连贯的回答，支持复杂推理和知识查询。
3.  **辅助编程与代码生成**: 能够理解编程语言并生成代码，辅助软件开发人员提升效率。
4.  **企业级NLP解决方案**: 广泛应用于各类行业，如金融、能源、媒体、汽车等，实现文本分类、信息抽取、情感分析等企业级自然语言处理需求，推动行业智能化转型。
5.  **学术研究与模型开发**: 作为PaddleNLP开发套件的一部分，为研究者和开发者提供易用且强大的工具，支持新型大模型的探索、训练与部署。


- [ERNIE-3.5-SE](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/ernie-3.5-se/README.md)

------------------------------------------------------------

## gpt-3


#### 简介
PaddleNLP是一个基于百度飞桨（PaddlePaddle）深度学习框架的强大且易用的自然语言处理（NLP）和大型语言模型（LLM）开发套件。它集成了丰富的模型库，旨在支持从研究到工业应用的广泛NLP任务，并专注于提供高效的大模型训练、无损压缩和高性能推理能力，以助力开发者实现产业级的大模型应用。该套件特别提到了对GPT-3等多种主流大型语言模型的支持。

#### 核心功能
*   **多任务支持：** 支持文本分类、神经搜索、问答系统、信息抽取、文档智能、情感分析等多种NLP任务。
*   **大型模型预训练：** 提供LLaMA/LLaMA2/LLaMA3、GPT-3、Qwen、Baichuan/Baichuan2、Mixtral等主流大模型的预训练能力，并持续更新支持更多模型。
*   **高效训练与推理：** 支持在多种硬件设备上进行高效的大模型训练、无损压缩和高性能推理。
*   **分布式训练：** 将飞桨的4D并行策略集成到Trainer API中，用户可通过修改配置轻松部署不同的分布式训练策略。
*   **模型统一实现与管理：** 对GPT等模型实现进行统一，并提供便捷的模型配置切换，实现“一键运行”不同模型。

#### 技术原理
PaddleNLP的LLM模块底层依赖于PaddlePaddle深度学习框架，其核心技术原理包括：
*   **4D并行策略：** 采用数据并行（Data Parallelism）、模型并行（Model Parallelism）、流水线并行（Pipeline Parallelism）和混合并行等多种并行策略，以实现超大规模模型的分布式高效训练。
*   **Trainer API：** 通过抽象的Trainer API简化了分布式训练的配置和执行，将复杂的并行策略封装起来，提升用户体验。
*   **预训练与微调机制：** 提供完善的数据准备、预训练和自定义数据集支持，以及性能测试报告，确保模型训练的有效性和可复现性。
*   **高效推理优化：** 针对大模型的推理性能进行了优化，支持无损压缩和高吞吐、低延迟的推理部署。

#### 应用场景
*   **学术研究：** 为研究人员提供易于使用的LLM开发环境，加速模型探索和算法创新。
*   **工业应用：** 适用于企业级的大型模型部署，如构建智能客服、内容生成、智能推荐系统等。
*   **自然语言理解：** 应用于各类文本分析任务，如情感倾向分析、实体识别、关系抽取等。
*   **信息检索与问答：** 支撑构建语义搜索系统和智能问答机器人。
*   **内容创作辅助：** 利用生成式LLM能力辅助文章撰写、代码生成、摘要提取等。
*   **特定领域模型开发：** 基于现有预训练模型进行微调，快速构建特定行业或业务领域的专业模型。


- [GPT-3](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/gpt-3)
- [llm/gpt-3/README.md at develop · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/llm/gpt-3/README.md)

------------------------------------------------------------

## opt

#### 简介
PaddleNLP是一个基于百度飞桨（PaddlePaddle）深度学习框架的大语言模型（LLM）开发套件。它旨在提供高效的大模型训练、无损压缩以及在多种硬件设备上的高性能推理能力，致力于帮助开发者实现大模型的工业级高效应用。

#### 核心功能
*   **大模型开发与管理**: 支持LLM和SLM（Small Language Model）的开发，提供易于使用的文本领域API。
*   **高效训练**: 实现大型模型的分布式训练，提升模型训练效率。
*   **模型优化**: 支持模型无损压缩和高性能推理。
*   **硬件兼容性**: 能够在多种硬件设备上进行高性能推理。
*   **应用示例丰富**: 提供多场景的应用示例，方便开发者快速上手和应用。

#### 技术原理
*   **基于PaddlePaddle框架**: 构建于百度飞桨深度学习框架之上，继承其高性能和灵活性。
*   **分布式训练**: 采用分布式训练机制，以应对大模型对计算资源的需求，提高训练速度。
*   **模型网络解耦**: 支持将模型网络与特定硬件（如XPU）的GPU部分分离，以优化性能和兼容性。
*   **算子优化**: 可能包含针对特定硬件（如XPU）的算子吸收（absorb mla）和融合块注意力（fused block attn）等底层优化技术，以提升计算效率。
*   **推理优化**: 通过特定的参数推断（infer param）和输出保存机制，实现高效的模型推理。


- [OPT](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/llm/opt)

------------------------------------------------------------

## 2.RLHF


#### 简介
PaddleNLP 是一个基于飞桨（PaddlePaddle）深度学习框架的大语言模型（LLM）开发套件，专注于自然语言处理领域。它提供了一套易用且功能强大的工具，旨在支持大模型的训练、压缩和高性能推理。其 RLHF（Reinforcement Learning with Human Feedback，人类反馈强化学习）示例是PaddleNLP在大模型对齐方面的一个重要实现，旨在通过结合人类偏好来优化大型语言模型的表现，使其输出更符合人类价值观和指令意图。

#### 核心功能
*   **人类偏好对齐LLM**：通过强化学习PPO（Proximal Policy Optimization）算法，使大型语言模型学习并对齐人类的偏好，生成更安全、有用和符合预期的内容。
*   **奖励模型微调**：支持使用人类偏好数据集（如PKU-Alignment/PKU-SafeRLHF-30K）来训练和微调奖励模型，该模型用于评估生成内容的质量。
*   **分布式训练支持**：集成3D分布式并行训练技术，以支持大规模模型的高效训练。
*   **生成加速**：在Rollout阶段利用预测优化技术，实现模型生成的加速。
*   **多样化数据处理**：支持多种内置数据集（如PKU-SafeRLHF、Alpaca），并提供灵活的配置接口，允许使用多数据集及指定采样比例进行训练。

#### 技术原理
PaddleNLP的RLHF实现主要基于强化学习中的PPO算法。其核心流程通常包括以下几个阶段：
1.  **监督微调 (SFT)**：初始阶段对预训练语言模型进行监督微调，使其能够按照指令生成初步的文本。
2.  **奖励模型 (RM) 训练**：利用人类偏好数据集（包含同一Prompt下不同Response的偏好标注），训练一个奖励模型。该模型能够对任意给定的文本序列进行评分，反映其符合人类偏好的程度。通常采用二元比较或者评分的方式收集数据。
3.  **强化学习 (RL)**：以奖励模型作为强化学习环境中的奖励函数，使用PPO算法对SFT后的模型进行优化。在PPO阶段，模型会根据Prompt生成Response，奖励模型对其评分，然后模型根据评分进行策略更新，以最大化奖励。
*   **PPO算法**：通过限制策略更新的步长，确保训练的稳定性和效率。它在策略网络和价值网络之间进行迭代优化，以找到最优策略。
*   **3D并行训练**：结合数据并行、模型并行和张量并行等技术，有效利用多GPU/多机资源，解决大模型训练的内存和计算瓶颈。
*   **Rollout阶段的预测优化**：在模型生成（Rollout）过程中，通过优化预测策略或使用加速技术，提高生成效率。


- [RLHF PPO](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/RLHF)

------------------------------------------------------------


# 1.PaddleMIX跨模态大模型

#### 简介
PaddleMIX是百度飞桨（PaddlePaddle）生态下的一个多模态大模型开发套件。它旨在集成图像、文本、视频等多种模态，为多模态任务提供开箱即用的开发体验，并支持灵活定制，以高效完成各类多模态大模型任务。该套件覆盖了从预训练、微调到生成和理解的广泛多模态应用，致力于助力通用人工智能的探索。

#### 核心功能
*   **多模态融合与处理：** 聚合图像、文本、视频等多种异构数据模态。
*   **多模态任务覆盖：** 支持视觉-语言预训练（VLP）、多模态微调、文本到图像生成（Text-to-Image）、文本到视频生成（Text-to-Video）以及多模态理解等广泛任务。
*   **模型与工具箱集成：** 内置端到端的大规模多模态预训练模型和扩散模型工具箱。
*   **开发与部署支持：** 提供开箱即用的开发环境，并支持灵活定制，以满足不同应用场景的需求。
*   **检索任务示例：** 包含多模态检索任务示例，如文本到图像相似度计算和信息检索工作流。

#### 技术原理
PaddleMIX基于飞桨深度学习框架构建，其核心技术原理在于融合处理不同模态的数据。通过深度学习模型，如Transformer架构及其变体，实现跨模态信息的对齐、融合和表示学习。
*   **跨模态预训练：** 利用海量多模态数据进行预训练，使模型学习到通用的跨模态表示能力，如视觉-语言预训练（VLP），通过共同嵌入空间理解图像和文本的关系。
*   **生成对抗网络 (GANs) / 扩散模型 (Diffusion Models)：** 在多模态生成任务中，如Text-to-Image和Text-to-Video，采用先进的生成模型，通过迭代优化和去噪过程，从文本描述生成高质量的视觉内容。
*   **注意力机制：** 广泛应用于模型内部，使模型能够关注不同模态数据中的关键信息，进行有效的特征提取和融合。
*   **高效计算与优化：** 依托飞桨框架的优势，提供高性能和灵活性的计算支持，优化模型训练和推理效率。

#### 应用场景
*   **图像摘要与描述：** 基于图像内容自动生成简洁的文本摘要或详细描述。
*   **视频问答系统：** 对视频内容进行理解，并回答与视频相关的用户提问。
*   **动画制作与内容生成：** 根据文本指令或概念生成图像和视频内容，辅助动画、设计和创意产业。
*   **多模态检索：** 实现跨模态的信息检索，例如使用文本查询来搜索相关的图像或视频。
*   **智能助手与交互：** 为具备多模态理解能力的智能助手提供技术支持，实现更自然的人机交互。
*   **教育与知识传播：** 将复杂的视觉信息转化为易于理解的文本，或将抽象的概念通过多媒体形式进行阐释。


- [跨模态大模型开发套件paddlemix（扩散类+多模态）](https://github.com/PaddlePaddle/PaddleMIX)

------------------------------------------------------------



# 1.PaddleRec推荐系统:点击预测DeepCTR等

#### 简介
本内容综合介绍了百度飞桨（PaddlePaddle）生态下的多个核心组件，包括面向人工智能学习与实践的一站式云平台AI Studio、大规模推荐算法库PaddleRec，以及基于深度学习的点击率（CTR）模型工具DeepCTR。这些工具共同致力于简化AI模型的开发、训练、部署流程，特别是聚焦于推荐系统领域，提供从数据处理到模型服务全链路的解决方案，并支持各类经典及前沿的深度学习推荐算法。

#### 核心功能
*   **AI Studio**: 提供在线编程环境、免费GPU/CPU算力、海量开源算法、数据集以及AI课程，支持模型的在线构建、训练和部署，并具备完善的教学管理系统。
*   **PaddleRec**: 作为飞桨深度学习平台上的推荐系统框架，涵盖了协同过滤到深度学习的多种推荐算法，支持内容理解、匹配、召回、排序等全链路技术，并提供端到端解决方案和分布式训练能力。
*   **DeepCTR**: 易用、模块化且可扩展的深度学习点击率模型库，提供了丰富的核心组件层，用户可轻松构建和定制复杂的CTR预测模型，如DeepFM等。
*   **PaddlePaddle Models**: 提供飞桨官方及社区贡献的大量产业级和学术前沿模型，覆盖视觉、NLP、推荐等多个领域，支持高效的大模型训练和高性能推理。

#### 技术原理
该生态体系基于百度自主研发的深度学习框架飞桨（PaddlePaddle）构建。技术原理主要体现在：
*   **深度学习模型**: 广泛应用各类深度神经网络（如DeepFM等）进行特征交叉、用户行为建模、CTR预测和排序。这些模型能够处理大规模稀疏特征数据。
*   **分布式训练**: 针对大规模推荐系统数据和复杂模型训练需求，PaddleRec支持离线分布式训练，利用多GPU/CPU集群加速模型迭代。
*   **模块化与可扩展性**: DeepCTR通过提供可重用的组件层，使得用户能够灵活组合构建新的模型结构，提高了模型开发效率。
*   **全链路覆盖**: 从数据源的日志处理、特征工程、模型训练到线上推理服务（如通过Serving部署），形成了端到端的推荐系统解决方案，实现了数据、模型、服务的闭环。
*   **云计算支持**: AI Studio依托云端计算资源，提供便捷的开发环境和强大的算力支持，降低了AI开发的门槛。

#### 应用场景
*   **个性化推荐**: 广泛应用于电商、内容平台（新闻、视频、音乐）、社交媒体等领域，用于为用户提供精准的商品、内容或服务推荐，提升用户体验和转化率。
*   **广告点击率预测**: 在在线广告投放中，用于准确预测用户点击广告的可能性，优化广告排序和投放策略，提高广告收益。
*   **AI教育与科研**: AI Studio为学生和研究人员提供强大的学习与实验平台，支持深度学习课程教学、项目实训和算法研究。
*   **企业级AI解决方案**: 飞桨模型库和PaddleRec可作为企业构建智能推荐、智能营销等AI应用的基础，满足其在不同行业场景下的定制化需求。


- [PaddleFleet-搜索推荐算法大全](https://aistudio.baidu.com/personalcenter/thirdview/940489)
- [PaddleRec: 包含推荐系统经典及最新算法LR、Wide&Deep、DSSM、TDM、MIND、Word2Vec、Bert4Rec、DeepWalk、SSR、AITM，DSIN，SIGN，IPREC、GRU4Rec、Youtube_dnn、NCF、GNN、FM、FFM、DeepFM、DCN、DIN、DIEN、DLRM、MMOE、PLE、ESMM、ESCMM, MAML、xDeepFM、DeepFEFM、NFM、AFM、RALM、DMR、GateNet、NAML、DIFM、Deep Crossing、PNN、BST、AutoInt、FGCNN、FLEN、Fibinet、ListWise、DeepRec、ENSFM，TiSAS，AutoFIS等，包含经典推荐系统数据集criteo 、movielens等](https://github.com/PaddlePaddle/PaddleRec)
- [推荐系统技术文档主页¶](https://paddlerec.readthedocs.io/en/latest/)
- [推荐全流程](https://github.com/PaddlePaddle/PaddleRec/blob/master/doc/whole_process.md)
- [DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models .](https://github.com/shenweichen/DeepCTR)
- [！社区模型--推荐系统](https://github.com/PaddlePaddle/models/blob/release%2F2.3/README.md)
- [AI 快车道-推荐系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/course/introduce/793)
- [基于deepFM模型的点击率预估模型](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/deepfm)

------------------------------------------------------------

## ElasticCTR预估任务完整方案

#### 简介
ElasticCTR（飞桨弹性计算推荐系统）是一个基于Kubernetes的企业级开源推荐系统解决方案。它整合了百度业务场景下打磨的高精度CTR（点击率预估）模型、飞桨框架的大规模分布式训练能力，以及工业级稀疏参数弹性调度服务。该方案旨在帮助用户在Kubernetes环境中一键部署推荐系统，并具备高性能、工业级部署和端到端体验等特点，同时作为一个开源套件，支持二次深度开发。

#### 核心功能
*   **一键部署与快速部署：** 支持在Kubernetes环境中快速部署CTR预估任务和Serving流程，用户只需配置数据源和样本格式即可完成训练与预测。
*   **高性能训练与预测：** 采用PaddlePaddle的全异步分布式训练方式，实现近乎线性的扩展能力，大幅节省训练资源。在线服务方面，利用Paddle Serving中高吞吐、低延迟的稀疏参数预估引擎，在高并发条件下表现优异。
*   **弹性计算与调度：** 具备工业级稀疏参数弹性调度服务，优化资源利用。
*   **端到端解决方案：** 提供从训练到预测的完整流程，并支持用户进行二次定制和开发。
*   **高精度CTR模型：** 融合了经过百度业务场景验证的高精度CTR模型。

#### 技术原理
ElasticCTR的核心技术原理主要包括：
*   **Kubernetes作为基础架构：** 整个推荐系统部署和运行在Kubernetes集群上，利用其容器编排、资源调度和弹性伸缩能力，实现系统的可扩展性和高可用性。
*   **PaddlePaddle分布式训练：** 采用飞桨（PaddlePaddle）深度学习框架进行模型训练，特别是利用其全异步分布式训练机制。这种机制允许不同计算节点之间异步通信和参数更新，从而提升训练效率和资源利用率，实现近乎线性的扩展性。
*   **稀疏参数处理与优化：** 推荐系统常涉及海量稀疏特征，ElasticCTR集成了工业级的稀疏参数弹性调度服务，高效管理和访问稀疏参数，解决大规模稀疏特征带来的存储和计算挑战。
*   **Paddle Serving在线服务：** 在线预测阶段，利用Paddle Serving作为推理引擎。Paddle Serving提供高吞吐、低延迟的服务能力，特别是其稀疏参数预估引擎，能在大并发场景下实现高效的实时预估，性能远超常见开源组件。
*   **CTR预估模型：** 方案内置并融合了在实际业务场景中不断优化的高精度CTR预估模型，通过学习用户行为和物品特征，预测用户点击的可能性。


- [ElasticCTR是分布式训练CTR预估任务和Serving流程一键部署的方案](https://github.com/PaddlePaddle/ElasticCTR)
- [ElasticCTR: ElasticCTR是分布式训练CTR预估任务和Serving流程一键部署的方案，用户只需配置数据源、样本格式即可完成一系列的训练与预测任务](https://gitee.com/paddlepaddle/elasticctr)

------------------------------------------------------------

## 推荐项目整理


- [基于 Milvus 和 MIND 算法的商品召回 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2250360?channelType=0&channel=0)
- [一点资讯技术编程大赛CTR赛道TOP1代码复现 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2390780?channelType=0&channel=0)
- [论文复现赛: 基于PaddleRec 24小时快速复现经典 CTR 预估算法 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2263714?channelType=0&channel=0)
- [用户购买预测paddlerec实现baseline - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1189943)
- [PaddleRec公开教程V2 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1268461)
- [告别电影荒，手把手教你训练符合自己口味的私人电影推荐助手 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1481839)
- [乘风破浪的调参侠！玩转特征重要性～从此精通LR - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/618918)
- [PaddleRec公开教程V3 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1431523)
- [飞桨AI Studio - 人工智能学习与实训社区](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/337565)
- [PaddleRec与Milvus深度结合，手把手带你体验工业级推荐系统召回速度 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1816335?channelType=0&channel=0)

------------------------------------------------------------


## 1.Pipelines：语义检索系统


#### 简介
百度文心（ERNIE）系列大模型是百度推出的一系列AI基础模型，其中ERNIE-Search专注于提升检索领域的效果，而文心百中则是一款基于ERNIE大模型构建的产业级智能搜索系统。这些技术共同构成了百度在AI搜索和通用大模型领域的布局，旨在提供高效、智能的语义理解和信息检索能力。

#### 核心功能
*   **ERNIE-Search**: 通过自蒸馏策略优化预训练模型，提升在检索任务中的性能和模型效果，旨在更准确地理解和响应用户检索意图。
*   **文心百中**:
    *   快速构建行业级专属AI搜索：支持用户通过简单步骤搭建定制化的在线搜索引擎。
    *   强大的语义理解能力：基于文心ERNIE大模型，实现对文本、视频、结构化数据等多元格式内容的深度语义理解。
    *   智能问答与信息抽取：能够从海量数据中精准回答问题并提取关键信息。
    *   降低成本：采用纯神经搜索架构，大幅减少传统搜索系统所需的人力成本。
    *   少样本学习：仅需少量数据即可在不同行业优化搜索结果。

#### 技术原理
*   **ERNIE-Search**: 采用预训练阶段细粒度交互向粗粒度交互的自蒸馏策略。这使得模型在训练过程中能够自我优化，提高在检索任务上的表现力，同时避免了传统方法中训练复杂教师模型的开销。其核心是深度学习和大规模预训练模型在检索领域的应用。
*   **文心百中**: 基于文心（ERNIE）大模型，特别是其在语义理解方面的突破性进展。系统采用“纯神经搜索架构”，摒弃了传统搜索引擎中复杂的逻辑和规则，转而通过大规模数据驱动的模型优化方式进行信息检索。利用ERNIE大模型的“少样本学习”能力，允许系统在仅有少量行业数据的情况下进行模型优化和效果提升。系统开发基于PaddlePaddle深度学习框架。

#### 应用场景
*   **企业内部知识管理**: 快速搭建企业级智能问答系统，提升员工获取内部资料和业务知识的效率。
*   **行业垂直搜索**: 为特定行业（如金融、医疗、教育、制造等）定制专业搜索引擎，实现精准信息检索和行业报告生成。
*   **内容平台智能推荐**: 为新闻、视频、电商等内容平台提供更智能的搜索和推荐服务，提高用户体验和内容分发效率。
*   **客户服务与支持**: 赋能智能客服，实现对用户问题的快速准确响应，提升服务质量和效率。
*   **数据分析与洞察**: 从海量非结构化数据中抽取关键信息，辅助决策和商业智能分析。


- [ERNIE-Search](https://wenxin.baidu.com/wenxin/modelbasedetail/ernie_search/)
- [文心百中 — 大模型驱动的产业级搜索系统](https://wenxin.baidu.com/baizhong/knowledgesearch/)
- [文心百中 — 大模型驱动的产业级搜索系统](https://wenxin.baidu.com/baizhong/index)
- [文心百中 — 帮助文档](https://wenxin.baidu.com/baizhong/doc/?id=Ylaqkc6qb)
- [PaddleNLP Pipelines：智能文本产线](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines)
- [In-batch negatives语义样本构建方式](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/semantic_indexing/README.md)
- [1.手把手搭建一个语义检索系统：neural_search](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/neural_search)
- [Segmentation fault (core dumped)](https://github.com/PaddlePaddle/PaddleNLP/issues/4988)
- [FAQ常见问题汇总](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/FAQ.md)
- [Pipelines API的预置模型介绍](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/API.md)
- [[Question]: 手把手搭建一个语义检索系统的问题 · Issue #3304 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/3304)
- [用户基于Pipelines做的网站，快捷的工程规范AI搜索引擎](https://xungui365.com/)
- [ERNIE 3.0 的RocketQA模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/benchmarks/README.md)

------------------------------------------------------------

## AIstudio项目

#### 简介
飞桨AI Studio星河社区是百度面向人工智能学习者和开发者提供的一站式学习、实训与开发平台。它集成了丰富的免费AI课程、大模型社区及应用、深度学习样例项目、经典数据集以及云端GPU算力资源。该平台旨在为用户提供自由灵活的AI应用创作与展示环境，并支持快速搭建如文本语义检索系统等AI解决方案。

#### 核心功能
*   **AI学习与实训：** 提供AI课程、大模型社区、样例项目、数据集等，支持用户进行AI技能学习与实践。
*   **AI应用开发与部署：** 支持基于Gradio、Streamlit或Static创建可视化交互式AI应用，并提供在线实例创建环境。
*   **文本语义检索系统搭建：** 提供PaddleNLP Pipelines，支持用户快速搭建端到端的语义检索系统，包括语料建库、召回、排序、模型调优、预测部署及前端UI等环节。
*   **资源与算力支持：** 提供云端超强GPU算力与存储资源，支持AI模型训练与运行。

#### 技术原理
该平台的核心技术原理主要围绕百度飞桨（PaddlePaddle）深度学习框架展开，并结合自然语言处理（NLP）和信息检索技术。
*   **深度学习框架：** 利用飞桨（PaddlePaddle）作为底层深度学习框架，支持各种AI模型的训练、开发与部署。
*   **语义检索：** 采用PaddleNLP Pipelines构建语义检索系统，其核心包括：
    *   **语义表示学习：** 通过深度学习模型将文本转化为高维向量表示（Embedding），捕捉文本的语义信息。
    *   **召回机制：** 通常采用向量检索技术，如Milvus等高效向量数据库，实现基于语义相似度的快速召回。
    *   **排序算法：** 对召回结果进行二次排序，通过更复杂的模型（如交叉编码器或双塔模型）进一步优化相关性。
    *   **模型组网与优化：** 涉及神经网络模型的架构设计、损失函数优化以及针对特定任务的数据增强和模型微调。
*   **交互式应用框架：** 支持Gradio、Streamlit等框架，通过Web技术实现AI模型的快速可视化交互界面，降低开发门槛。

#### 应用场景
*   **学术研究与教育：** AI学习者和学生可以通过平台进行AI知识学习、项目实践和技能提升。
*   **企业级智能搜索：** 用于文献检索、企业内部知识库搜索、商品搜索等场景，提升信息检索效率和准确性。
*   **内容推荐系统：** 如短视频推荐、新闻资讯推荐等，通过语义匹配提供个性化内容。
*   **智能客服与问答系统：** 构建基于语义理解的智能问答系统，提升客户服务体验。
*   **AI产品原型开发：** 开发者可以利用其快速搭建并展示AI应用原型。
*   **大规模文本分类：** 对海量文本进行高效准确的分类处理。


- [2022.PaddleNLP Pipelines带你十分钟搭建检索系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4442670)
- [2021动手搭建一套端到端文本语义检索系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3351784)

------------------------------------------------------------

## 语义检索系统


- [1.简单！PaddleNLP带你十分钟搭建检索系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4442670?channelType=0&channel=0)
- [2.端到端语义检索系统：semantic-search](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines/examples/semantic-search)
- [2.WINDOWS环境下搭建端到端语义检索系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Install_windows.md)
- [2.端到端两路召回语义检索系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Multi_Recall.md)
- [2.Neural Search](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/semantic-search/Neural_Search.md)

------------------------------------------------------------


# 1.信息抽取（关系、实体、序列标注、关键词抽取）


- [魔塔-零样本信息抽取](https://www.modelscope.cn/topic/9ae88b6a1ffd4de59a9f1948314ebc2b/pub/summary)
- [学术模型：【关系抽取、多文档摘要，稠密段落检索】](https://github.com/PaddlePaddle/models/tree/release/2.3/research#%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86)
- [学术模型【关系抽取、文档级抽取，知识图谱表示学习】](https://github.com/PaddlePaddle/models/tree/release/2.3/research#%E7%9F%A5%E8%AF%86%E5%9B%BE%E8%B0%B1)
- [！ERNIE 1.0-Large-CW实体抽取、分类问题：ERNIE1.0改成3.0即可](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-1.0)
- [ERNIE-M 是面向多语言建模的预训练](https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-m/README_zh.md)
- [序列标注](https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/applications/tasks/sequence_labeling)
- [LIC2021 DuEE 事件抽取基线](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/DuEE/README.md)
- [使用PaddleNLP完成中文命名实体识别](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/msra_ner/README.md)
- [快递单信息抽取 (Waybill Information Extraction)](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/waybill_ie/README.md)
- [序列标注](https://github.com/PaddlePaddle/ERNIE/blob/ernie-kit-open-v1.0/applications/tasks/sequence_labeling/README.md)
- [MPM论坛评论建议挖掘](https://github.com/PaddlePaddle/Research/blob/master/NLP/NAACL2019-MPM/README.md)
- [论文复现LUKE_paddle_stable](https://github.com/Beacontownfc/paddle_luke_stable)

------------------------------------------------------------

## 关系抽取&多对多实体抽取



- [！多对多信息抽取](https://github.com/PaddlePaddle/ERNIE/blob/ernie-kit-open-v1.0/applications/tasks/information_extraction_many_to_many/README.md)
- [信息抽取：多对多关系&实体属性抽取](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/0l57wvrj7)
- [信息抽取：一对一关系抽取](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/rl57wu7yl)
- [！ERNIEKit关系预测任务](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/wl580a1c1)
- [DocRED模型排行榜](https://competitions.codalab.org/competitions/20717#results)
- [文档级关系抽取SSAN](https://github.com/PaddlePaddle/Research/tree/master/KG/AAAI2021_SSAN)
- [文档级DocuNet: Code and dataset for the IJCAI 2021 paper "Document-level Relation Extraction as Semantic Segmentation".](https://github.com/zjunlp/DocuNet)
- [文档级ATLOP: Source code for paper "Document-Level Relation Extraction with Adaptive Thresholding and Localized Context Pooling", AAAI 2021](https://github.com/wzhouad/ATLOP)
- [基于预训练模型完成实体关系抽取](https://aistudio.baidu.com/aistudio/projectdetail/1639963)
- [LIC2021 DuIE 关系抽取基线](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/DuIE/README.md)
- [CCKS 2022 通用信息抽取 -- 基于UIE的基线系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/information_extraction/DuUIE/README.md)
- [ARNOR关系数据集](https://github.com/PaddlePaddle/Research/blob/master/NLP/ACL2019-ARNOR/README.md)

------------------------------------------------------------

## 文档级数据集


#### 简介
本汇总涵盖了人工智能，特别是自然语言处理（NLP）和信息抽取领域的重要数据集与研究项目。这些资源主要围绕关系抽取、命名实体识别、事件抽取、文档级信息抽取、生物医学信息抽取以及科学文献分析等任务，提供了从大规模公开数据集到特定领域（如生物医学、科学论文）的多种数据资源和相关工具，旨在推动相关领域的技术发展和性能评估。

#### 核心功能
这些数据集和项目提供了以下核心功能：
*   **关系抽取 (Relation Extraction):** 提供用于识别文本中实体间语义关系的数据集，如TACRED、DocRED和FewRel，支持对句子级和文档级关系的学习与抽取。
*   **命名实体识别 (Named Entity Recognition - NER):** 包含用于识别生物医学、化学等领域特定命名实体的数据集，如JNLPBA。
*   **信息抽取 (Information Extraction - IE):** 涵盖了从非结构化文本中抽取出结构化信息的任务，包括文档级IE (SciREX) 和科学IE (SciIE)。
*   **事件抽取 (Event Extraction):** 支持从文本中识别事件及其论元的任务，例如DWIE数据集。
*   **对话关系抽取 (Dialog Relation Extraction):** 专注于从对话文本中抽取实体关系，如DialogRE。
*   **数据集管理与评估:** 提供了SOTA数据集的汇总平台，方便研究者查找、比较不同任务的最佳性能数据集。
*   **图神经网络应用:** 探索基于图结构的神经网络在信息抽取任务中的应用，例如Edge-Oriented Graph项目。

#### 技术原理
所涉及的技术原理主要围绕深度学习和自然语言处理，包括：
*   **深度神经网络 (DNN):** 广泛应用于关系抽取、NER和事件抽取等任务，通过捕捉文本的复杂模式学习语义表示。
*   **图神经网络 (GNN):** 特别是边缘导向图（Edge-Oriented Graph）等，用于建模实体间关系和文档级信息，能够捕捉非局部依赖和复杂结构信息。
*   **预训练语言模型 (PLM):** 通常作为特征提取器或微调基础模型，提升各类NLP任务的性能。
*   **远程监督 (Distant Supervision):** 一种常用的弱监督方法，通过知识库自动标注训练数据，以解决大规模标注数据匮乏的问题，如TACRED和DocRED的构建。
*   **小样本学习 (Few-shot Learning):** 针对数据稀缺场景，通过少量样本进行高效学习，如FewRel数据集旨在推动此方向研究。
*   **实体和关系表示学习:** 将文本中的实体和关系映射到低维向量空间，以便模型进行计算和推理。
*   **特定领域知识融合:** 在生物医学、科学文献等领域，结合领域词典、本体或领域知识图谱来增强信息抽取的准确性。

#### 应用场景
这些数据集和研究成果的应用场景广泛，主要包括：
*   **智能信息检索与知识图谱构建:** 从海量文本中自动抽取实体和关系，构建领域知识图谱，支持智能问答、语义搜索等。
*   **生物医学信息学:** 用于从医学文献中识别疾病、基因、蛋白质、药物等实体及其相互作用，加速生物医学发现和药物研发。
*   **科学文献分析:** 自动化地从科学论文中抽取关键信息，如研究目标、方法、结果和结论，辅助科研人员进行文献综述、趋势分析和知识发现。
*   **法律与金融文本分析:** 抽取合同、报告中的关键条款、事件和关系，辅助法律审查、风险评估和市场分析。
*   **智能客服与对话系统:** 理解用户意图和对话上下文中的实体关系，提升对话系统的智能化水平。
*   **自然语言理解研究:** 作为标准基准数据集，用于评估和比较不同NLP模型的性能，推动关系抽取、NER、事件抽取等任务的算法创新。


- [数据集 | SOTA！模型](https://sota.jiqizhixin.com/datasets)
- [JNLPBA Dataset - NLP Hub - Metatext](https://metatext.io/datasets/jnlpba)
- [CDR和GDA数据集](https://github.com/fenchri/edge-oriented-graph)
- [DocRED文档级数据集](https://github.com/thunlp/DocRED)
- [DialogRE: The First Human-Annotated Dialogue-Based Relation Extraction Dataset](https://dataset.org/dialogre/)
- [斯坦福 TACRED 主页](https://nlp.stanford.edu/projects/tacred/)
- [ProKil/FewRel: A Large-Scale Few-Shot Relation Extraction Dataset](https://github.com/ProKil/FewRel)
- [Multi-Task Identification of Entities, Relations, and Coreferencefor Scientific Knowledge Graph Construction](http://nlp.cs.washington.edu/sciIE/)
- [ACE 2005 Multilingual Training Corpus - Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC2006T06)
- [ACE 2004 Multilingual Training Corpus - Linguistic Data Consortium](https://catalog.ldc.upenn.edu/LDC2005T09)
- [allenai/SciREX: Data/Code Repository for https://api.semanticscholar.org/CorpusID:218470122](https://github.com/allenai/SciREX)
- [klimzaporojets/DWIE: DWIE (Deutsche Welle corpus for Information Extraction) dataset. Introduced in our "DWIE: an entity-centric dataset for multi-task document-level information extraction" paper (accepted in Information Processing and Management)](https://github.com/klimzaporojets/DWIE)

------------------------------------------------------------

## 关系推理


- [Research/README.md at master · PaddlePaddle/Research](https://github.com/PaddlePaddle/Research/blob/master/KG/ACL2021_GRAN/README.md)
- [Research/KG/CoKE at master · PaddlePaddle/Research](https://github.com/PaddlePaddle/Research/tree/master/KG/CoKE)

------------------------------------------------------------

## 关键词抽取


- [StructBert关键词抽取-中文-base-ICASSP2023-MUG-Track4 · 模型库](https://modelscope.cn/models/damo/nlp_structbert_keyphrase-extraction_base-icassp2023-mug-track4-baseline/summary)

------------------------------------------------------------


# 1.图学习PGL-ERNIESage


- [说明文档Paddle Graph Learning (PGL) — pgl 2.1.5 documentation](https://pgl.readthedocs.io/en/latest/introduction/introduction.html)
- [PGL主页](https://github.com/PaddlePaddle/PGL/blob/main/README.zh.md)
- [基于PaddleNLP的ErnieSage模型介绍](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_graph/erniesage)
- [PGL:TransE  TransR  RotatE](https://github.com/PaddlePaddle/PGL/blob/main/legacy/examples/pgl-ke/README.md)
- [PGL/apps/GNNAutoScale at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/tree/main/apps/GNNAutoScale)
- [Graph4Rec](https://github.com/PaddlePaddle/PGL/blob/main/apps/Graph4Rec/README.md)
- [PGL/apps/Graph4KG at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/tree/main/apps/Graph4KG)
- [PGL论文算法复现](https://github.com/PaddlePaddle/PGL/tree/main/examples/citation_benchmark)
- [PGLBox：GPU版本引擎](https://github.com/PaddlePaddle/PGL/tree/main/apps/PGLBox)
- [PGLBox全面解决图训练速度、成本、稳定性、复杂算法四大问题！](https://mp.weixin.qq.com/s/s3lSYC06G-N85SBHi4seqQ)
- [OGB-LSC @ KDD Cup 2021 | Open Graph Benchmark](https://ogb.stanford.edu/kddcup2021/results/)
- [图学习：PGL: Paddle Graph Learning (PGL) is an efficient and flexible graph learning framework based on PaddlePaddle](https://github.com/PaddlePaddle/PGL)
- [图学习课程PGL： PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/blob/main/course/README.md)

------------------------------------------------------------

## 关系推理

- [Research/KG/ACL2021_GRAN at master · PaddlePaddle/Research](https://github.com/PaddlePaddle/Research/tree/master/KG/ACL2021_GRAN)
- [Coke：补全缺失实体等：链接预测和路径查询回答](https://github.com/PaddlePaddle/Research/blob/master/KG/CoKE/README.md)

------------------------------------------------------------

## 图谱和文本信息融合数据集


- [对话数据集：图的三元组和来自文档的文本](https://github.com/PaddlePaddle/Research/blob/master/NLP/EMNLP2019-AKGCM/README.md)

------------------------------------------------------------

## 实体对齐



- [MaoXinn/SEU](https://github.com/maoxinn/seu)
- [MaoXinn/PSR: The Code of "Are Negative Samples Necessary in Entity Alignment? AnApproach with High Performance, Scalability and Robustness" CIKM 2021](https://github.com/maoxinn/psr)
- [OpenEA 数据集 v2.0](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3)
- [实体对齐DBP2.0数据](https://figshare.com/articles/dataset/DBP2_0/14872080)

------------------------------------------------------------

## 课程




- [图神经网络7日打卡营 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/course/introduce/1956)
- [图神经网络7日打卡营常见问题整理 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/1259681)
- [PGL全球冠军团队带你攻破图神经网络_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1rf4y1v7cU?vd_source=8b49296a88726fc4482af0f68854e4b2)
- [斯坦福CS224W课程](http://web.stanford.edu/class/cs224w/)

------------------------------------------------------------

## ERNIESage




- [2.x版本基于PaddleNLP的ErnieSage模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_graph/erniesage/README.md)
- [1.x版本ERNIE-Sage（注释更详细的）](https://wenxin.baidu.com/wenxin/modelbasedetail/ernie_sage/)
- [1.8版本PGL图学习——ERNIESage运行实例](https://aistudio.baidu.com/aistudio/projectdetail/667443)
- [PGL实现ERNIESage图网络](https://github.com/PaddlePaddle/PGL/tree/static_stable/examples/erniesage)

------------------------------------------------------------

## 飞桨图模型合集

- [PGL/examples at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/tree/main/examples)
- [PGL/1-Introduction.ipynb at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/blob/main/legacy/tutorials/1-Introduction.ipynb)
- [PGL/legacy/examples at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/tree/main/legacy/examples)
- [PGL/README.md at main · PaddlePaddle/PGL](https://github.com/PaddlePaddle/PGL/blob/main/ogb_examples/graphproppred/ogbg_molpcba/README.md)

------------------------------------------------------------

## GraphSage项目


- [PGL图学习——ERNIESage运行实例](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/57957)
- [PGL：STGCN时空序列预测初探 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/592858?channelType=0&channel=0)
- [图神经网络入门节点分类比赛](https://aistudio.baidu.com/aistudio/competition/detail/59/0/datasets)

------------------------------------------------------------


# 1.文本分割


- [lxchtan/PoNet: Official code for ICLR 2022 paper: "PoNet: Pooling Network for Efficient Token Mixing in Long Sequences".](https://github.com/lxchtan/PoNet)
- [文本自动分段：texttiling算法](https://github.com/blmoistawinde/HarvestText#%E8%87%AA%E5%8A%A8%E5%88%86%E6%AE%B5)
- [PoNet文本话题分割模型-中文-base-ICASSP2023-MUG-Track1 · 模型库](https://modelscope.cn/models/damo/nlp_ponet_document-segmentation_topic-level_chinese-base/files)
- [BERT文本分割-中文-通用领域 · 模型库](https://modelscope.cn/models/damo/nlp_bert_document-segmentation_chinese-base/quickstart)
- [魔搭社区-相关文章见ATA](https://modelscope.cn/models?page=1&tasks=universal-information-extraction&type=nlp)
- [SpokenNLP/alimeeting4mug/readme.md at main · alibaba-damo-academy/SpokenNLP](https://github.com/alibaba-damo-academy/SpokenNLP/blob/main/alimeeting4mug/readme.md)

------------------------------------------------------------



# 1.文本分类任务


- [0.零样本文本分类UTC](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/zero_shot_text_classification/README.md)
- [0.文本分类合集](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification)
- [0.PLSC：Paddle大规模分类预训练](https://github.com/PaddlePaddle/PLSC)
- [0.ERNIE-DOC：长文本分类任务](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification/ernie_doc)
- [1.PP-MiniLM分类](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/model_compression/pp-minilm)
- [1.UTC是否支持多分类、多标签、层次分类、层次多标签 的多种还是一种？ · Issue #4973 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/4973)
- [1.utc中提到的zero-shot-classification如何进行多阶段微调 · Issue #4835 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/4835)
- [1.UTC微调，分类标签过多问题 · Issue #4730 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/4730)
- [1.[Bug]: zero shot text classification的taskflow离线加载模型失败 · Issue #4731 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/4731)
- [1.[Bug]: TypeError: 'numpy.float64' object is not iterable · Issue #4465 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/4465)
- [--single_label 报错：KeyError: 'eval_macro_f1'](https://github.com/PaddlePaddle/PaddleNLP/issues/5713)
- [[Question]: 在利用uie模型进行分类时，多标签且单个标签较长的任务进行预测时，返回结果为空[{ }] · Issue #6681 · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/issues/6681)

------------------------------------------------------------

## 0.文本分类应用合集


- [1.总：文本分类任务指南](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification)
- [1.文本多分类](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_class#%E6%96%87%E6%9C%AC%E5%A4%9A%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E6%8C%87%E5%8D%97)
- [2.多标签分类](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/multi_label#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1)
- [3.层次分类](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/text_classification/hierarchical#%E5%A4%9A%E6%A0%87%E7%AD%BE%E5%B1%82%E6%AC%A1%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1)

------------------------------------------------------------

## 1.情感分类：评论观点抽取与情感倾向性分析


- [总览：情感分析应用](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/sentiment_analysis/README.md)
- [评论观点抽取与情感倾向性分析](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/sentiment_analysis/ASO_analysis/README.md)
- [UIE通用情感信息抽取](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/sentiment_analysis/unified_sentiment_extraction)
- [基于SKEP的情感分析方案](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/sentiment_analysis/ASO_analysis)
- [端到端情感分析系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/sentiment_analysis/README.md)
- [SKEP情感分类模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/sentiment_analysis/skep/README.md)

------------------------------------------------------------

## 3.ERNIE套件文本分类



- [文本分类ERNIE、三个模型modelhub添加最新模型](https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/applications/tasks/text_classification)
- [ernie：多分类、多标签合集](https://github.com/PaddlePaddle/ERNIE/blob/ernie-kit-open-v1.0/applications/tasks/text_classification/README_CODE.md)
- [文心大模型ERNIEKit旗舰版 - 进阶任务：Prompt tuning文本分类 | 百度AI开放平台](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/ol57w627n)

------------------------------------------------------------

## 项目合集


- [基于PaddleNLP搭建评论观点抽取和情感分析系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3360011)
- [基于ERNIE3.0的中文评论分类 （套件开发）](https://aistudio.baidu.com/aistudio/projectdetail/4355513)
- [【含模型融合】面向微博话题的群体情感识别Baseline - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4446086)
- [[比赛分享]讯飞-基于论文摘要的文本分类与查询性问答第4名(并列第3)的思考 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4405354)

------------------------------------------------------------


# 1.文本匹配纠错


- [文本匹配总览](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/README.md)
- [文本匹配ERNIE组件](https://github.com/PaddlePaddle/ERNIE/tree/ernie-kit-open-v1.0/applications/tasks/text_matching)
- [无监督语义匹配模型 DiffCSE](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/diffcse/README.md)
- [基于预训练模型 ERNIE-Gram 的单塔文本匹配](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/ernie_matching/README.md)
- [使用预训练模型Fine-tune完成中文文本匹配任务](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/sentence_transformers/README.md)
- [SimBERT模型](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/simbert/README.md)
- [无监督语义匹配模型 SimCSE](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/simcse/README.md)
- [使用SimNet完成文本匹配任务](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/simnet/README.md)
- [千言-问题匹配鲁棒性评测基线](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/question_matching/README.md)
- [文本匹配-erniekit](https://github.com/PaddlePaddle/ERNIE/blob/ernie-kit-open-v1.0/applications/tasks/text_matching/README.md)

------------------------------------------------------------

## 文本纠错


- [中文文本纠错工具pycorrector: pycorrector is a toolkit for text error correction. 文本纠错，Kenlm，ConvSeq2Seq，BERT，MacBERT，ELECTRA，ERNIE，Transformer，T5等模型实现，开箱即用。](https://github.com/shibing624/pycorrector)
- [中文文本纠错任务：ERNIE for Chinese Spelling Correction](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_correction/ernie-csc/README.md)
- [fiyen/PaddlePaddle-DocCRT: 基于百度ERNIE和Pycorrector的文本编辑和批改的软件](https://github.com/fiyen/PaddlePaddle-DocCRT)
- [FreeFlyXiaoMa/pycorrector: 错别字纠正算法。调用pycorrector接口，使用规则。](https://github.com/FreeFlyXiaoMa/pycorrector)
- [Pycorrector -demo展示](https://huggingface.co/spaces/shibing624/pycorrector)

------------------------------------------------------------



# 1.文档智能（DI, Document Intelligence）



- [智能文档分析_智能文档处理_智能文档审核_智能文档审阅_智能文档比对_智能文档审校-百度AI开放平台](https://ai.baidu.com/tech/nlp/Textanalysis)
- [信息抽取应用（文档文本）](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/information_extraction/README.md)
- [文档智能应用](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/document_intelligence)
- [ERNIE-Layout](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/ernie-layout/README_ch.md)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/README.md)
- [PaddlePaddle/PaddleOCR - 码云 - 开源中国](https://gitee.com/paddlepaddle/PaddleOCR/tree/release/2.6)
- [UIE优化表格关键信息抽取 -](https://aistudio.baidu.com/aistudio/projectdetail/5704766?sUid=2900009&shared=1&ts=1679480566567)
- [百度网盘AI大赛——表格检测进阶：表格的结构化 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/704/0/related-material)
- [百度网盘AI大赛——表格检测 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/702/0/introduction)
- [文档抽取问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/document-intelligence/README.md)
- [UIE产业应用案例：风险事件抽取、表单提取、知识图谱构建](https://aistudio.baidu.com/aistudio/education/group/info/28328)
- [UIE大模型产业范例-跨模态信息抽取、医疗文档信息抽取等](https://aistudio.baidu.com/aistudio/course/introduce/28653)

------------------------------------------------------------

## PDFPlumber


- [pdfplumber：探测 PDF 以获取有关每个字符、矩形、线条等的详细信息——并轻松提取文本和表格。](https://github.com/jsvine/pdfplumber)
- [Python 操作pdf文件(pdfplumber读取PDF写入Excel)_猿小鱼的博客-CSDN博客_pdfplumber](https://blog.csdn.net/u014096024/article/details/126289068?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-126289068-blog-122579548.pc_relevant_3mothn_strategy_and_data_recovery&spm=1001.2101.3001.4242.2&utm_relevant_index=4)
- [PDFPlumber使用入门](https://blog.csdn.net/fuhanghang/article/details/122579548)
- [pdfplumber提取pdf表格数据并保存到excel文件中](https://www.jb51.net/article/256498.htm)

------------------------------------------------------------

## PP-Structure文档分析


- [PP-Structure 文档分析](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/README_ch.md)
- [PP-Structure 快速开始](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/quickstart.md)
- [表格识别](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/table/README_ch.md)
- [PP-Structure 系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/models_list.md)

------------------------------------------------------------

## PaddleOCR


- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/README_ch.md)
- [PaddleOCR/README_ch.md at release/2.6 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/README_ch.md)
- [PaddleOCR/PP-StructureV2_introduction.md at release/2.6 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/PP-StructureV2_introduction.md)
- [PaddleOCR/数据集](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6/doc/doc_ch/dataset)
- [【校园AI Day-AI workshop】基于OCR的PDF文档关键信息抽取 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4116071)
- [【校园AI Day-AI workshop】自定义区域OCR识别文件重命名 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4048105)

------------------------------------------------------------

## 公开数据集


- [Microsoft Research Open Data](https://msropendata.com/datasets/505fcbe3-1383-42b1-913a-f651b8b712d3)
- [TableBank Dataset](https://doc-analysis.github.io/tablebank-page/index.html)

------------------------------------------------------------

## 邮件解析库


- [mmpi: NextB的恶意邮件识别项目](https://github.com/a232319779/mmpi#%E4%B8%89%E6%89%AB%E6%8F%8F%E7%BB%93%E6%9E%9C%E8%BE%93%E5%87%BA)
- [mmpi，一款邮件快速检测python库](https://github.com/a232319779/mmpi)

------------------------------------------------------------

## 项目合集

- [模型抽取PDF版上市公司公告](https://aistudio.baidu.com/aistudio/projectdetail/4497591)
- [PaddleNLP文档智能技术重磅升级，动手搭建端到端文档抽取问答模型 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4881278)
- [基于PaddleNLP UIE模型提取《人民日报》PDF新闻信息 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4536480?channelType=0&channel=0)
- [汽车说明书跨模态智能问答](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/document_intelligence/doc_vqa#readme)
- [汽车说明书跨模态智能问答，AI客服24小时都在线 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4049663)

------------------------------------------------------------


# 1.问答系统（含rocketQA训练）

- [1.RocketQA训练信息检索和问答的密集检索，包括中文和英文SOTA模型。](https://github.com/PaddlePaddle/RocketQA)
- [1.问答系统总览](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/question_answering/README.md)
- [1.PaddleNLP Pipelines：智能文本产线](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/pipelines#%E6%99%BA%E8%83%BD%E6%96%87%E6%9C%AC%E4%BA%A7%E7%BA%BF%E5%BA%93)
- [2.端到端智能问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/question-answering/README.md)
- [2.WINDOWS环境下搭建端到端智能问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/question-answering/Install_windows.md)
- [2.FAQ智能问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/FAQ/README.md)
- [3.机器阅读问答 (MRQA)](https://github.com/PaddlePaddle/Research/blob/master/NLP/MRQA2019-BASELINE/README.md)
- [3.论文复现mpnet: mpnet_paddle](https://github.com/junnyu/paddle-mpnet)
- [重点问题：FAQ 智能问答系统 · 效果不好改进](https://github.com/PaddlePaddle/PaddleNLP/issues/5702)
- [保险智能问答](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/question_answering/supervised_qa/faq_finance)
- [无监督检索式问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/question_answering/unsupervised_qa/README.md)

------------------------------------------------------------


## 1.1有监督检索式问答系统



- [保险智能问答](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/question_answering/supervised_qa/faq_finance/README.md)
- [政务问答检索式 FAQ System](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/question_answering/supervised_qa/faq_system/README.md)

------------------------------------------------------------

## 1.2无监督检索式问答系统



- [无监督检索式问答系统](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/applications/question_answering/unsupervised_qa)
- [无监督智能检索问答系统](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/pipelines/examples/unsupervised-question-answering/README.md)

------------------------------------------------------------

## AIstudio项目合集


- [2.PaddleNLP带你十分钟搭建FAQ智能问答系统](https://aistudio.baidu.com/aistudio/projectdetail/4612920)
- [3.PaddleNLP Pipelines带你10分钟完成抽取式智能问答系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4442857)
- [4.PaddleNLP Pipelines带你十分钟搭建FAQ智能问答系统 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4465498)
- [1.手把手搭建FAQ保险问答系统](https://aistudio.baidu.com/aistudio/projectdetail/4612899)
- [5.手把手搭建FAQ保险问答系统 -](https://aistudio.baidu.com/aistudio/projectdetail/3882519)

------------------------------------------------------------


# 2.Parl 飞桨强化学习模型库


- [使用教程 — PARL 2.2.1 文档](https://parl.readthedocs.io/zh_CN/latest/parallel_training/setup.html)
- [PARL:强化学习模型-](https://github.com/PaddlePaddle/PARL)
- [蘑菇书EasyRL](https://datawhalechina.github.io/easy-rl/#/chapter1/chapter1_questions&keywords)
- [EASY RL强化学习：案例与实践](https://aistudio.baidu.com/aistudio/education/group/info/27444)
- [papers with code](https://paperswithcode.com/paper/multi-agent-actor-critic-for-mixed)
- [PaddlePaddle/RLSchool: 四轴例子强化学习环境集](https://github.com/PaddlePaddle/RLSchool)
- [强化学习工程模型](https://github.com/PaddlePaddle/models)
- [冠军解决方案：NIPS2020强化学习电网调度赛事](https://github.com/PaddlePaddle/PARL/tree/develop/examples/NeurIPS2020-Learning-to-Run-a-Power-Network-Challenge)
- [国家电网调控AI创新大赛：电网运行组织智能安排](https://github.com/PaddlePaddle/PARL/blob/develop/examples/Baselines/GridDispatch_competition/README.md)
- [飞船游戏的一个PPO算法的基线方案](https://github.com/PaddlePaddle/PARL/blob/develop/examples/Baselines/Halite_competition/paddle/README.md)
- [【自动模拟驾驶游戏】SAC in Carla simulator](https://github.com/PaddlePaddle/PARL/blob/develop/examples/CARLA_SAC/README.md)
- [Qmix星际争霸](https://github.com/PaddlePaddle/PARL/blob/develop/examples/QMIX/README.md)
- [MAL多智能体机器翻译任务](https://github.com/PaddlePaddle/Research/blob/master/NLP/EMNLP2019-MAL/README.md)
- [飞桨框架2.0造一个会下五子棋的AI模型](https://www.paddlepaddle.org.cn/documentation/docs/zh/practices/reinforcement_learning/AlphaZero.html)

------------------------------------------------------------

## MATAD3


- [ZiyuanMa/MATD3: An implementation of multi-agent TD3 with paddlepaddle and parl](https://github.com/ZiyuanMa/MATD3)
- [ZiyuanMa/MuZero：MuZero 在 PyTorch 和 Ray for Reversi 中的实现](https://github.com/ZiyuanMa/MuZero)

------------------------------------------------------------

------------------------------------------------------------



# 2.paddleTS时序模型


- [欢迎使用PaddleTS — 文档](https://paddlets.readthedocs.io/zh_CN/latest/)
- [paddleTS文档.](https://paddlets.readthedocs.io/zh_CN/latest/source/installation/overview.html)
- [PaddlePaddle/PaddleTS:](https://github.com/PaddlePaddle/PaddleTS/)
- [Paddle ST-DM （基于PaddlePaddle平台实现的时空大数据挖掘研究与应用](https://github.com/PaddlePaddle/Research/blob/master/ST_DM/README.md)
- [PP-TS_基础模型_时序预测_NVDIA-飞桨AI Studio星河社区](https://aistudio.baidu.com/modelsdetail/339?modelId=339)
- [1.PaddleTS飞桨时序建模算法库，预测性维护、智慧能耗分析等一网打尽](https://mp.weixin.qq.com/s?__biz=Mzg2OTEzODA5MA==&mid=2247589068&idx=1&sn=852077f62169bd197cd4fd45385a7a47&scene=21#wechat_redirect)
- [2.飞桨时序建模算法库PaddleTS全新升级！时序表征学习帮你突破数据表象，实现效果进阶！](https://mp.weixin.qq.com/s/pkpwD48wwXEk5AzwNgepKw)
- [PP-TS基于启发式搜索和集成方法的时序预测模型，使预测更加准确](https://mp.weixin.qq.com/s/OkPCMmAiqXz7qNUUu5OO6g)
- [高致病性传染病的传播趋势预测——时间序列预测算法Prophet - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/525311?channelType=0&channel=0)
- [【PaddleTS】大坝变形监测应用 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4417167)
- [时序建模算法库PaddleTS系列直播课](https://aistudio.baidu.com/aistudio/education/group/info/27798)
- [时序建模算法库PaddleTS技术与实践](https://aistudio.baidu.com/aistudio/education/preview/3597165)
- [PaddleTS——时间序列预测](https://aistudio.baidu.com/projectdetail/5009543)
- [PaddleTS操作——异常检测](https://aistudio.baidu.com/projectdetail/5010223?channelType=0&channel=0)
- [PaddleTS操作——时间序列表征](https://aistudio.baidu.com/projectdetail/5009645?channelType=0&channel=0)
- [【PaddleTS】大坝变形监测应用 - 飞桨AI Studio星河社区](https://aistudio.baidu.com/projectdetail/4417167?channelType=0&channel=0)
- [基于PaddleTS的土壤湿度预测](https://aistudio.baidu.com/projectdetail/5295659?channelType=0&channel=0)

------------------------------------------------------------



# 3.机器翻译


- [Machine Translation using Transformer](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/transformer/README.md)
- [PaddleHub机器翻译：文档的批量翻译 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/2348650)
- [机器翻译 PaddlePaddle/Research](https://github.com/PaddlePaddle/Research/blob/master/NLP/ACL2019-JEMT/README.md)
- [PaddleNLP/examples/machine_translation/README.md at develop · PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/README.md)
- [有道智云控制台](https://ai.youdao.com/console/#/)
- [资源包 - 机器翻译 - 控制台](https://console.cloud.tencent.com/tmt/resource_bundle)
- [腾讯-自动翻译 - 机器翻译 - 控制台](https://console.cloud.tencent.com/tmt)

------------------------------------------------------------


# 4.paddleCV图像视频类（多模态、OCR、分割、视频）


- [PaddleX: 可视化图形化开发界面](https://github.com/PaddlePaddle/PaddleX)
- [PaddleDetection目标检测](https://github.com/PaddlePaddle/PaddleDetection)
- [paddlecv](https://github.com/PaddlePaddle/models/tree/release/2.4/paddlecv)
- [PaddlePaddle/PaddleYOLO: 🚀🚀🚀 YOLO series of PaddlePaddle implementation, PP-YOLOE+, RT-DETR, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv10, YOLO11, YOLOX, YOLOv5u, YOLOv7u, YOLOv6Lite, RTMDet and so on. 🚀🚀🚀](https://github.com/PaddlePaddle/PaddleYOLO)
- [PaddleCV_飞桨-源于产业实践的开源深度学习平台](https://www.paddlepaddle.org.cn/paddle/paddlecv)
- [ERNIE-ViL 2.0 跨模态理解大模型](https://github.com/PaddlePaddle/ERNIE/blob/ernie-kit-open-v1.0/Research/ERNIE-ViL2/readme_en.md)
- [PaddleVideo:](https://github.com/PaddlePaddle/PaddleVideo)
- [Paddle3D](https://github.com/PaddlePaddle/Paddle3D)
- [PaddleDepth: 飞桨深度信息增强开发套件](https://github.com/PaddlePaddle/PaddleDepth/blob/develop/README_ch.md)
- [PASSL：视觉库自监督学习算法](https://github.com/PaddlePaddle/PASSL/blob/main/README_cn.md)
- [PaddleViT-是SOTA模型和相关工具的算法开发和实验平台。](https://github.com/BR-IDL/PaddleViT/blob/develop/README_cn.md)
- [ERNIE/README_zh.md at repro · PaddlePaddle/ERNIE](https://github.com/PaddlePaddle/ERNIE/blob/repro/ernie-vil/README_zh.md)
- [PaddleYOLO/README_cn.md at release/2.5 · PaddlePaddle/PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO/blob/release/2.5/README_cn.md)
- [PaddlePaddle/models: Pre-trained and Reproduced Deep Learning Models （『飞桨』官方模型库，包含多种学术前沿和工业场景验证的深度学习模型）](https://github.com/PaddlePaddle/models#PaddleSpeech)

------------------------------------------------------------

## PPOCR


- [PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/latest/index.html)
- [PaddlePaddle/PaddleOCR: Awesome multilingual OCR toolkits based on PaddlePaddle (practical ultra lightweight OCR system, support 80+ languages recognition, provide data annotation and synthesis tools, support training and deployment among server, mobile, embedded and IoT devices)](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR/doc/doc_ch/PP-OCRv4_introduction.md at release/2.7 · PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/PP-OCRv4_introduction.md)
- [RapidAI/RapidOCR: A cross platform OCR Library based on PaddleOCR & OnnxRuntime & OpenVINO.](https://github.com/RapidAI/RapidOCR/tree/main)
- [信创级开源OCR - 为世界内容安全贡献力量](https://github.com/RapidAI/RapidOCR/blob/main/docs/README_zh.md)
- [PP-ChatOCR：基于文心大模型的通用图像关键信息抽取利器](https://mp.weixin.qq.com/s/8CHj2xIdZH85-Xoe05Ha0Q)
- [【PP-ChatOCR】大模型+小模型，通用关键信息抽取新思路 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/6488689)
- [PP-OCRv4在线体验 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/6611435)
- [【PP-ChatOCR】基于LLM+OCR技术的通用文本图像智能分析系统](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=332)
- [PP-OCRv4 是实用的超轻量通用OCR系统](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286)

------------------------------------------------------------

## PaddleSports

- [PaddlePaddle/PaddleSports](https://github.com/PaddlePaddle/PaddleSports)

------------------------------------------------------------

## paddleRS遥感

- [AI快车道-遥感影像智能解译开发套件PaddleRS - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/course/introduce/27835)
- [paddleRS遥感图像处理](https://github.com/PaddlePaddle/PaddleRS)

------------------------------------------------------------

## paddleseg图像分割

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.6/README_CN.md)
- [AP-Kai/segment-anything-pd: Segment Anything With PaddlePaddle](https://github.com/AP-Kai/segment-anything-pd)
- [百度网盘AI大赛-图像处理挑战赛：水印智能消除赛 Baseline - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3859316)
- [百度网盘AI大赛-去水印模型冲刺赛亚军方案 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/5457474?channelType=0&channel=0)
- [飞桨学习赛：水印智能消除赛 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/785/0/introduction)
- [飞桨学习赛：文档图像摩尔纹消除 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/787/0/introduction)
- [百度网盘AI大赛：文档图像摩尔纹消除(赛题一) Baseline - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3220041)
- [飞桨学习赛：文档图像阴影消除 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/competition/detail/796/0/introduction)
- [文档阴影消除：『网盘赛』基于自定义训练模板的 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/3465771)
- [百度网盘AI大赛-水印智能消除赛B榜第16名方案 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4018956)
- [百度网盘AI大赛 -去水印模型冲刺赛第1名方案](https://github.com/dowdyboy/bdpan_shuiyin_competition)
- [百度网盘AI大赛-文档图像方向识别赛第3名方案](https://github.com/dowdyboy/bdpan_ori_competition_b)
- [百度网盘AI大赛-通用场景手写文字擦除赛第4名方案](https://github.com/dowdyboy/bdpan_erase_competition)
- [百度网盘AI大赛——图像处理挑战赛：文档图片去遮挡 B榜第8名方案](https://github.com/dowdyboy/bdpan_over)
- [百度网盘AI大赛——图像处理挑战赛：文档图像超分，本方案在B榜上取得了第3名的成绩](https://github.com/dowdyboy/bdpan_sr_competition)

------------------------------------------------------------

## 多模态


- [跨模态检索任务简介](https://ai.baidu.com/ai-doc/ERNIE-Ultimate/Hl5805h7n)
- [多模态技术创新赛-图像描述生成](http://www.aiinnovation.com.cn/#/trackDetail?id=27)
- [多模态技术创新赛-图像描述生成](https://github.com/MUGE-2021/image-caption-baseline)
- [多模态技术创新赛-基于文本的图像生成](http://www.aiinnovation.com.cn/#/trackDetail?id=28)
- [多模态技术创新赛-基于文本的图像生成](https://github.com/MUGE-2021/image-generation-baseline)
- [多模态技术创新赛-多模态检索](http://www.aiinnovation.com.cn/#/trackDetail?id=29)
- [多模态技术创新赛-多模态检索](https://github.com/MUGE-2021/image-retrieval-baseline)
- [UNIMO2语言与视觉一体的预训练模型](https://github.com/PaddlePaddle/Research/blob/master/NLP/UNIMO-2/README.md)
- [Research/README.md at master · PaddlePaddle/Research](https://github.com/PaddlePaddle/Research/blob/master/NLP/UNIMO/README.md)
- [[2205.03521v1] 良好的视觉引导使提取器更好：多模态实体和关系提取的分层视觉前缀](https://arxiv.org/abs/2205.03521v1)
- [zjunlp/HVPNeT：NAACL2022 论文“Good Visual Guidance Makes A Better Extractor: Hierarchical Visual Prefix for Multimodal Entity and Relation Extraction”的代码](https://github.com/zjunlp/HVPNeT)

------------------------------------------------------------

## 文生图



- [PaddleHub/README_ch.md at develop · PaddlePaddle/PaddleHub](https://github.com/PaddlePaddle/PaddleHub/blob/develop/README_ch.md)

------------------------------------------------------------

## 图像预训练大模型UFO

- [VIMER-UFO 2.0 (文心-CV大模型)](https://github.com/PaddlePaddle/VIMER/blob/main/UFO/README_ch.md)
- [StrucTexT v2.0文档图像理解基础模型](https://github.com/PaddlePaddle/VIMER/tree/main/StrucTexT/v2)
- [大模型应用新范式：统一特征表示优化（UFO）](https://ai.baidu.com/support/news?action=detail&id=2767)
- [VIMER-UFO-2.0:功能更强大更通用的视觉大模型重磅发布 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4046655)

------------------------------------------------------------


# 4.图像分类paddleClas

- [PLSC/README.md at master · PaddlePaddle/PLSC](https://github.com/PaddlePaddle/PLSC/blob/master/README.md)
- [PaddleClas图像识别图像分类](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/README_ch.md)
- [基于PaddleClas的天气以及时间多标签分类比赛 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4248894)
- [基于PaddleClas多标签图像分类 -](https://aistudio.baidu.com/aistudio/projectdetail/4247343)
- [基于PaddleX的【稻田医生】稻田病害分类 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4094423)
- [使用PaddleHub进行图像分类 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/147010)
- [重走长征路-PaddleClas训练ImageNet 1K数据集实践-后台任务版 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4401152)
- [【AI+农业】苹果医生-叶面病虫害分类-从数据集到Fastdeploy快速部署 - 飞桨AI Studio](https://aistudio.baidu.com/aistudio/projectdetail/4411585?channelType=0&channel=0)

------------------------------------------------------------


# 5.PaddleFL联邦学习隐私计算


- [文档详细说明：Distributed AI - PaddleDTX](https://paddledtx.readthedocs.io/zh_CN/latest/details/DAI/)
- [百度超级链-区块链开放平台](https://xuper.baidu.com/)
- [开源中心-开源社区-百度超级链](https://xuper.baidu.com/n/ps/opensource)
- [PaddleDTX多方安全计算](https://github.com/PaddlePaddle/PaddleDTX/blob/master/README_CN.md)
- [PaddleSleeve:安全与隐私工具隐私评测及保护能力](https://github.com/PaddlePaddle/PaddleSleeve)
- [PaddleFL联邦学习](https://github.com/PaddlePaddle/PaddleFL/blob/master/README_cn.md)
- [【PaddlePaddle Hackathon 3】链桨项目贡献合集 · Issue #44067 · PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle/issues/44067)
- [【PaddlePaddle Hackathon 第三期】任务总览 · Issue #43938 · PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle/issues/43938)
- [可信分布式AI课程](https://aistudio.baidu.com/aistudio/education/lessonvideo/3209673)
- [Federated隐语XGBoost — SecretFlow documentation](https://secretflow.antfin-inc.com/components/federated_learning/horizontal_federated_learning/tree.html)

------------------------------------------------------------


# 5.PaddleGAN生成对抗网络



- [PaddleGAN生成对抗网络](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/README_cn.md)
- [飞桨paddleGAN](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/52570)
- [应用PaddleGAN，不仅可以修复历史影像，还可以进行AI换脸、动作迁移、图像生成、图像风格迁移等等功能](https://www.paddlepaddle.org.cn/paddlegan)
- [PaddleGAN: PaddlePaddle GAN library, including lots of interesting applications like First-Order motion transfer, Wav2Lip, picture repair, image editing, photo2cartoon, image style transfer, GPEN, and so on.](https://github.com/PaddlePaddle/PaddleGAN/tree/develop)

------------------------------------------------------------


# 5.PaddleSpatial：时空大数据计算工具和平台



- [PaddleSpatial/tutorials at main · PaddlePaddle/PaddleSpatial](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/tutorials)
- [PaddleSpatial：区域分割、时空迁移学习、时间序列预测](https://github.com/PaddlePaddle/PaddleSpatial/blob/main/README_cn.md)
- [时空大数据计算工具PaddleSpatial解读与开源共建_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV18a411i7K9/?vd_source=8b49296a88726fc4482af0f68854e4b2)

------------------------------------------------------------


# 5.paddlespeech


- [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech/blob/develop/README_cn.md)
- [智能语音指令解析系统全流程搭建，实现语音工单信息抽取](https://aistudio.baidu.com/aistudio/projectdetail/4399703)
- [Parakeet语音合成](https://github.com/PaddlePaddle/Parakeet)

------------------------------------------------------------


# 6..paddle量桨：量子机器学习：解决优化问题等



- [paddleQuantum量桨：量子机器学习：解决优化问题等](https://github.com/PaddlePaddle/Quantum/blob/master/README_CN.md)
- [量桨-Paddle Quantum](https://qml.baidu.com/)
- [量桨解决组合优化问题](https://qml.baidu.com/tutorials/combinatorial-optimization/quantum-finance-application-on-portfolio-optimization.html)

------------------------------------------------------------


# 6.PaddlePALM多任务学习框架


- [PaddlePALM](https://github.com/PaddlePaddle/PALM/blob/master/README_zh.md)

------------------------------------------------------------

# 6.PaddleScience飞桨科学工具包



- [PaddleScience](https://github.com/PaddlePaddle/PaddleScience)
- [PaddleScience飞桨科学工具包快速实践](https://aistudio.baidu.com/aistudio/projectdetail/4278591)

------------------------------------------------------------


# 6.paddle迁移学习


- [PaddleTransfer: 飞桨迁移学习算法库](https://github.com/PaddlePaddle/PaddleTransfer)

------------------------------------------------------------

**[⬆ 返回README目录](../README.md#目录)**

**[⬆ Back to Contents](../README-EN.md#contents)**