# 本地三风格问答模型

使用本地 Qwen3-0.6B 基础模型和已训练的 LoRA，提供通俗、正式、幽默三种回答方式。支持浏览器和终端提问，不调用云端模型 API。每个问题独立回答，页面上的历史消息不会作为上下文发送给模型。

## 1. 已有完整文件夹：直接启动

如果文件夹包含 `.venv/`、`.python/`、基础模型和 LoRA，不需要重新安装依赖。

- 双击 `start_web.bat`：启动网页，自动打开 http://127.0.0.1:7860 。首次提问才加载模型。
- 双击 `start_chat.bat`：启动终端聊天，加载完成后输入问题，选择 1 通俗、2 正式、3 幽默；输入 `exit` 退出。

也可以在项目根目录打开 PowerShell：

```powershell
# 网页聊天
.\.venv\Scripts\python.exe .\web_chat.py

# 或终端聊天（选择其中一个运行即可）
.\.venv\Scripts\python.exe .\chat_trained.py
```

网页运行期间保持终端打开，按 Ctrl+C 停止服务。避免同时启动多个模型进程占用显存。

## 2. 协作者首次配置

以下步骤用于只有源码、没有可用环境的电脑；已有环境的本机无需执行。

### 系统与硬件

- 当前验证环境：Windows、Python 3.11.9、NVIDIA CUDA GPU。
- 当前聊天代码要求 CUDA；没有提供 CPU 或 Apple MPS 推理入口。
- 安装适用于显卡且支持所选 PyTorch CUDA 构建的 NVIDIA 驱动。先运行 `nvidia-smi` 确认驱动能够识别显卡。
- 显存需求随输入长度、生成长度和训练批次变化；训练通常比推理需要更多显存。
- 预编译 PyTorch 推理通常无需单独安装 CUDA Toolkit；需要兼容的 NVIDIA 驱动。

### 创建 Python 环境并安装依赖

先安装 64 位 Python 3.11。在项目根目录执行；如已有 `.venv`，请先备份或换用一个空的环境目录，不要直接覆盖原环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers==4.57.6 peft==0.21.2 datasets==5.0.1 accelerate==1.15.0 safetensors==0.8.0
```

这些版本来自本项目现有环境，不是跨硬件测试承诺。上面的 PyTorch 构建为 CUDA 12.8，驱动不兼容时需要选择合适的 PyTorch 构建并重新验证。其余间接依赖由 pip 安装。网页后端使用 Python 标准库，不需要 Flask、Gradio、Node.js 或前端构建工具。

检查环境：

```powershell
.\.venv\Scripts\python.exe -c "import torch,transformers,peft,datasets,accelerate; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

应看到 `CUDA: True`。如果安装提示找不到指定版本，请先检查软件源是否提供这些版本，不要直接混装不同版本。

### VS Code

用 VS Code 打开整个项目文件夹，通过 `Python: Select Interpreter` 选择 `.venv\Scripts\python.exe`。运行 `chat_trained.py` 可终端提问，运行 `web_chat.py` 可打开网页。终端工作目录应为项目根目录。

## 3. 准备模型文件

代码仅从本地读取模型，缺失文件时不会自动下载。

基础模型来源：<https://huggingface.co/Qwen/Qwen3-0.6B> 。下载完整模型仓库中的实际权重和分词器文件，放到 `models/Qwen3-0.6B/`。不要把 Git LFS 指针文件当作权重。

已训练 LoRA 需要向项目维护者获取，放到 `run_150epochs/adapter/`；它不是基础模型下载内容。本仓库若未分发 LoRA，则需要获取训练结果或自行训练。

```text
LLM/
├── models/Qwen3-0.6B/
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   └── 分词器文件（tokenizer.json、tokenizer_config.json、vocab.json、merges.txt 等）
├── run_150epochs/
│   ├── TRAINING_COMPLETE.json
│   └── adapter/
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── 配套分词器文件及 chat_template.jinja
├── mode_instructions.json
├── web_chat.py
├── web.html
└── chat_trained.py
```

终端聊天还会检查 `TRAINING_COMPLETE.json`，请保留真实训练完成记录。网页直接检查并加载适配器。当前运行路径依据脚本所在目录定位，不依赖原作者的盘符。

## 4. 项目文件用途

| 文件或目录 | 用途 |
| --- | --- |
| `web_chat.py` | 本地 HTTP 服务与模型推理，监听 127.0.0.1:7860 |
| `web.html` | 网页界面，无外部前端依赖 |
| `chat_trained.py` | 终端问答入口 |
| `start_web.bat` / `start_chat.bat` | 自动切换到项目目录并调用环境 Python |
| `mode_instructions.json` | 三种风格的提示指令 |
| `train_150.py` | 固定 150 轮的 LoRA 训练、评估及结果生成脚本 |
| `resume_tests.py` | 恢复批量测试；当前交付缺少它依赖的部分历史评估文件 |
| `style_train_429_v3.json` | 429 道训练题，三种风格共 1,287 条 |
| `style_test_71_v3.json` | 71 道测试题，三种风格共 213 条 |
| `instinwild_ch.json` | 51,504 条原始问答，尚未接入训练；存在质量问题，不能视为已审核知识库 |
| `interactive_answers.jsonl` | 终端问答日志，可能包含私人内容；网页不写入此文件 |
| `run_150epochs/` | 训练结果、适配器和运行记录 |
| `models/` | 基础模型 |
| `.venv/` | 本地依赖环境 |
| `.python/` | 本地复制的 Python 本体；协作者正常安装 Python 并创建环境后不需要此目录 |
| `文件清单.csv` | 临时导出的文件清单，不参与运行 |

## 5. 模型更新与训练

更新网页模型：停止网页进程，备份原 `adapter/`，替换为新版本 LoRA 和配套分词器，再重新启动。只刷新浏览器不会更新已经加载到内存中的模型。替换基础模型时需要保证它与 LoRA 匹配，并同步修改代码中的模型目录。

训练前关闭聊天服务释放显存，备份现有结果。不要直接在唯一一份模型结果上试验。

```powershell
# 训练脚本提供的数据/环境预检查入口
.\.venv\Scripts\python.exe .\train_150.py --check-only

# 正式执行：固定 150 轮，并在训练后进行评估
.\.venv\Scripts\python.exe .\train_150.py
```

预检查并非无副作用的纯语法检查，可能加载分词器并写入检查报告。运行训练前先阅读脚本。

当前脚本对训练/测试条数及分组有固定断言，不能直接换成 5 万条数据；需修改数据加载、划分和验证逻辑。150 轮是历史实验设置，不代表推荐轮数；已有实验出现测试损失上升，应使用验证集选择训练配置。当前 `run_150epochs` 不能只凭目录名称视为可继续训练的检查点，LoRA 推理文件不等同于完整 Trainer 检查点。

`resume_tests.py` 还需要历史 `data_manifest.json`、`metrics.json` 等文件，当前文件夹并未完整包含这些文件，不能直接完整运行。日常聊天不受影响。

## 6. 相对路径和换电脑

项目代码使用 `Path(__file__).resolve().parent` 定位资源，启动脚本使用 `%~dp0` 切换到项目目录。请保持目录结构。

随项目复制的 `.venv/pyvenv.cfg` 使用相对 Python 路径，启动时必须位于项目根目录；建议使用两个 bat 文件。复制的虚拟环境仍可能有带原路径的第三方工具启动器，因此包管理统一使用 `python.exe -m pip`，不保证任意环境工具可搬迁使用。

相对路径不保证环境可以跨系统、CPU 架构或显卡驱动直接运行。协作者使用自行安装的 Python 创建环境，是更稳定的配置方式。

## 7. 常见问题

- 网页打不开：先运行 `start_web.bat` 并查看终端；端口 7860 若被占用，先关闭旧服务。
- 首次回答慢：首次请求需要导入依赖并加载模型，后续请求复用模型。
- `CUDA: False`：检查 NVIDIA 驱动及是否安装 CUDA 版 PyTorch。
- 缺少模块：确认使用项目 `.venv` 的解释器，而不是系统中的另一个 Python。
- 显存不足：关闭其他模型进程，缩短输入，必要时调整生成长度和训练批次。
- 输出不准确：风格微调不等于可靠知识库；原始问答需筛选，回答需独立评估。
- 修改风格指令：编辑 `mode_instructions.json`；提示词修改不能代替模型训练。

## 8. GitHub 协作与数据

建议提交代码、启动脚本、风格配置、经过许可的数据及此说明。不要提交 `.venv/`、`.python/`、聊天日志、缓存和临时文件。模型权重使用独立下载渠道或合适的大文件存储，基础模型单文件约 1.5 GB，无法通过普通 GitHub 文件上传发布。

发布训练数据或权重前，确认来源许可证和个人信息；项目目前没有统一声明原创代码的开源许可证，不应由此 README 推断获得了全部数据和模型的再分发许可。
