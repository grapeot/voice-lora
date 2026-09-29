# voice-lora

voice-lora 用作者自己写的中文文章，训练一个把 AI 写的中文改成这位作者文风的本地 LoRA。

在作者本人的中文博客案例（187 篇、2,313 个段落）中，基于 Qwen3.5-9B-Base 训练的 bf16 LoRA（单张 RTX 5090 训练 14 分钟），留出段落上“像作者写的”分类器概率达到 0.87（AI 输入为 0.18，检索 8 段原文作例子的基线为 0.33，作者原文本身为 0.86）。在作者本人对 20 段模型输出的盲评里，16 段被判为“是我写的”（作者自己的真原文认出 17 段），和基线二选一时模型胜 14/20。

## 为什么需要

AI 起草的中文有明显的腔调：用词端着、爱加铺垫和总结、英文术语全译成中文。

用 prompt 描述文风或者给几段例子，只能改掉一小部分用词，句式、语气和节奏基本不变。

## 核心方法：反向合成数据

本项目不需要人工标注，核心逻辑是反向合成数据：

1. **AI 改写**：把作者原文用 AI 改写成 AI 腔。这里使用 5 种 prompt：默认、正式书面、模仿目标 AI 文章、扩写、精简。
2. **反向配对**：将生成结果反过来配对，输入是各种形态的“AI 腔”，目标输出是“作者原文”。
3. **LoRA 微调**：通过监督微调，让模型学习如何把 AI 腔改回作者自然的措辞。

```
作者原文 ──[AI 改写 (5 种 prompt)]──▶ AI 腔文本
                                         │
                    反向配对 (AI 腔 ➔ 原文)
                                         ▼
                                   LoRA SFT 训练
```

## v1 案例结果与局限

以作者本人的中文博客为语料（187 篇、2,313 个段落），选用 Qwen3.5-9B-Base 基座与 bf16 LoRA，在单张 RTX 5090 上训练 14 分钟。

### 1. 留出段落指标

在模型未见过的留出段落上，“像作者写的”分类器概率对比如下：

| 评估对象 | “像作者写的”分类器概率 |
|---|---|
| AI 输入 | 0.18 |
| 大模型 + 检索 8 段作者原文作例子的基线 | 0.33 |
| 本模型 (Qwen3.5-9B LoRA) | 0.87 |
| 作者原文本身 | 0.86 |

### 2. 作者本人盲评

作者本人盲评（答题时不知道每段的来源）：

- 20 段模型输出里，被作者判为“是我写的”达到 **16 段**。
- 20 段作者真原文里，作者本人认出 **17 段**。
- 和基线二选一时，模型胜 **14/20**。

### 3. 局限与下一版目标

在真正由 AI 起草的整篇文章上，分类器概率只从 0.10 提到 0.31。

原因是当前训练数据里 AI 改写保留了原文的结构，模型只学到了措辞层；而真实 AI 文章往往从结构到论证都带有 AI 腔。v2 试过按“节”用要点让 AI 重写来学结构层，自动指标更好，但会编造内容，没有采用，复盘见 [docs/v2_experiment.md](docs/v2_experiment.md)。

### 4. v1.1：加入其他模型家族的改写

v1 的 AI 腔全部来自同一个开源 27B 模型。v1.1 在它的 5 种改写之外，每段再加 Gemini 3.8 Flash 和 DeepSeek V4.1 Flash 各 3 种改写（共 18,163 条训练对；Gemini 部分按 token 计约 6 美元，DeepSeek 部分更便宜），训练配置不变，取验证 loss 最低的 0.19 epoch。下表统一用 v1.1 数据训练的分类器打分（见 [docs/classifier.md](docs/classifier.md)，数值和上面 v1 的表不能直接比）：

| 测试输入 | AI 输入 | v1 | v1.1 | 作者原文 | chrF v1 → v1.1 | 数字错 v1 → v1.1 |
|---|---|---|---|---|---|---|
| 27B 改写的留出段落（291） | 0.09 | 0.77 | 0.76 | 0.75 | 32.6 → 32.9 | 2.1% → 2.1% |
| Gemini 改写的留出段落（285） | 0.02 | 0.76 | 0.74 | 0.75 | 27.3 → 26.8 | 2.8% → 3.5% |
| DeepSeek 改写的留出段落（289） | 0.25 | 0.74 | 0.76 | 0.75 | 45.5 → 49.7 | 2.8% → 0.0% |
| 4 篇 AI 起草的整篇文章（147 段） | 0.04 | 0.12 | 0.15 | - | - | 0 → 0 |

留出段落上两版都已经和作者原文持平，分类器分不出高下；v1.1 的收益集中在 DeepSeek 式输入（更接近原文、不改数字）和整篇文章上，幅度不大。人工对读一篇文章：v1.1 改动更保守，语义走偏更少，也有少数段落原样未改。

## 怎么用

### 1. 环境准备

CPU 侧（语料切分、数据构建、评估与盲评）：

```bash
uv venv && uv pip install -e '.[dev]'
```

训练与推理：
- **训练环境**：需要单独的 GPU 环境（Unsloth），例如：
  ```bash
  uv venv --python 3.12 && uv pip install unsloth --torch-backend=auto
  ```
- **推理服务**：权重合并后用 vLLM 部署独立服务，或转换为 GGUF 给 llama.cpp 的 `llama-server` 使用（9B 模型转 Q8_0 后约 9.1G，在 Mac 上改写一篇 40 段的文章约 1 分钟）。LM Studio 能加载同一个 GGUF，但它不传重复惩罚的窗口参数，改写会明显偏保守，见下文“用训练好的模型”。

### 2. 配置文件

复制配置模板：

```bash
cp config.example.yaml local/config.yaml
```

在 `local/config.yaml` 中配置核心参数：
- `corpus.paths`：作者文章路径（支持 glob 与 frontmatter 解析）。
- `rewriter`：改写模型接口（支持任何 OpenAI 兼容接口，填 base_url 与 model）。
- `author.name`：作者称呼（训练和推理 prompt 统一使用该称呼）。

### 3. 执行流程

所有阶段脚本均接受 `--config local/config.yaml`，按编号顺序执行：

| 阶段 | 脚本 | 说明 | 关键产物 |
|---|---|---|---|
| 语料切段 | `scripts/01_units.py` | 排除非手写文章，切分为段落单元，按文章划分训练/验证/测试集 | `units.jsonl`, `posts.csv` |
| 生成改写 | `scripts/02_generate.py` | 用 5 种 prompt 改写原文（可先加 `--pilot 50` 验证质量） | `rewrites.jsonl` |
| 过滤构建 | `scripts/03_build.py` | 检查数字保留与生造词，生成训练对（保留率 ≥ 90%） | `sft_train.jsonl`, `sft_val.jsonl` |
| 运行基线 | `scripts/07_baseline_fewshot.py` | 运行“改写模型 + 检索 8 段原文”基线供对比 | `eval/baseline_fewshot.jsonl` |
| 模型训练 | `scripts/04_train.py` | 在 GPU 上进行 Unsloth LoRA 训练（每 0.25 epoch 保存） | `runs/<run>/` |
| 候选打分 | `scripts/05_infer_hf.py` | 用候选 checkpoint 对验证集批量推理打分 | 候选输出与评分 |
| 选检查点 | `scripts/10_select_checkpoint.py` | 识别验证 loss 回升与过头标记，推荐最佳 checkpoint | `selection.json` |
| 权重合并 | `scripts/06_merge.py` | 合并 LoRA 权重与基座模型，便于后续高效服务 | 合并后的模型目录 |
| 批量改写 | `scripts/08_rewrite.py` | 调用推理服务并发改写留出段落与完整目标文章（实验用；单篇文章用 `voice-lora rewrite`） | 改写输出结果 |
| 指标评估 | `scripts/09_evaluate.py` | 汇总分类器得分、AI 腔词率与字级指标 | `eval/metrics.md` |
| 准备盲评 | `scripts/11_blind_eval.py` | 生成供作者答题的盲评 HTML 页面 | 盲评答卷页面 |
| 计算盲评 | `scripts/12_score_blind.py` | 导入作者答案并评分，给出认出率与胜率 | `score.json` |
| 并排对照 | `scripts/13_compare.py` | 一篇文章改写前后逐段并排，标出 AI 腔词；`--name` 可给多个模型，多栏对照 | `*.compare.html` |
| 分类器 | `scripts/15_classifier.py` | `check` 检查分类器可信度（按改写模型、段落长度）；`score` 给任意 markdown 文章打分 | `eval/classifier_check.json` |

### 4. 用训练好的模型：`voice-lora` 命令

训练流水线之外的日常使用走一个命令行工具（`uv pip install -e .` 后可用），不依赖实验工作目录：

```bash
# 模型卡：从训练用的 config 生成，保证 prompt 里的指令和训练时一致（格式见 card.example.yaml）
voice-lora card --config local/config.yaml --model <服务端模型 id> --base-url http://localhost:1234/v1 --out my-model.yaml

voice-lora rewrite article.md --card my-model.yaml --out-dir out/         # 改写，写出 .md 和逐段日志 .blocks.jsonl
voice-lora compare article.md out/article.<name>.md --classifier clf.json  # 原稿与改写逐段并排的 HTML
voice-lora fit --config local/config.yaml --out clf.json                   # 把分类器存成 JSON（不用 pickle）
voice-lora score *.md --classifier clf.json                                # 每篇文章的 P(作者) 与 AI 腔词率
voice-lora check --config local/config.yaml                                # 分类器可信度（见 docs/classifier.md）
```

采样参数（temperature 0.7、top_p 0.95、重复惩罚 1.05，惩罚覆盖整个 prompt）是正确调用的一部分，写在 `src/voice_lora/rewrite.py` 里，自己写客户端时请复用 `request_body`：

- greedy 解码下模型大多在照抄输入；
- 重复惩罚必须覆盖整个 prompt，它是让模型不照抄原文的主要力量。vLLM 默认就是这样；llama.cpp 默认只看最近 64 个 token，所以要发 `repeat_last_n`。同一段文字，和输入的相似度在 vLLM 上是 0.786，在 llama-server 覆盖全文时是 0.784，在默认窗口下是 0.849。
- LM Studio 只认 `repeat_penalty`，而且不传 `repeat_last_n`，没法正确调用这个模型。GGUF 请用 llama.cpp 的 `llama-server`。

### 5. 调优说明

`scripts/10_select_checkpoint.py` 会自动分析各 checkpoint 的指标表现，给出推荐：
- **识别“过头”（overshoot）**：当输出比作者本人还像作者（P(作者) 显著高于原文，或 AI 腔词率过低）时打上标记。这代表模型用力过猛，产生漫画式夸张模仿。
- **识别验证 loss 回升**：每个段落包含多条改写，通常 0.5 epoch 左右验证 loss 已见底。如果验证 loss 开始回升，说明模型在死记样本，脚本会建议缩短训练。

## 给 AI Agent 的入口

若把仓库交给 Claude Code、Codex 等 coding agent 执行，让它先读：

```
skills/voice-lora/SKILL.md
```

该文件详细记录了验收标准、必须停下来向人类确认的事项（语料边界、目标文章类型、GPU 资源协调、盲评答题）以及 v1 踩过的陷阱（如分类器迎合效应、文章级与段落级差异、Qwen3.5 tokenizer left padding 等）。

## 仓库边界

本开源仓库仅包含流程代码与工程脚本：
- 不包含作者的文章语料。
- 不包含合成的训练数据。
- 不包含训练出的模型权重。

所有生成数据与模型输出默认落在 git 忽略的 `local/` 目录中。

## License

[MIT](LICENSE)

---

## English Summary

voice-lora is a toolkit designed to train a lightweight LoRA from an author's own Chinese writing, transforming AI-drafted text into the author's distinct tone and vocabulary. It eliminates manual data labeling by synthesizing reverse training pairs, where the author's original writing is rewritten into typical AI prose across five distinct prompts. In a v1 benchmark on 187 blog posts, the author judged 16 of 20 model outputs to be their own writing in a blind test (the author accepted 17/20 of their own genuine paragraphs), and the model won 14 of 20 pairwise comparisons against a few-shot retrieval baseline.
