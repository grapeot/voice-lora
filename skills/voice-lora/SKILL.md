---
name: voice-lora
description: 用一位作者自己写的中文文章，训练一个把 AI 写的中文改成这位作者文风的 LoRA，并用盲评判断它像不像。覆盖造数据、训练、选 checkpoint、评估、部署和多轮调优。
---

# voice-lora：把 AI 写的中文改成某位作者的声音

## 元数据

- **类型**：Workflow
- **适用场景**：有一位作者数十篇以上的中文原创文章，想得到一个本地模型，把 AI 起草的中文改成这位作者的措辞和语气
- **输出位置**：配置文件里 `workdir` 指向的目录（默认在 gitignore 的 `local/` 下）
- **代码入口**：`scripts/01_units.py` … `scripts/14_export_gguf.sh`，每一步都读同一个 `--config`

## 目标与边界

目标：得到一个 LoRA（外加合并后的完整权重），输入一段 AI 写的中文，输出内容和顺序不变、读起来像这位作者写的中文；同时给出能说服作者本人的证据。

做法的核心是反向合成数据：把作者的原文用 AI 改写成 AI 腔，再把数据对反过来训练（AI 腔 → 原文）。不需要任何人工标注。

不做的事：

- 不注入知识，不改论点。模型只改措辞和句子，事实必须原样保留。
- `paragraph_rewrite`（v1）改不了段落结构和论证方式（AI 改写保留了原文结构，模型从没见过结构差异）。`outline_regen`（v2，按节、先抽要点再让 AI 重写）试过结构层，结果会编造内容，不推荐使用，见 `docs/v2_experiment.md`。
- 不负责发布模型或数据。语料和模型都属于作者，是否分发由人决定。

## 验收标准

一轮完整的训练算完成，要同时满足：

1. `03_build.py` 的 `build_report.json` 里保留率 ≥ 90%，抽读 20 条改写：确实换成了 AI 腔，且没有丢事实。
2. `10_select_checkpoint.py` 给出了推荐 checkpoint，且推荐点没有 `overshoot` 或 `number_drift` 标记。
3. `09_evaluate.py` 的 `metrics.md` 里，所选模型在留出段落上的 P(作者) 明显高于 `baseline_fewshot`（同一改写模型 + 检索 k 段原文作例子）。打不过这个基线，说明写 prompt 就够了，微调不值得。
4. 真实目标文章（`evaluate.articles`，也就是以后真正要改的那类 AI 文章）上的指标单独报告，不能只报留出段落。
5. 作者本人做完盲评（`11_blind_eval.py` → 作者在浏览器里答题并导出 → `12_score_blind.py`），结果写进工作记录。盲评是唯一不会被模型"骗过"的判据。

v1 的参照值（作者本人的博客，187 篇）：留出段落 P(作者) 0.87，原文 0.86；基线 0.33。盲评：模型段落被作者判为本人写的 16/20，作者自己的真原文是 17/20；对基线胜率 14/20。

## 需要停下来问人的地方

只有这几件事是人的决定，其余自己推进：

- **语料边界**：哪些文章是作者亲手写的。时间段、英文版、转载、代写都要问清楚，写进 `corpus.exclude`。这是整个方法的地基，猜错了等于在教模型"把 AI 腔改成另一种 AI 腔"。除了问，还要在语料里搜一遍"本文由 AI 撰写""下面的文章是 AI 写的"这类声明：作者可能写过演示 AI 能力的文章，全文都是 AI 输出，按日期划分抓不到（`exclude.slugs` 单独排除）。
- **目标文章**：以后要改的是哪类 AI 文章（谁起草的、什么题材），用来配置 `style_exemplars` 和 `evaluate.articles`。
- **盲评**：只有作者本人能做。给作者一个页面，别替作者判断。
- **占用共享资源**：训练要独占一张 GPU。如果这张卡在跑别的服务，先问能不能借、借多久。

## 资源与环境

- **改写模型**（`rewriter`）：任何 OpenAI 兼容接口。v1 用的是本地部署的 Qwen3.8-27B，关掉思考，并发 6，改写 11,565 次约 40 分钟。改写模型的腔调越接近以后真正要改的 AI 文章越好。
- **训练**：单张 32GB GPU（v1 用 RTX 5090）。Qwen3.5-9B bf16 LoRA 峰值约 22GB。环境单独建：`uv venv --python 3.12 && uv pip install unsloth --torch-backend=auto`。
- **推理服务**：vLLM，再单独一个环境（它的 torch/transformers 版本和 Unsloth 冲突）。
- **Mac/CPU 侧**：`uv pip install -e .` 就能跑 01–03、07–12，不需要 GPU。

环境和版本细节见 `references/environment.md`。

## 流程地图

各阶段的产物都落在 `workdir` 里，前一步的产物是后一步的输入：

| 阶段 | 脚本 | 产物 | 判断是否到位 |
|---|---|---|---|
| 语料 | `01_units.py` | `units.jsonl`、`posts.csv`、`corpus_report.json` | 排除原因的计数和作者说的边界对得上；单元长度中位数在 150–250 字 |
| 造数据 | `02_generate.py --pilot 50` → 全量 | `rewrites.jsonl` | pilot 里每个 prompt 的改写都读几条：换了腔调、没丢事实 |
| 过滤 | `03_build.py` | `sft_train/val.jsonl`、`eval_inputs.jsonl` | 保留率 ≥ 90% |
| 基线 | `07_baseline_fewshot.py` | `eval/baseline_fewshot.jsonl` | 有它才能判断微调值不值 |
| 训练 | `04_train.py`（先 `--max-steps 30` 冒烟） | `runs/<run>/` | loss 在降；每 0.25 epoch 一个 checkpoint |
| 选点 | `serve_vllm_lora.sh` 一次挂上多个 checkpoint → `08_rewrite.py --serve-model <name> --skip-articles` 逐个打分 → `10_select_checkpoint.py` | `selection.json` | 见"调优"；每个 checkpoint 约 20 秒（`05_infer_hf.py` 也能用，但要 3–4 分钟） |
| 部署 | `06_merge.py` → `serve_vllm.sh`；桌面端用 `14_export_gguf.sh` 一步合并并转 GGUF | 合并权重、服务、GGUF | 单条请求 ~80 tok/s（9B/5090） |
| 评估 | `08_rewrite.py` → `09_evaluate.py` | `eval/metrics.md` | 验收标准 3、4 |
| 盲评 | `11_blind_eval.py` → 作者 → `12_score_blind.py` | `blind/<model>/score.json` | 验收标准 5 |
| 对照 | `13_compare.py` | `eval/articles/<stem>.<name>.compare.html` | 给作者看一篇真实文章改写前后的逐段并排 |

## 调优

调优是一个循环：训练 → 给几个 checkpoint 打分 → `10_select_checkpoint.py` → 根据提示改配置 → 再训。每轮只改一个变量，并沿用上一轮的数据划分（`corpus.split.from_file`），否则结果之间不可比。

`10_select_checkpoint.py` 的规则：在没有标记的 checkpoint 里取验证 loss 最低的。标记含义：

- `overshoot`：输出比作者本人还"像作者"（P(作者) 超过原文 0.03 以上，或 AI 腔词率低于原文的 60%）。这不是更好，是用力过猛，读起来会变成对作者的漫画式模仿。
- `number_drift`：超过 5% 的段落改动了数字。

它还会提示两种训练长度问题：验证 loss 在第一次评估时就最低（训短一点），或者在最低点之后明显回升（在背答案）。

v1 的真实过程可以当作例子：

1. 第一次按 2 epoch 训 instruct 基座。选点脚本的输出：验证 loss 在 0.25 epoch 最低（1.335），之后一路上升到 1.656；1 epoch 的点 P(作者) 0.905、AI 腔词率 0.67（原文 0.86 / 1.43），标记 `overshoot`，实际输出里出现了过火的粗口。
2. 原因：每个原文段落有 5 条改写，1 个 epoch 等于同一个目标看了 5 遍。改成 0.5 epoch、学习率在这 0.5 epoch 内完整退火，验证 loss 降到 1.319（instruct）和 1.294（Base），没有标记。
3. Base 和 instruct 基座的指标打平，Base 验证 loss 略低，选 Base。
4. 留出段落指标到顶以后，瓶颈出在真实目标文章上（P(作者) 只有 0.31）。原因在数据构造，不在训练，于是下一轮换数据构造器，而不是继续调超参。

完整数字和选点输出见 `references/optimization.md`。

## 已知陷阱

以下都在 v1 里真实发生过：

- **训练目标里不能有输入没有的事实。** 只要一部分样本的原文里带着 AI 输入里没有的数字，模型就学会"补"数字，推理时会凭空编出来。v2 第一版因此在 22% 的测试节里编了数字。构造器的过滤要保证原文的事实在输入里全部出现（`min_number_coverage: 1.0`），`09_evaluate.py` 的 `number_drift` 标记会报出这类问题。
- **分数更高，内容却在编。** v2（按要点重写整节）在自动指标上全面领先 v1（留出节 P(作者) 0.86 对 0.62，AI 腔词 2.1 对 5.4），人工读真实文章却发现它编引语、编细节、删论点、加粗体公式和"换句话说"。分类器只认用词习惯，看不出编造和格式化。评估真实文章时必须人工读几节，拿原稿逐句对照，见 `docs/v2_experiment.md`。
- **分类器指标会被 LoRA 骗过。** P(作者) 分类器和 LoRA 用的是同一批数据对，LoRA 天然会迎合它。它只能用来比较 checkpoint 和基线，不能当"像不像"的证据。它对训练时没见过的模型家族会偏松，换了数据要跑 `15_classifier.py check`，详见 `docs/classifier.md`。
- **留出段落好，不等于真实文章好。** v1 在留出段落上是 0.87，在真正要改的 AI 文章上只有 0.31：训练输入是"AI 改写的作者原文"，结构和论证仍是作者的；真实 AI 文章从结构到论证都是 AI 的。每一轮都要在真实目标文章上单独测。
- **按段落切分会泄漏。** 同一段的几条改写如果分进了不同的 split，测试集就被训练集看过了。`01_units.py` 按文章切分，别改成按段落。
- **过滤规则太严会丢掉有价值的数据。** 最初要求英文词完全一致，丢了 25%。AI 改写常把英文术语译成中文，而把它们改回英文恰恰是作者文风的一部分。现在只禁止凭空多出英文名、只要求数字不变，丢弃率约 5%。
- **Qwen3.5 加载出来的是多模态 processor。** 直接 `tokenizer(prompts)` 会把第一个参数当成图片，报 `Incorrect padding`。要用 `processor.tokenizer`，并在它上面设 left padding（`05_infer_hf.py` 已处理）。
- **HF generate 太慢。** 9B 单流 17–20 tok/s，只有带宽上限的五分之一，改一篇文章要 3–6 分钟。批量工作一律合并权重后走 vLLM：单流 84 tok/s，两篇文章 63 段全部并发只要 5 秒。
- **长段落偶尔会丢后半段并编造内容。** v1 在一篇 40 段的文章里出现过一次：原文"调查了 27 家保险计划的总精算师，近 70% 的受访者……"被改成"调查了 27 家健康保险的损失甚至都还没有被定义出来"。`08_rewrite.py` 默认对整篇文章开启保险：数字变了或长度比超出 0.6–1.6 就重新采样一次，仍不合格则保留原文并打印出来。别关掉它。
- **改写会换掉链接的锚文本。** 一篇 19 个链接的文章，按原锚文本只能挂回 10 个。`restore_links` 现在按相似度把链接挂到改写后对应的片段上（不跨越锚文本里没有的标点），实在找不到就附在段末，链接一个都不丢。
- **整篇并发和逐段串行效果一样。** 串行模式把上一段的输出作为上文，更贴近训练；并发模式用上一段的 AI 原文作为上文。v1 实测两者 P(作者) 0.29 对 0.32，差别在噪声内，默认用并发。
- **盲评有天花板。** 作者也只认出了 17/20 段自己的真原文。"判为作者写的"达到 80% 就已经接近上限，别把 100% 当目标，也别从作者判错的单条里读出太多东西。
- **LM Studio 看不到新模型。** 模型目录里一个坏掉的软链接会让它的索引扫描中途报错退出，排在后面的模型都不会出现；`lms ls` 里看不到时，先查 `~/.lmstudio/.internal/model-index-cache.json` 的 `errors`。放模型用 `lms import`，改动后需要重启应用才会重建索引。
- **GGUF 转换时报 `TokenizersBackend does not exist`。** 合并权重是 transformers 5.x 存的，转换环境里的 transformers 也要 ≥ 5.5。
- **远端杀进程别用 `pkill -f`，也别在同一条命令里既 grep 进程又提到它的名字。** 通过 ssh 执行时，模式串会匹配到这条 ssh 会话自己的命令行（`grep "[x]yz"` 只能避开 grep 本身，避不开同一条命令里别处出现的 `xyz`），把会话一起杀掉。先单独查出 PID，再单独 kill。
- **等待远端长任务时，等待条件要同时覆盖失败。** 只等成功标志的循环，在任务崩溃时会空转；v1 因此空等了一整夜。

## 参考文档

- `references/builders.md`：两种数据构造器的设计，包括 v2 的节级 outline_regen。
- `references/optimization.md`：v1 的完整调优记录和选点输出。
- `references/environment.md`：GPU 环境、版本约束，以及从在线服务借一张卡的做法。
