# RFC

## 数据流

```
作者文章 ──01_units──▶ units.jsonl（段落单元，按文章切 train/val/test）
                         │
                02_generate（改写模型 × builder 的 prompt）
                         ▼
                   rewrites.jsonl ──03_build（过滤）──▶ sft_train/val.jsonl, eval_inputs.jsonl
                                                              │
                       04_train（Unsloth bf16 LoRA）◀──────────┘
                         │
      05_infer_hf（给 checkpoint 打分）──▶ 10_select_checkpoint
                         │
          06_merge ──▶ serve_vllm.sh / GGUF ──▶ 08_rewrite（留出段落 + 整篇文章）
                         │
     07_baseline_fewshot ─┴─▶ 09_evaluate ──▶ 11_blind_eval ──▶ 作者 ──▶ 12_score_blind
```

所有产物都写在配置的 `workdir` 下，文件名固定，前一步的产物就是后一步的输入。

## 关键决定

**反向合成数据。** 把作者原文改写成 AI 腔，再反过来训练。人不需要标注，数据量由作者的写作量决定。

**按文章切分。** 同一段的所有改写必须在同一个 split，否则测试集泄漏。`corpus.split.from_file` 允许沿用上一轮的划分，保证多轮结果可比。

**多种改写 prompt。** 同一个 prompt 采样多次，得到的是同一种 AI 腔；不同 prompt 对应不同的 AI 腔（默认、正式、模仿目标文章、扩写、精简）。其中模仿目标文章的那一种让训练输入贴近真实输入。

**过滤只卡事实。** 数字必须保留，不能凭空多出英文名。英文术语被译成中文不算错：把它们改回英文本身就是很多作者的文风。

**短训练。** 每个目标段落对应多条改写，1 个 epoch 等于同一个目标看了好几遍。v1 的验证 loss 在 0.25–0.5 epoch 见底，1 epoch 时已经过头。默认 0.5 epoch，学习率在这段内完整退火。

**Base 基座。** 同样设置下 Base 和 instruct 的段落指标打平，Base 的验证 loss 更低。

**评估的三层。** 自动指标（人/AI 分类器、AI 腔词表、chrF、数字核对）和 LoRA 用的是同一批数据对，LoRA 天然会迎合它们，所以只用来比较 checkpoint。真实目标文章单独测，因为它和留出段落的分布不同。最终判据是作者的盲评。

**推理引擎。** HF generate 只用于快速给 checkpoint 打分；批量和整篇改写合并权重后走 vLLM 或 GGUF。整篇文章默认所有段落并发，上一段的 AI 原文作上文；和逐段串行（上一段输出作上文）效果相当，速度快一个数量级。

## 数据构造器接口

`src/voice_lora/builders/<name>.py` 提供：

- `make_jobs(cfg, units, done) -> list[job]`：要调用改写模型的任务，跳过已完成的。
- `make_worker(cfg, client, style_pool)`：执行单个任务，返回写进 `rewrites.jsonl` 的一行。
- `build(cfg, units, rewrites) -> (examples_by_split, eval_inputs, reasons)`：过滤并写出 SFT 数据。

SFT 样本的格式对所有构造器相同：单轮 user/assistant，user 是固定指令 + 可选上文 + AI 侧文字，assistant 是作者原文。

## v2：outline_regen

v1 的局限是结构层：AI 改写保留了原文的段落和论证，模型只学到措辞。v2 以"节"为单位：从原文抽出要点和事实清单，让 AI 只凭清单写出这一节，训练（AI 节 → 原文节）。设计细节和选择"节"的理由见 `skills/voice-lora/references/builders.md`。

## 已知限制

- 自动指标依赖改写模型的腔调；换改写模型后，分类器和 AI 腔词表要重新学。
- 盲评依赖作者的时间，样本量小（v1 每项 20 组），显著性有限。
- 目前只支持中文语料（切段和过滤规则按中文写）。
