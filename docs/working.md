# Working

## Changelog

### 2026-09-29

- 以 v1 实验为基础重构为配置驱动的公开仓库：`src/voice_lora/`（corpus、builders、pairs、metrics、evaluation、blind、articles、llm），阶段脚本 `scripts/01–12`。
- 数据构造器抽象：`paragraph_rewrite`（v1）实现，`outline_regen`（v2）占位。
- checkpoint 选择与"过头"检测做成代码（`10_select_checkpoint.py`）；盲评改成本地静态页面 + JSON 导出。
- `corpus.split.from_file`：沿用上一轮的数据划分。
- 用 v1 数据复跑验证：单元、SFT 数据、评估输入逐字节一致；评估与选点复现 v1 结果。
- 中文 skill：`skills/voice-lora/SKILL.md` 及 `references/`。
- 整篇改写加保险（`articles.guarded`）：数字变化或长度比异常时重采样一次，仍不合格保留原文。起因：一篇 40 段文章里有一段丢了后半段并编造内容。
- 链接模糊挂回（`mdproc.restore_links`）：按相似度挂到改写后的片段，不跨标点，找不到就附在段末。一篇 19 个链接的文章从挂回 10 个提升到 19 个。
- 数字比较忽略千分位（`14,000` 与 `14000`）；只用于评估和保险，训练过滤规则不变以保证 v1 可复现。
- `08_rewrite.py` 增加 `--serve-url/--serve-model`，可直接指向 LM Studio；新增 `13_compare.py` 并排对照页。
- 用 llama.cpp 转 Q8_0 GGUF 并在 LM Studio 上跑通。
- 实现 v2 构造器 `outline_regen`：`corpus.make_sections` 按节切分；两阶段生成（extract 要点与事实 → draft 整节）；`keep_draft` 双向卡事实；`CommandClient` 以命令模板调用命令行 agent；`01_units.py` 支持构造器自定义单位，`02_generate.py` 支持 `--stage/--split`。
- 整篇推理支持按节改写（`--unit section`），对照页按改写单元对齐。
- `corpus.exclude.slugs`：排除文中声明由 AI 撰写的文章。
- outline_regen 支持多个 drafter（`drafters` 列表，后面的模型候选加 `@name` 后缀）并新增 `draft_formal`；每条候选记录 `source`。
- 列表编号（"1) … 2) …"）不再算作必须保留的数字：Gemini 常把列表改成行文，此前被误判为丢事实。

## Lessons Learned

- 每个目标段落对应多条改写时，验证 loss 很早见底；1 epoch 起输出会"过头"。默认 0.5 epoch。
- 人/AI 分类器会被 LoRA 骗过，只用于比较 checkpoint；最终判据是盲评。
- 留出段落和真实 AI 文章的表现可以差很远（v1：0.87 对 0.31），瓶颈在数据构造而不是训练。
- Qwen3.5 加载出的是多模态 processor，文本推理要用 `processor.tokenizer`。
- 远端杀进程不要用 `pkill -f`，模式串会匹配到 ssh 会话自己的命令行。
