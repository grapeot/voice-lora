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
- v2 实验结论：不采用。真实文章上编造内容、删论点、加 AI 式格式，自动指标看不出来；复盘 `docs/v2_experiment.md`（先由 AI 起草，再用 v1 改写，人工修两处）。
- `mdproc.split_blocks`：紧跟文字的引用块或表格单独成块、原样保留（此前"原稿：\n> 引用"会被一起送进模型改写）。
- v2 过滤默认要求原文数字全部出现在 AI 版里（此前 0.7）：15% 的训练样本"目标比输入多数字"，模型因此在 16/74 个测试节里编数字。
- `serve_vllm_lora.sh`：一次挂多个 LoRA checkpoint 打分（每个约 20 秒，HF generate 要 3–4 分钟）；`14_export_gguf.sh`：合并并转 GGUF 一步完成；`08_rewrite.py --skip-articles`。
- API 调用失败时透传 HTTP 状态、finish_reason 和响应片段（此前只报 `KeyError: 'message'`）。
- 列表编号（"1) … 2) …"）不再算作必须保留的数字：Gemini 常把列表改成行文，此前被误判为丢事实。

### CLI 重构

- 使用侧收成一个命令行工具 `voice-lora`（`rewrite`、`compare`、`score`、`check`、`fit`、`card`），逻辑搬进 `src/voice_lora/{rewrite,compare,classify,main}.py`；`08_rewrite.py`、`13_compare.py`、`15_classifier.py` 变成薄壳，参数不变。训练流水线仍是编号脚本（实验运行和只能在 GPU 环境跑的步骤）。
- 调用卡（`card.example.yaml`）：服务地址、模型 id、训练时的指令和采样参数。调用方只认调用卡，不再需要训练 config；`voice-lora card` 从 config 生成，指令不会和训练时不一致。
- 分类器可以存成 JSON（词表、idf、系数、AI 腔词表、作者称呼），加载后结果和现训的一致。
- `outline_regen` 挪到 `voice_lora/experimental/`，需要 `builder.allow_experimental: true` 才能用。
- 回归：重构前后，对照页 HTML 逐字节相同，`15_classifier.py score/check` 输出相同。

### 2026-09-29：改写 skill 的核查表按实测漂移更新

- v1.1 在 7 篇真实的第三人称说明文（技术长文、新闻简报、工程复盘）上逐段改写、逐段人工核查，一共约 70 处漂移，最高危的两类是动作主体归属错误（7 篇中 4 篇命中）和方向反转。原核查表低估了这几类：“人称和口吻”只举了元数据泄漏的例子；数字保险的能力边界也没讲清。
- `skills/voice-lora-rewrite/SKILL.md` 的“改完之后必须做的”一节：
  - 表格前写明数字保险只拦数值不符；
  - “人称和口吻”改为“归属（谁做的动作）”；
  - 新增“方向与时间”“错字与病句”两行；
  - “判断强度”扩展为同时覆盖绝对化和弱化两个方向；
  - “格式”补上自拟标签这种形态；
  - 表格前新增两条机械预检：一条查新出现的第一人称，一条按段比较列表行数。两条在两份已知漂移的日志上，都命中了人工核查找到的全部同类段落。

## Lessons Learned

- 数字保险不覆盖关系词和归属：数字不变，也可能把“恶化”写成“降低”、把“后”写成“内”、把“团队”写成“我们”。这几类只能逐段对照原文，便宜的 grep 预检可以先把最可疑的段落圈出来。
- 每个目标段落对应多条改写时，验证 loss 很早见底；1 epoch 起输出会"过头"。默认 0.5 epoch。
- 人/AI 分类器会被 LoRA 骗过，只用于比较 checkpoint；最终判据是盲评。
- 留出段落和真实 AI 文章的表现可以差很远（v1：0.87 对 0.31），瓶颈在数据构造而不是训练。
- Qwen3.5 加载出的是多模态 processor，文本推理要用 `processor.tokenizer`。
- 远端杀进程不要用 `pkill -f`，模式串会匹配到 ssh 会话自己的命令行。
