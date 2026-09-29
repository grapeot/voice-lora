# Test

## 离线单元测试（CI 必跑）

`python -m pytest -q`，不需要 GPU 和网络：

- 语料：排除规则（文件名、标签、状态、日期区间含首尾、汉字比例）、未闭合 frontmatter、清洗、切段与合并、按文章分层切分。
- 数据构造：过滤规则（数字、英文名、长度比、术语翻译放行）、指令来自配置、`style_mimic` 的确定性、split 与评估输入的选择、未知 prompt / builder 报错、v2 占位抛 `NotImplementedError`。
- markdown：非正文块透传、链接还原。
- 指标与盲评：数字核对的写法容忍、AI 腔词表、盲评题目构造（真假两边不取同一段）与计分、checkpoint 推荐跳过 `overshoot` 并提示 loss 回升。

## 端到端验证（需要改写模型和 GPU）

一次完整运行算验证通过，要满足 `skills/voice-lora/SKILL.md` 的验收标准。重构后的代码用 v1 的真实数据复跑过：语料单元、SFT 训练集和验证集、评估输入与 v1 逐字节一致；`09_evaluate.py` 和 `10_select_checkpoint.py` 复现了 v1 的指标和选点。
