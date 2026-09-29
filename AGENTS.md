# AGENTS.md

这个仓库训练一个 LoRA，把 AI 写的中文改成某位作者的文风。要执行一轮训练，读 `skills/voice-lora/SKILL.md`；这份文件只讲怎么开发这个仓库本身。

## 结构

- `src/voice_lora/`：可复用模块。`builders/` 是可替换的数据构造器，其余模块被所有构造器共用。
- `scripts/`：按执行顺序编号的阶段脚本，都接受 `--config`。GPU 脚本（04–06）需要训练环境，其余在 CPU 上跑。
- `skills/voice-lora/`：给 agent 的 skill 与参考文档（中文）。
- `blind_eval/`：盲评页面模板。
- `docs/`：`prd.md`（做什么）、`rfc.md`（怎么做与关键决定）、`test.md`（什么算验证完成）、`working.md`（按天的 changelog 与教训，每次改动后更新）。
- `local/`（gitignore）：真实配置、语料衍生数据、训练输出、盲评答案。不属于这个仓库。

## 约束

- 仓库是公开的。不要提交任何作者语料、模型权重、训练输出、盲评答案，代码、文档、测试和 commit message 里也不能出现私有信息：主机名、内网地址、个人路径、账号、密钥。示例一律用假值（`example.com`、`/path/to/...`、`replace-with-your-key`）。
- 所有和具体作者或机器相关的东西都走配置，不写死在代码里。
- 训练和推理用的 prompt 格式由 `pairs.user_message` 统一生成，两边必须一致。
- 改评估逻辑时，用 v1 的真实输出复跑 `09_evaluate.py` / `10_select_checkpoint.py`，确认结论不变或变化有解释，再写进 `docs/working.md`。
- 提交前：`ruff check src tests scripts` 与 `python -m pytest -q` 通过。
- 只有维护者明确要求时才 commit / push。

## 环境

- CPU 侧：`uv venv && uv pip install -e '.[dev]'`。
- 训练侧和 vLLM 各用独立环境，见 `skills/voice-lora/references/environment.md`。
