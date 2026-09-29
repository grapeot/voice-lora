# 环境与资源

## 三个 Python 环境，不要混

| 用途 | 安装 | 说明 |
|---|---|---|
| 数据、评估、盲评（CPU 即可） | `uv venv && uv pip install -e '.[dev]'` | 跑 01–03、07–12 |
| 训练、HF 推理、合并（GPU） | `uv venv --python 3.12 && uv pip install unsloth --torch-backend=auto` | 跑 04–06 |
| 推理服务（GPU） | 单独的 venv 装 vLLM | `serve_vllm.sh` 的第三个参数指向它的 `vllm` |

v1 实测的版本组合：unsloth 2026.9.12、torch 2.12.1+cu132、transformers 5.5.0、trl 0.24.0；vLLM 0.27.1（自带 torch 2.13）。Unsloth 要求 torch < 2.13、transformers ≤ 5.5，vLLM 和常见的推理服务框架用的版本更新，装进同一个环境会互相破坏。

## GPU

- Qwen3.5-9B bf16 LoRA（r=32，batch 4 × 梯度累积 4，max_seq 2048）峰值约 22GB，32GB 的卡够用。
- RTX 5090（Blackwell，sm_120）可用；第一次跑时 Gated DeltaNet 的 Triton kernel 要编译，前几步会慢。
- 吞吐参考（单张 5090）：训练约 5.3 条/秒（每条约 670 字符），0.5 epoch（8,926 条）14 分钟；vLLM 单流 84 tok/s；HF generate 单流 17–20 tok/s。

## 从在线服务借一张卡

如果训练用的 GPU 平时在跑推理服务（例如一个 router 后面挂着多份副本），不要直接杀进程。顺序是：先在 router 上注销这份副本，等它手上的请求跑完，再按 PID 停掉；用完后反过来：先起副本、确认健康，再注册回 router。这样服务不中断，只是少一份容量。借卡前先征得服务主人同意。

## 改写模型

任何 OpenAI 兼容接口都行，配置在 `rewriter`。经验：

- 关掉思考模式（不同服务的开关不同：`chat_template_kwargs.enable_thinking=false`、`reasoning_effort: none` 等），否则慢且输出里夹着推理。
- 并发先小后大，`02_generate.py --pilot 50` 同时量吞吐。v1 用 27B、并发 6，约 720 输出 tok/s。
- 改写模型的腔调越接近以后要改的 AI 文章越好；做不到时，`style_mimic` prompt 用目标文章做示例能补一部分。

## 本地部署

- vLLM：`06_merge.py` 合并后 `serve_vllm.sh`，`08_rewrite.py` 走 `serve.base_url`。
- 桌面端（Mac 或单张 NVIDIA 卡）：用 llama.cpp 的 `convert_hf_to_gguf.py --outtype q8_0` 把合并后的权重转成 GGUF（9B 约 9.1G，转换环境需要 transformers ≥ 5.5），用 llama.cpp 的 `llama-server -m <gguf> --alias <id> --port 8091 -c 16384 -np 4 -ngl 99 --jinja` 起服务，`voice-lora rewrite --card` 指向它。不要用 LM Studio：它的重复惩罚只看最近 64 个 token，也不接受请求里的窗口参数，改写会明显偏保守。在 M3 Ultra 上 4 并发改写一篇 40 段的文章约 45 秒。按这套训练数据训出来的模型不会输出思考内容，不需要额外开关。
- 模型文件大，别让它们进备份：LoRA、合并权重和 GGUF 放在备份范围之外，或在备份配置里单独排除。
