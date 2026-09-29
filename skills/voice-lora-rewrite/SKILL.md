---
name: voice-lora-rewrite
description: 用训练好的 voice-lora 模型（例如已发布的鸭哥文风模型 ai_smell_remover），把一篇 AI 起草的中文 markdown 文章逐段改成作者的文风，并产出供人审阅的对照页。触发说法包括“voice rewrite”“voice lora 改写”“去 AI 味”“改成作者的文风”。覆盖起服务、调用、采样参数，以及改完之后必须做的事实漂移核查。模型输出不能直接当最终稿。训练和复现模型见 skills/voice-lora/SKILL.md。
---

# voice-lora-rewrite：用训练好的文风模型改一篇文章

## 元数据

- **类型**：Tool
- **适用场景**：手上有一篇 AI 起草、事实已经核对过的中文 markdown 文章，想让措辞和语气更像作者本人
- **模型**：已发布的 [ai_smell_remover](https://huggingface.co/grapeot/ai_smell_remover)（鸭哥文风，Q8_0 GGUF），或者按 `skills/voice-lora/SKILL.md` 用自己的文章训练的模型
- **服务**：OpenAI 兼容的服务。GGUF 用 llama.cpp 的 `llama-server`，合并权重用 vLLM；不要用 LM Studio，原因见“采样参数”
- **代码入口**：`voice-lora` 命令（`uv pip install git+https://github.com/grapeot/voice-lora`，或在仓库里 `uv pip install -e .`）：`rewrite`（改写）、`compare`（对照页）、`score`（可选打分）
- **输出位置**：`--out` 指定的文件，同名 `.blocks.jsonl` 是逐段日志

## 它做什么，不做什么

模型逐段改写正文段落，只改措辞和句子，不改段落结构、论证顺序和事实。标题、引用块、表格、代码块和图片原样保留；段落里的链接在改写后按锚文本找回原位，找不到的挂在段末。

还有一层保险：如果某段改写后数字对不上，或者长度比超出了 0.6–1.6，就重试一次，还不行就保留原文。

不做的事：

- 不核查事实。AI 写出来的错误还是会照抄，所以要基于核查过事实的稿件进行润色。
- 不改结构。AI 文章的结构和套路还是在。
- 不适用于作者自己的文字。它的目标是将 AI 的写作方式向作者靠拢，所以对作者自己的文字只会适得其反。

## 用法

### 1. 准备模型和服务

**用已发布的鸭哥文风模型（ai_smell_remover）**，从零开始：

```bash
# 下载权重（9.1 GB）和调用卡
hf download grapeot/ai_smell_remover ai_smell_remover-v1.1-Q8_0.gguf voice-lora-card.yaml --local-dir ./ai_smell_remover

# 装 llama.cpp：Mac 用 brew install llama.cpp；Linux、Windows 用 llama.cpp 的 release 包，或者自己编译
llama-server -m ./ai_smell_remover/ai_smell_remover-v1.1-Q8_0.gguf --alias ai_smell_remover-v1.1 \
  --port 8091 -c 16384 -np 4 -ngl 99 --jinja
curl -s http://127.0.0.1:8091/health          # {"status":"ok"} 即可

# 装 voice-lora
uv pip install git+https://github.com/grapeot/voice-lora
```

- `-np 4` 开 4 个并行槽位，`-c` 是 4 个槽位合计的上下文长度，`-ngl 99` 把所有层放到 GPU 上。
- 下载下来的 `voice-lora-card.yaml` 就是调用卡，已经指向上面这个服务，不用改。
- 目前只在 Mac（M3 Ultra，Metal 后端）上测试过。llama.cpp 本身支持 Linux 和 Windows（CPU、CUDA 等后端），换平台时只需换成对应平台的 llama.cpp。9B 的 Q8_0 需要大约 10 GB 的显存或统一内存。

**用自己训练的模型**：GGUF 同样用 `llama-server` 起服务；合并权重用 vLLM，按 `skills/voice-lora/references/environment.md` 起服务（合并后的权重，或者基座加 `--lora-modules`）。

### 2. 准备调用卡

调用卡是一个小 YAML，里面写着服务地址、服务端的模型 id、训练时用的指令和采样参数（格式见 `card.example.yaml`）。用发布的模型时，直接用下载下来的 `voice-lora-card.yaml`。用自己的模型时，从训练这个模型的 config 生成，这样 prompt 和训练时一致：

```bash
voice-lora card --config <训练时用的 config> --model <identifier> --base-url http://127.0.0.1:8091/v1 --out model.yaml
```

### 3. 改写

```bash
voice-lora rewrite /path/to/article.md --card voice-lora-card.yaml --out /path/to/article.voice.md
```

- 输出是一行 JSON，其中 `units` 是改写的单元数量，`kept_original` 是被保险退回的原文段落，在审阅时需要检查。
- 终端中的 `links appended` 指的是没能找回原位、附加到段落末尾的链接。
- 对于多篇文章，可以使用 `--out-dir DIR` 而不是 `--out`，文件名格式为 `<文章名>.<模型名>.md`。

### 4. 出对照页，人工读

```bash
voice-lora compare /path/to/article.md /path/to/article.voice.md
```

页面是原文和改写并排展示，显示改写前后分数，AI 腔词用波浪线标出。默认用打包附带的分类器（以鸭哥为参照，见 `docs/classifier.md`），自己训练的分类器加 `--classifier`。多个改写文件就是多栏对比。

### 5.（可选）打分

```bash
voice-lora score /path/to/article.md /path/to/article.voice.md
```

P(作者) 只说明措辞像不像，适合比较同一篇文章改写前后，不能证明改得对，见 `docs/classifier.md`。

## 改动量：应该是多少

正确使用的时候，对一个 40 段已经改过的文章逐段和原文比，平均相似度大约 0.78–0.81，基本上没有原样返回的段落。在 AI 的原始初稿上改得更多。注意模型的目标是“内容和顺序不变，只换措辞”，所以不会大改句式和结构。

如果相似度在 0.85 以上，或者原样返回的段落明显变多，几乎可以肯定是重复惩罚没有按下面的方式生效。最常见的原因是用了 LM Studio，或者自己发请求时漏了参数。

想让它改得更多：

- **喂 AI 原始初稿，不要喂已经润色过的版本。**
- **不要调高温度。** temperature 1.0 会编出原文没有的英文引语、夸大判断的强度，数字保险拦不住。
- 可以多跑一两次，逐段挑。

## 采样参数：别改

调用卡和 `voice-lora rewrite` 默认 temperature 0.7、top_p 0.95、重复惩罚 1.05，惩罚覆盖整个 prompt。自己写客户端时，用 `voice_lora.rewrite.request_body` 生成请求体。

- **重复惩罚必须覆盖整个 prompt。** 要改写的原文就在 prompt 里，惩罚压低照抄原文的概率，这是模型肯换说法的主要原因。
  - vLLM 的 `repetition_penalty` 默认就作用于全部 prompt 和输出。
  - llama.cpp 默认只看最近 64 个 token（`repeat_last_n`），同样的 1.05 在它上面几乎不起作用，所以请求里要带 `repeat_last_n: 4096`。
  - 同 14 段、各采样 6 次，和原文的相似度：vLLM 0.786；llama-server 覆盖全文 0.784；llama-server 默认窗口 0.849；不加惩罚 0.901。
- **不要用 LM Studio。** 它加载 GGUF 用的也是 llama.cpp，但它只认 `repeat_penalty`（不认 `repetition_penalty`），而且不把 `repeat_last_n` 传下去，所以做不到覆盖全文的惩罚，改写明显偏保守（相似度 0.84–0.89）。
- **不要用 greedy（temperature 0）。** greedy 下模型大多在照抄输入。
- 同一篇文章每次跑出来都不一样，每段改成什么样是随机的。多跑一两次再逐段挑，比调高温度安全：默认温度跑两次、每段取分类器分数高的那版，P(作者) 从 0.11–0.12 提到 0.15。

## 改完之后必须做的：事实漂移核查

**模型输出不能直接作为最终稿交给用户。** 模型只会更改措辞，但会带来事实漂移。数字保险只针对数字和长度，下面这些它都无法防范。调用方在完成改写后必须执行事实漂移核查，一旦发现有漂移就要做局部修改（surgical edit）改回去，才能把结果作为最终稿。

### 1. 逐段对照

逐段日志在 `<输出>.blocks.jsonl` 里，每行的 `input` 是改写前的段落，`output` 是模型原始输出。逐行对照这两者，也可以打开 `voice-lora compare` 生成的对照页。重点查：

| 查什么 | 实际出现过的例子 |
|---|---|
| 数字、日期、金额、百分比 | 保险一般会拦住，但仍要看单位和修饰词（“约”“最高”）有没有丢 |
| 人名、机构名、产品名，以及它们之间的关系 | “被 Salesforce 出资 36 亿美元收购的 Fin”变成“对 Salesforce 投入 36 亿美元收购的 Fin” |
| 判断强度 | “以更大的倍数膨胀”变成“呈指数级增长”；“可能”变成“一定”；增删“据估计”“尚未确认” |
| 新增内容 | 编出原文没有的引语，尤其是整句英文引语，并安到某个机构头上 |
| 删掉的内容 | 删掉半句论点，或者删掉一个限定条件 |
| 意思变了 | “同样的动力还在别处运转”变成“推动了其他领域的发展”；“训练目标是区分 A 和 B”变成“用 A 和 B 来训练” |
| 格式 | 模型自己加上列表符号（`*`、`-`）或加粗，把完整的句子拆成列表碎片，或者去掉原有的加粗小标题 |
| 人称和口吻 | 说明文变成第一人称（“我的博客”“判成我写的”），元数据的中文字段名换成英文 |
| 行内代码 | 命令、路径、参数名的反引号被去掉，或者内容被改；逐个比对改写前后的行内代码 |

把中文说法改成英文术语（例如 ROI、task），通常是作者的习惯，不算漂移。

### 2. 局部改回，不要整段重跑

- 仅修改发生漂移的个别字/半句，恢复为改写前的表达，其他部分保留改写后的措辞。直接在 markdown 文件上编辑，在原文中找到对应位置后进行替换。
- 当一段中漂移过多，无法局部改回时，整段换回改写前的原文。
- 不要让模型反复重改同一段来“碰运气”，因为每次重改都会引入新的漂移。
- 结构和格式上的改动不要照单全收。模型偶尔会把散文段拆成列表或编号，或者加上、去掉加粗。除非明显更好，否则保留改写后的文字，把结构恢复成原稿的样子（例如把列表拼回段落），不要因为格式变了就连文字一起换回原稿。

### 3. 统一排版和前后衔接

改写是一段一段地给模型的，每段只给上一段做参考。所以改过的段落和没改的段落用的是两套不同的写法，混在一篇文章里面：

- **排版**：模型会模仿作者的排版习惯，比如直引号 `"`、半角括号、中英文之间不加空格、“AI味”不加空格。没改写的部分还是原稿的写法：标题、表格、代码块，以及紧跟标题下面、中间没有空行的列表（这种块整块被当成标题原样保留）。实测一份 skill 文档里，被改写的 6 行全部出现了这类不一致。
- **术语**：同一个词在不同段落写法不同，比如“AI 腔词”和“AI腔词”，“真人作者”和“其他作者”。
- **衔接**：改写过的段落和前后未改的段落之间，可能出现连接词重复、指代变化（“该分类器”变成“分类器”），或者小节的起承转合断开。

做法：选定全文原本的写法（一般是原稿），把改写过的段落改成一致。不要全局替换（行内代码、`P(作者)` 等符号、型号名、链接地址都要保持原样；例如把数字之间的连字符统一成破折号时，Qwen3.5-9B 会被改坏）。改完后通读一遍段落之间的衔接。

### 4. 汇报

向用户说明改动了几个段落，做了哪些局部修正（逐条列出原文、模型输出、修正后），以及哪几段整段换回了原文。

## 速度参考

同一篇 40 段、约 4,000 个输出 token 的文章：

| 部署 | 耗时 | 速度 |
|---|---|---|
| Mac M3 Ultra，llama-server，Q8_0 GGUF，并发 4 | 约 45 秒 | 约 85 tok/s |
| 单张 RTX 5090，vLLM bf16 | 约 6 秒 | 约 650 tok/s |

两个部署用的都是同一个模型，greedy 且不加重复惩罚时，40 个段落中有 25 个完全相同，平均相似度为 0.99。如果换了部署，可以用同样的方法先检查一下：`--temperature 0 --repetition-penalty 1.0 --no-guard`，用两个部署分别跑一遍同一篇文章，逐段比较。
