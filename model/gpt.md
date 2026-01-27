# GPT (Generative Pre-trained Transformer) 详解

## 概述

GPT 是基于 Transformer 解码器架构的自回归语言模型，通过**大规模预训练 + 微调**的两阶段范式，在自然语言处理和生成任务上取得了突破性进展。

### 发展历程

| 版本 | 发布时间 | 参数量 | 训练数据 | 核心贡献 |
|------|----------|--------|----------|----------|
| GPT-1 | 2018.06 | 117M | BooksCorpus (5GB) | 首次展示 GPT 架构潜力 |
| GPT-2 | 2019.02 | 1.5B | WebText (40GB) | 展示大规模语言模型能力 |
| GPT-3 | 2020.05 | 175B | Common Crawl (500B tokens) | Few-shot learning 能力 |
| GPT-3.5 | 2022.11 | - | - | ChatGPT 对话能力 |
| GPT-4 | 2023.03 | - | 多模态数据 | 多模态与复杂推理能力 |

---

## 核心架构

### 1. 基础架构

GPT 采用 **仅解码器 (Decoder-only)** 的 Transformer 架构：

```
输入嵌入 → 位置编码 → [Transformer Decoder] × N → 输出
                     ↑
                  掩码自注意力
```

### 2. 与原始 Transformer 的区别

| 特性 | 原始 Transformer | GPT |
|------|------------------|-----|
| 架构 | 编码器 + 解码器 | 仅解码器 |
| 注意力 | 双向（编码器）+ 掩码（解码器） | 全部掩码（单向） |
| 训练目标 | 编码器-解码器注意力 | 自回归语言建模 |
| 应用场景 | 机器翻译 | 文本生成、分类等 |

### 3. 关键组件

#### 3.1 掩码自注意力 (Masked Self-Attention)

确保每个位置只能看到**之前的位置**：

$$\text{Attention}_i = \text{softmax}\left(\frac{Q_i K_{1:i}^T}{\sqrt{d_k}}\right) V_{1:i}$$

#### 3.2 代码实现

```python
import torch
import torch.nn as nn
import math

class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT 使用 GELU 而不是 ReLU
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络 + 残差连接 + 层归一化
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x
```

---

## 预训练阶段

### 1. 训练目标：自回归语言建模

给定序列 $x = (x_1, x_2, ..., x_n)$，最大化对数似然：

$$\mathcal{L}(\theta) = \sum_{i=1}^n \log P(x_i | x_{<i}; \theta)$$

### 2. 模型输出

对于位置 $i$，模型输出下一个 token 的概率分布：

$$P(x_i | x_{<i}) = \text{softmax}(W_o h_i)$$

其中 $h_i$ 是第 $i$ 个位置的隐藏状态。

### 3. 训练流程

```python
def train_step(model, input_ids, optimizer, criterion):
    """
    单个训练步骤
    input_ids: [batch_size, seq_len]
    """
    # 创建因果掩码
    seq_len = input_ids.size(1)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)

    # 前向传播
    logits = model(input_ids, mask=causal_mask)  # [batch_size, seq_len, vocab_size]

    # 准备目标（下一个 token）
    targets = input_ids[:, 1:]  # [batch_size, seq_len-1]
    logits = logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]

    # 计算损失
    loss = criterion(logits.reshape(-1, logits.size(-1)),
                     targets.reshape(-1))

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### 4. 关键超参数

| 参数 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| 层数 | 12 | 48 | 96 |
| 隐藏层维度 | 768 | 1600 | 12288 |
| 注意力头数 | 12 | 25 | 96 |
| 词表大小 | 40478 | 50257 | 50257 |
| 序列长度 | 512 | 1024 | 2048 |

---

## 微调阶段

### 1. 微调方法

对于下游任务（如分类），在序列末尾添加任务特定的层：

```python
class GPTClassifier(nn.Module):
    def __init__(self, gpt_model, num_classes):
        super().__init__()
        self.gpt = gpt_model
        self.classifier = nn.Linear(gpt_model.d_model, num_classes)

    def forward(self, input_ids):
        # 获取最后一个 token 的输出
        outputs = self.gpt(input_ids)  # [batch_size, seq_len, d_model]
        last_hidden = outputs[:, -1, :]  # [batch_size, d_model]

        # 分类
        logits = self.classifier(last_hidden)  # [batch_size, num_classes]
        return logits
```

### 2. 微调策略

#### 完全微调 (Full Fine-tuning)
- 更新所有模型参数
- 需要大量标注数据
- 效果最好但计算成本高

#### 冻结 + 微调 (Freeze + Fine-tune)
- 冻结预训练层的参数
- 只微调顶层或任务特定层
- 数据少时有效，但性能有限

#### LoRA (Low-Rank Adaptation)
```python
class LoRALayer(nn.Module):
    """低秩适应层"""
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(original_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original_layer.out_features))

    def forward(self, x):
        # 原始输出 + LoRA 增量
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * (self.alpha / self.rank)
```

---

## 推理与生成

### 1. 自回归生成

生成文本时，**逐 token** 地预测下一个词：

```python
def generate(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50):
    """
    文本生成函数
    """
    model.eval()

    # 编码提示词
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出
            outputs = model(input_ids)
            logits = outputs[:, -1, :]  # 最后一个位置

            # 应用温度
            logits = logits / temperature

            # Top-k 采样
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[0, indices[0]] = -float('inf')

            # 转换为概率
            probs = torch.softmax(logits, dim=-1)

            # 采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接到输入
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 检查结束符
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

### 2. 采样策略

#### 2.1 贪婪采样 (Greedy Sampling)
选择概率最大的词，但可能导致重复和缺乏多样性。

```python
next_token = torch.argmax(logits, dim=-1, keepdim=True)
```

#### 2.2 Top-k 采样
只从概率最高的 k 个词中采样。

```python
top_k = 50
values, indices = torch.topk(logits, top_k)
logits = torch.full_like(logits, -float('inf'))
logits.scatter_(1, indices, values)
```

#### 2.3 核采样 (Nucleus Sampling / Top-p)
从累计概率达到 p 的最小词集中采样。

```python
top_p = 0.9
sorted_probs, sorted_indices = torch.sort(probs, descending=True)
cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
mask = cumsum_probs > top_p
sorted_probs[mask] = 0
sorted_probs = sorted_probs / sorted_probs.sum()

next_token = torch.multinomial(sorted_probs, 1)
next_token = sorted_indices.gather(1, next_token)
```

#### 2.4 温度 (Temperature)
控制输出的随机性：
- 温度低：更确定，更保守
- 温度高：更随机，更多样

```python
logits = logits / temperature  # 温度范围通常在 0.7-1.0 之间
```

---

## GPT 的关键特性

### 1. Few-shot / Zero-shot Learning

**Zero-shot**: 不给示例，直接完成任务

```
输入: "把这句话翻译成英文：你好世界"
输出: "Hello, world"
```

**Few-shot**: 给少量示例

```
输入:
"把这句话翻译成英文：你好世界 -> Hello, world"
"把这句话翻译成英文：再见 -> Goodbye"
"把这句话翻译成英文：谢谢 ->"

输出: "Thank you"
```

### 2. 指令微调 (Instruction Tuning)

通过在**指令数据集**上微调，让模型更好地理解自然语言指令：

```python
instruction_data = [
    {
        "instruction": "总结以下文章",
        "input": "...",
        "output": "..."
    },
    {
        "instruction": "解释这个概念",
        "input": "量子力学",
        "output": "..."
    }
]
```

### 3. RLHF (Reinforcement Learning from Human Feedback)

**三阶段流程**：

1. **有监督微调 (SFT)**: 在高质量指令-响应对上微调
2. **奖励模型 (RM)**: 训练一个模型评估人类偏好
3. **强化学习 (RL)**: 使用 PPO 优化模型，最大化奖励

```python
# PPO 训练伪代码
for batch in dataloader:
    # 收集生成
    responses = model.generate(prompts)

    # 计算奖励
    rewards = reward_model(prompts, responses)

    # 计算优势
    advantages = compute_advantages(rewards)

    # PPO 更新
    loss = ppo_loss(model, old_model, prompts, responses, advantages)
    loss.backward()
    optimizer.step()
```

---

## 完整 GPT 模型实现

```python
import torch
import torch.nn as nn
import math

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 块
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享
        self.output_layer.weight = self.token_embedding.weight

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)

    def forward(self, input_ids, mask=None):
        batch_size, seq_len = input_ids.shape

        # 位置索引
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = x * math.sqrt(self.d_model)

        # 创建因果掩码（如果未提供）
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            mask = mask.to(input_ids.device)

        # Transformer 块
        for block in self.blocks:
            x = block(x, mask)

        # 输出层
        x = self.ln_f(x)
        logits = self.output_layer(x)

        return logits
```

---

## 训练技巧

### 1. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    with autocast():
        logits = model(input_ids)
        loss = criterion(logits, targets)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

### 2. 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    with autocast():
        logits = model(batch)
        loss = criterion(logits, targets) / accumulation_steps

    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. 学习率调度

```python
# Cosine 学习率衰减
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, 100000)
```

---

## 优化方向

### 1. 架构改进

| 改进 | 说明 |
|------|------|
| Flash Attention | 优化注意力计算，减少内存访问 |
| Rotary Positional Embedding | 更好的位置编码 |
| Grouped Query Attention | 减少注意力头数，提升效率 |

### 2. 训练优化

| 技术 | 说明 |
|------|------|
| DeepSpeed | 分布式训练框架 |
| Megatron-LM | 大规模模型并行训练 |
| FSDP | 全速分片数据并行 |

### 3. 推理优化

| 技术 | 说明 |
|------|------|
| KV Cache | 缓存键值对，加速生成 |
| 量化 | FP16/INT8 量化，减少显存 |
| Speculative Decoding | 小模型预测，大模型验证 |

---

## 应用场景

### 1. 文本生成
- 创意写作
- 代码生成
- 文档撰写

### 2. 对话系统
- 聊天机器人
- 客户服务
- 智能助手

### 3. 问答系统
- 开放域问答
- 知识检索

### 4. 内容创作
- 博客写作
- 营销文案
- 社交媒体内容

---

## 常见问题

### Q1: GPT 与 BERT 的主要区别？

**GPT**:
- 仅解码器架构
- 自回归（单向）生成
- 适合文本生成任务

**BERT**:
- 仅编码器架构
- 双向理解
- 适合理解任务（分类、NER 等）

### Q2: 如何选择 GPT 的参数？

**小模型 (≤ 1B)**:
- 资源受限
- 特定任务微调

**中模型 (1B-10B)**:
- 通用任务
- 平衡性能和成本

**大模型 (≥ 10B)**:
- Few-shot / Zero-shot
- 复杂推理任务

### Q3: 微调时应该冻结哪些层？

- **数据少**：冻结底层，只微调顶层
- **数据多**：微调所有层
- **计算受限**：使用 LoRA 或 Adapter

### Q4: 如何处理长文本？

- **滑动窗口**：分段处理
- **层次模型**：长文本摘要 + 细节
- **长上下文架构**：Longformer、Linear Attention

---

## 扩展阅读

### 经典论文
- [Improving Language Understanding by Generative Pre-Training (GPT-1)](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) (2018)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) (2020)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) (2022)

### 相关资源
- [OpenAI API 文档](https://platform.openai.com/docs/api-reference)
- [nanoGPT](https://github.com/karpathy/nanoGPT) - 最小化 GPT 实现
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [LLaMA](https://ai.meta.com/blog/large-language-model-meta-ai/) - 开源大语言模型

---

## 总结

GPT 的核心优势：
1. **自回归生成**：流式生成高质量文本
2. **Few-shot 能力**：通过大规模预训练获得泛化能力
3. **任务泛化**：一个模型处理多种任务
4. **持续进化**：通过数据扩展和架构改进不断提升

GPT 的成功推动了整个大语言模型领域的发展，并为 AI 应用提供了强大的基础设施。
