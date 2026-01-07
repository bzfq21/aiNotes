# Transformer 架构详解

## 概述

Transformer 是由 Vaswani 等人在 2017 年提出的深度学习架构，通过**自注意力机制 (Self-Attention)** 彻底改变了自然语言处理领域。

### 核心贡献
- 完全基于注意力机制，摒弃了传统的循环和卷积结构
- 实现了**并行计算**，大幅提升了训练效率
- 在机器翻译等任务上达到了 SOTA 性能

## 架构概览

```
输入嵌入 → 位置编码 → [编码器] → [解码器] → 输出
         ↑         ↑          ↑
       堆叠N次   堆叠N次    堆叠N次
```

### 编码器-解码器结构

**编码器 (Encoder)**:
- 处理输入序列，生成上下文表示
- 由 N 层相同的子层组成（通常 N=6）

**解码器 (Decoder)**:
- 基于编码器输出生成目标序列
- 同样由 N 层相同的子层组成

---

## 核心组件

### 1. 自注意力机制 (Self-Attention)

#### 基本原理
自注意力机制让序列中的每个位置都能**关注整个序列**的所有其他位置。

#### 数学公式

对于查询向量 Q、键向量 K 和值向量 V：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- $Q, K, V \in \mathbb{R}^{n \times d_k}$
- $d_k$ 是向量的维度
- $\frac{1}{\sqrt{d_k}}$ 是缩放因子，防止点积过大导致梯度消失

#### 计算步骤

1. **生成 Q, K, V**
   ```python
   Q = X @ W_Q  # Query
   K = X @ W_K  # Key
   V = X @ W_V  # Value
   ```

2. **计算注意力分数**
   ```python
   scores = Q @ K.T / sqrt(d_k)
   ```

3. **应用 Softmax**
   ```python
   attention_weights = softmax(scores, axis=-1)
   ```

4. **加权求和**
   ```python
   output = attention_weights @ V
   ```

### 2. 多头注意力 (Multi-Head Attention)

#### 核心思想
将 Q、K、V 分别投影到不同的子空间，并行计算多个注意力，然后拼接。

#### 公式
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

其中：
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

#### 参数配置
- 头数 $h$：通常为 8
- 每头维度 $d_k = d_v = d_{model} / h$ (例如 $d_{model}=512$ 时，$d_k=64$)

#### 代码示例
```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # 生成 Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        # 应用 mask（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 和加权求和
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(output)
```

### 3. 位置编码 (Positional Encoding)

#### 为什么需要位置编码？
自注意力机制本身**不包含位置信息**，需要显式注入序列顺序。

#### 公式

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$ 是位置索引
- $i$ 是维度索引

#### 代码实现
```python
import math
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]
```

### 4. 前馈神经网络 (Feed-Forward Network)

#### 结构
每个编码器和解码器层都包含一个位置级的前馈网络：

$$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

#### 参数配置
- 输入/输出维度：$d_{model}$
- 隐藏层维度：$d_{ff} = 4 \times d_{model}$（例如 2048）

#### 代码示例
```python
class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
```

### 5. 残差连接和层归一化 (Residual Connection & LayerNorm)

#### 结构
每个子层周围都应用了残差连接和层归一化：

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

#### 重要性
- **残差连接**：缓解梯度消失，加速训练
- **层归一化**：稳定训练，加速收敛

#### 代码实现
```python
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

### 6. 掩码 (Masking)

#### 填充掩码 (Padding Mask)
忽略序列中的填充位置：

```python
def create_padding_mask(seq, pad_idx):
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)
```

#### 前瞻掩码 (Look-ahead Mask)
解码器中防止看到未来信息（因果掩码）：

```python
def create_lookahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 1  # True 表示需要被 mask 的位置
```

---

## 完整架构

### 编码器层 (Encoder Layer)

```
输入
  ↓
[多头自注意力 + 残差连接 + 层归一化]
  ↓
[前馈网络 + 残差连接 + 层归一化]
  ↓
输出
```

### 解码器层 (Decoder Layer)

```
输入 + 编码器输出
  ↓
[带掩码的多头自注意力 + 残差连接 + 层归一化]
  ↓
[编码器-解码器注意力 + 残差连接 + 层归一化]
  ↓
[前馈网络 + 残差连接 + 层归一化]
  ↓
输出
```

---

## 训练细节

### 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| $d_{model}$ | 512 | 模型维度 |
| $d_{ff}$ | 2048 | 前馈网络隐藏层维度 |
| Heads | 8 | 多头注意力头数 |
| N | 6 | 编码器/解码器层数 |
| Dropout | 0.1 | Dropout 比率 |
| Label Smoothing | 0.1 | 标签平滑参数 |

### 优化器

使用 **Adam** 优化器，学习率采用**预热调度器**：

$$lrate = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})$$

#### 代码实现
```python
class NoamOpt:
    "学习率预热优化器"
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "更新参数和学习率"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "实现上面的学习率公式"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))
```

### 正则化技巧

1. **Dropout**：在每个子层的输出和残差连接后应用
2. **标签平滑**：减少模型对训练数据的过度自信

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)
```

---

## Transformer 变体

### 1. BERT (Bidirectional Encoder Representations from Transformers)

- **类型**：仅编码器
- **特点**：双向上下文理解，适合分类、问答等任务
- **预训练任务**：掩码语言模型（MLM）+ 下句预测（NSP）

### 2. GPT (Generative Pre-trained Transformer)

- **类型**：仅解码器
- **特点**：单向自回归生成，适合文本生成
- **预训练任务**：标准语言建模

### 3. T5 (Text-to-Text Transfer Transformer)

- **类型**：完整的编码器-解码器
- **特点**：统一所有 NLP 任务为文本到文本的格式
- **预训练任务**：去噪自编码

### 4. BART (Bidirectional and Auto-Regressive Transformers)

- **类型**：编码器-解码器
- **特点**：结合 BERT 和 GPT 的优势
- **预训练任务**：文本去噪

---

## 实践技巧

### 1. 梯度裁剪
防止梯度爆炸：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 混合精度训练
使用 FP16 加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. 学习率监控
验证学习率是否合理：

```python
for param_group in optimizer.param_groups:
    print(f"Current LR: {param_group['lr']}")
```

### 4. 常见问题与解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 损失不下降 | 学习率过大/过小 | 使用学习率调度器 |
| 内存溢出 | 批次大小过大 | 减小批次，使用梯度累积 |
| 训练不稳定 | 梯度爆炸 | 增加梯度裁剪 |
| 过拟合 | 模型过大/数据少 | 增加 Dropout，数据增强 |

---

## 数学细节

### 缩放点积注意力的动机

点积的大小随维度 $d_k$ 增长，导致 Softmax 进入**极小梯度区域**：

```python
# 大点积导致梯度消失
scores = Q @ K.T  # [n, n]
# 如果 d_k = 512，scores 均值约为 256
# softmax([256, ...]) 接近 [1, 0, 0, ...]，梯度几乎为 0
```

缩放 $\frac{1}{\sqrt{d_k}}$ 将点积均值保持在合理范围。

### 多头注意力的优势

1. **表示多样性**：每个头学习不同的注意力模式
2. **并行计算**：可以高效地在 GPU 上并行执行
3. **避免瓶颈**：单头注意力可能难以捕捉复杂关系

### 位置编码的性质

- **唯一性**：每个位置有唯一的编码
- **相对位置感知**：$PE_{pos+k}$ 可以通过线性变换从 $PE_{pos}$ 得到
- **可泛化**：可以处理比训练时更长的序列

---

## 扩展阅读

### 经典论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) (2018)
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (2020)

### 相关资源
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Hugging Face Transformers 库](https://github.com/huggingface/transformers)

---

## 总结

Transformer 的核心创新：
1. **自注意力机制**：建模长距离依赖
2. **多头注意力**：捕捉不同的语义关系
3. **位置编码**：注入序列顺序信息
4. **并行计算**：高效训练大规模模型

这些设计使其成为现代 NLP 的基石，并启发了后续的大量研究和应用。
