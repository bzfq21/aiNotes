### GPT 模型的前向传播、损失计算与反向传播更新权重详解

GPT（Generative Pre-trained Transformer）模型基于 Transformer 解码器架构，主要用于自回归语言建模任务。下面我将详细解释 GPT 在前向传播后计算交叉熵损失，然后通过反向传播更新权重的过程。整个过程遵循深度学习的标准训练范式：前向传播计算输出和损失，反向传播计算梯度，优化器更新参数。

我将逐步解释每个部分，包括如何推导公式。假设我们处理一个序列长度为 \( T \) 的输入序列 \( \mathbf{x} = (x_1, x_2, \dots, x_T) \)，其中每个 \( x_t \) 是 token ID。GPT 的目标是预测下一个 token，即 \( p(x_{t+1} | x_1, \dots, x_t) \)。为了简化，我们假设词汇表大小为 \( V \)，模型有 \( L \) 层 Transformer 块。

#### 1. 前向传播（Forward Propagation）
前向传播从输入序列开始，计算模型的输出 logits（未归一化的概率分数）。

- **输入嵌入和位置编码**：
  输入 token \( x_t \) 通过嵌入矩阵 \( \mathbf{W}_e \in \mathbb{R}^{V \times d} \)（\( d \) 是隐藏维度）转换为嵌入向量 \( \mathbf{e}_t = \mathbf{W}_e[x_t] \)。
  加上位置编码 \( \mathbf{p}_t \)（通常是固定或可学习的），得到初始隐藏状态 \( \mathbf{h}_t^{(0)} = \mathbf{e}_t + \mathbf{p}_t \)。
  对于整个序列，初始隐藏矩阵 \( \mathbf{H}^{(0)} = [\mathbf{h}_1^{(0)}, \dots, \mathbf{h}_T^{(0)}] \in \mathbb{R}^{T \times d} \).

- **Transformer 层**：
  GPT 使用多层自注意力解码器。每层 \( l = 1 \) 到 \( L \) 包括：
  - 多头自注意力（Masked Multi-Head Attention）：防止未来信息泄露，使用下三角掩码。
    查询 \( \mathbf{Q} = \mathbf{H}^{(l-1)} \mathbf{W}_Q^{(l)} \)，键 \( \mathbf{K} = \mathbf{H}^{(l-1)} \mathbf{W}_K^{(l)} \)，值 \( \mathbf{V} = \mathbf{H}^{(l-1)} \mathbf{W}_V^{(l)} \)。
    注意力分数 \( \mathbf{A} = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M} \right) \)，其中 \( \mathbf{M} \) 是掩码（未来位置为 \( -\infty \)）。
    输出 \( \mathbf{Z}^{(l)} = \mathbf{A} \mathbf{V} \)（多头后concat并投影）。
  - 前馈网络（Feed-Forward）： \( \mathbf{FFN}(\mathbf{z}) = \max(0, \mathbf{z} \mathbf{W}_1^{(l)} + \mathbf{b}_1^{(l)}) \mathbf{W}_2^{(l)} + \mathbf{b}_2^{(l)} \)。
  - 层归一化和残差连接： \( \mathbf{H}^{(l)} = \text{LayerNorm}(\mathbf{H}^{(l-1)} + \mathbf{Z}^{(l)} + \mathbf{FFN}(\mathbf{Z}^{(l)})) \).

  最终输出隐藏状态 \( \mathbf{H}^{(L)} = [\mathbf{h}_1^{(L)}, \dots, \mathbf{h}_T^{(L)}] \).

- **输出 logits**：
  通过线性投影到词汇表： \( \mathbf{o}_t = \mathbf{h}_t^{(L)} \mathbf{W}_o + \mathbf{b}_o \)，其中 \( \mathbf{W}_o \in \mathbb{R}^{d \times V} \)。
  预测概率 \( \mathbf{p}_t = \text{softmax}(\mathbf{o}_t) \in \mathbb{R}^V \)，表示 \( p(x_{t+1} | x_1, \dots, x_t) \)。（注意：在自回归中，损失只计算从第二个 token 开始的预测。）

前向传播的整体函数可以表示为 \( \mathbf{o} = f(\mathbf{x}; \theta) \)，其中 \( \theta \) 是所有参数（如 \( \mathbf{W}_e, \mathbf{W}_Q^{(l)}, \dots \)）的集合。

#### 2. 交叉熵损失计算（Cross-Entropy Loss）
GPT 使用负对数似然（Negative Log-Likelihood）作为损失函数，等价于交叉熵。对于一个序列，目标标签 \( \mathbf{y} = (y_1, y_2, \dots, y_T) \)（其中 \( y_t = x_{t+1} \)，最后一个可能忽略），损失为：

\[
\mathcal{L}(\theta) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log p(y_t | x_1, \dots, x_{t-1}) = -\frac{1}{T-1} \sum_{t=1}^{T-1} \log (\mathbf{p}_{t-1}[y_t])
\]

其中 \( \mathbf{p}_{t-1} = \text{softmax}(\mathbf{o}_{t-1}) \)，\( \mathbf{o}_{t-1} \) 是位置 \( t-1 \) 的 logits。

在批量中，平均多个序列的损失。交叉熵的公式推导自信息论：它度量预测分布 \( \mathbf{p} \) 与真实 one-hot 分布 \( \mathbf{q} \)（\( q[y_t] = 1 \)，其余0）的差异：

\[
\mathcal{L} = -\sum_{v=1}^V q_v \log p_v = -\log p_{y_t}
\]

这是前向传播后的直接计算。

#### 3. 反向传播（Backpropagation）
反向传播使用链式法则计算损失 \( \mathcal{L} \) 相对于每个参数 \( \theta_i \) 的梯度 \( \frac{\partial \mathcal{L}}{\partial \theta_i} \)，从输出层向输入层传播。

- **基本原理**：深度网络是复合函数 \( \mathcal{L} = g(f(\mathbf{x}; \theta)) \)，梯度通过链式法则 \( \frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial o} \cdot \frac{\partial o}{\partial \theta} \) 计算。反向传播从 \( \frac{\partial \mathcal{L}}{\partial o} \) 开始，向后传递误差。

- **从损失到 logits 的梯度**：
  对于单个位置 \( t \) 的损失 \( \ell_t = -\log (\mathbf{p}_t[y_{t+1}]) \)，其中 \( \mathbf{p}_t = \text{softmax}(\mathbf{o}_t) \)。
  Softmax 的雅可比矩阵为 \( \frac{\partial p_v}{\partial o_u} = p_v (\delta_{vu} - p_u) \)，其中 \( \delta_{vu} = 1 \) 如果 \( v=u \)。
  因此，\( \frac{\partial \ell_t}{\partial \mathbf{o}_t} = \mathbf{p}_t - \mathbf{e}_{y_{t+1}} \)，其中 \( \mathbf{e}_{y} \) 是 one-hot 向量（\( e_y[y] = 1 \)，其余0）。
  
  推导过程：
  - \( \ell_t = -\log p_{y} \)，\( p_y = \frac{\exp(o_y)}{\sum_v \exp(o_v)} \)。
  - \( \frac{\partial \ell_t}{\partial o_u} = -\frac{\partial \log p_y}{\partial o_u} = - \frac{1}{p_y} \cdot \frac{\partial p_y}{\partial o_u} \)。
  - 如果 \( u = y \)，\( \frac{\partial p_y}{\partial o_y} = p_y (1 - p_y) \)，所以 \( \frac{\partial \ell_t}{\partial o_y} = p_y - 1 \)。
  - 如果 \( u \neq y \)，\( \frac{\partial p_y}{\partial o_u} = -p_y p_u \)，所以 \( \frac{\partial \ell_t}{\partial o_u} = p_u \)。
  - 整体：\( \frac{\partial \ell_t}{\partial \mathbf{o}_t} = \mathbf{p}_t - \mathbf{e}_y \)。
  
  对于整个序列，\( \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} = \frac{1}{T-1} (\mathbf{p}_t - \mathbf{e}_{y_{t+1}}) \)（从 t=0 开始索引调整）。

- **从 logits 到最终隐藏状态的梯度**：
  Logits 来自 \( \mathbf{o}_t = \mathbf{h}_t^{(L)} \mathbf{W}_o + \mathbf{b}_o \)。
  所以，\( \frac{\partial \mathcal{L}}{\partial \mathbf{h}_t^{(L)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} \mathbf{W}_o^\top \)。
  参数梯度：\( \frac{\partial \mathcal{L}}{\partial \mathbf{W}_o} = \sum_t (\mathbf{h}_t^{(L)})^\top \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} \)，\( \frac{\partial \mathcal{L}}{\partial \mathbf{b}_o} = \sum_t \frac{\partial \mathcal{L}}{\partial \mathbf{o}_t} \)（矩阵形式）。

- **在 Transformer 层中的反向传播**：
  从第 L 层向后：
  - 假设当前层输入为 \( \mathbf{H}^{(l-1)} \)，输出 \( \mathbf{H}^{(l)} \)。
  - 误差从 \( \frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(l)}} \) 传入（对于 l=L，是从上面计算的）。
  - LayerNorm 的梯度：LayerNorm 是 \( \text{LN}(\mathbf{z}) = \gamma \frac{\mathbf{z} - \mu}{\sigma} + \beta \)，其梯度涉及标准化反向。
  - FFN 的梯度：FFN 是两层线性 + ReLU，反向类似 MLP：从输出向输入传播，ReLU 梯度为 0 或 1。
  - 自注意力的梯度：最复杂部分。
    - 输出 \( \mathbf{Z} = \text{softmax}(\frac{\mathbf{QK}^\top}{\sqrt{d_k}} + \mathbf{M}) \mathbf{V} \)。
    - 梯度 \( \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \) 传入。
    - 向 V 的梯度：\( \frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \)。
    - 向 A 的梯度：\( \frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \mathbf{V}^\top \)。
    - Softmax 的梯度：\( \frac{\partial \mathcal{L}}{\partial \mathbf{S}} = \mathbf{A} \odot (\frac{\partial \mathcal{L}}{\partial \mathbf{A}} - \mathbf{A} \cdot \sum \frac{\partial \mathcal{L}}{\partial \mathbf{A}}) \)，其中 \( \mathbf{S} = \frac{\mathbf{QK}^\top}{\sqrt{d_k}} \)。
    - 然后向 Q 和 K：\( \frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \mathbf{K} / \sqrt{d_k} \)，\( \frac{\partial \mathcal{L}}{\partial \mathbf{K}} = (\frac{\partial \mathcal{L}}{\partial \mathbf{S}})^\top \mathbf{Q} / \sqrt{d_k} \)。
    - 最终向权重：\( \frac{\partial \mathcal{L}}{\partial \mathbf{W}_Q} = (\mathbf{H}^{(l-1)})^\top \frac{\partial \mathcal{L}}{\partial \mathbf{Q}} \)，类似 K、V。
  
  残差连接将梯度直接传递到前层：\( \frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(l-1)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{H}^{(l)}} + \) 来自注意力/FFN 的贡献。

- **到嵌入层的梯度**：
  继续向后，直到 \( \mathbf{H}^{(0)} \)，然后 \( \frac{\partial \mathcal{L}}{\partial \mathbf{W}_e} = \sum_t \frac{\partial \mathcal{L}}{\partial \mathbf{e}_t} \mathbf{e}_{x_t}^\top \)（one-hot 形式，或 embedding 的梯度更新）。

整个反向传播通过自动微分框架（如 PyTorch）实现，但手动推导就是逐层应用链式法则。

#### 4. 更新权重（Weight Update）
计算所有梯度 \( \nabla_\theta \mathcal{L} = \{\frac{\partial \mathcal{L}}{\partial \theta_i}\} \) 后，使用优化器更新参数。GPT 常用 Adam 或 AdamW（带权重衰减）。

- **基本梯度下降**：\( \theta_i \leftarrow \theta_i - \eta \frac{\partial \mathcal{L}}{\partial \theta_i} \)，其中 \( \eta \) 是学习率。
- **Adam 更新公式**（更常用）：
  Adam 维护一阶矩 \( \mathbf{m} \) 和二阶矩 \( \mathbf{v} \)（对每个参数）。
  - \( \mathbf{g} = \nabla_\theta \mathcal{L} \)（当前梯度）。
  - \( \mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \)。
  - \( \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \)。
  - 偏差校正：\( \hat{\mathbf{m}}_t = \mathbf{m}_t / (1 - \beta_1^t) \)，\( \hat{\mathbf{v}}_t = \mathbf{v}_t / (1 - \beta_2^t) \)。
  - 更新：\( \theta_t = \theta_{t-1} - \eta \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \)。
  
  典型超参数：\( \beta_1 = 0.9 \)，\( \beta_2 = 0.999 \)，\( \epsilon = 10^{-8} \)，\( \eta \approx 10^{-4} \) 到 \( 10^{-3} \)。
  AdamW 在更新前添加权重衰减：\( \theta \leftarrow \theta - \lambda \theta \)，其中 \( \lambda \) 是衰减率。




### 解释 Softmax + Cross-Entropy 中的雅可比矩阵使用

你提到了雅可比矩阵（Jacobian matrix），并质疑为什么用到它，因为 loss 函数的输出是标量（scalar）。这是一个很好的问题！确实，最终的损失 \( \mathcal{L} \) 或单个位置的 \( \ell_t = -\log p_y \) 是标量，但在反向传播中，我们需要计算这个标量相对于输入向量（ logits \(\mathbf{o}\) ）的梯度，这是一个向量：\( \frac{\partial \ell_t}{\partial \mathbf{o}} \in \mathbb{R}^V \)。为了得到这个向量梯度，我们必须通过链式法则处理中间的向量函数（softmax），而 softmax 的导数就是一个雅可比矩阵。

让我一步步详细推导，解释为什么需要雅可比，以及如何简化到最终的公式 \( \frac{\partial \ell_t}{\partial \mathbf{o}} = \mathbf{p} - \mathbf{e}_y \)。这在深度学习中很常见，尤其是在分类任务中，softmax + cross-entropy 的组合有高效的反向传播实现。

#### 1. 问题设置
- 输入：logits 向量 \( \mathbf{o} = (o_1, o_2, \dots, o_V) \in \mathbb{R}^V \)，V 是词汇表大小。
- Softmax 输出：概率向量 \( \mathbf{p} = \text{softmax}(\mathbf{o}) \in \mathbb{R}^V \)，其中 \( p_v = \frac{\exp(o_v)}{\sum_{u=1}^V \exp(o_u)} \)。
- 目标：one-hot 向量 \( \mathbf{e}_y \)，其中 \( e_y[y] = 1 \)，其余为 0（y 是真实标签）。
- 损失：交叉熵 \( \ell = -\sum_{v=1}^V e_{y,v} \log p_v = -\log p_y \)（因为 one-hot，只剩 \(-\log p_y\) ）。

注意：\( \ell \) 是标量，但它是复合函数：\( \ell = f(\mathbf{p}) \)，其中 \( f(\mathbf{p}) = -\log p_y \)（更一般是 \( -\mathbf{e}_y^\top \log \mathbf{p} \)），而 \( \mathbf{p} = g(\mathbf{o}) = \text{softmax}(\mathbf{o}) \)。所以，\( \ell = f(g(\mathbf{o})) \)。  

#### 2. 链式法则的应用
要计算梯度 \( \frac{\partial \ell}{\partial \mathbf{o}} \)（这是一个向量，每个分量 \( \frac{\partial \ell}{\partial o_u} \))，使用链式法则：

\[
\frac{\partial \ell}{\partial o_u} = \sum_{v=1}^V \frac{\partial \ell}{\partial p_v} \cdot \frac{\partial p_v}{\partial o_u}
\]

- 这里，\( \frac{\partial \ell}{\partial p_v} \) 是标量 loss 相对于向量 p 的梯度（一个向量 \( \in \mathbb{R}^V \))。
- \( \frac{\partial p_v}{\partial o_u} \) 是 softmax 函数的导数，对于每个 v 和 u 的偏导，形成一个矩阵：雅可比矩阵 \( \mathbf{J} \in \mathbb{R}^{V \times V} \)，其中 \( J_{vu} = \frac{\partial p_v}{\partial o_u} \)。
- 因此，向量形式：\( \frac{\partial \ell}{\partial \mathbf{o}} = \mathbf{J}^\top \frac{\partial \ell}{\partial \mathbf{p}} \)。（转置是因为链式法则是矩阵乘法。）

这就是为什么用到雅可比：softmax 是向量到向量的函数（\( \mathbb{R}^V \to \mathbb{R}^V \))，它的导数不是简单标量或向量，而是一个矩阵，表示输出每个分量对输入每个分量的敏感度。即使最终 loss 是标量，中间的复合需要这个矩阵来“传递”梯度。

#### 3. 计算 \( \frac{\partial \ell}{\partial \mathbf{p}} \)
首先，计算 loss 对 p 的梯度：

\[
\ell = -\sum_v e_{y,v} \log p_v = -\log p_y
\]

对于每个 v：

- 如果 v = y，\( \frac{\partial \ell}{\partial p_y} = -\frac{1}{p_y} \)。
- 如果 v ≠ y，\( \frac{\partial \ell}{\partial p_v} = 0 \)（因为 loss 只依赖 p_y）。

所以，\( \frac{\partial \ell}{\partial \mathbf{p}} = \left(0, \dots, -\frac{1}{p_y}, \dots, 0\right) \)，位置 y 为 -1/p_y，其余 0。向量形式：\( \frac{\partial \ell}{\partial \mathbf{p}} = -\frac{\mathbf{e}_y}{\mathbf{p}} \)（逐元素除法，但只有 y 位置非零）。

更精确：\( \left( \frac{\partial \ell}{\partial \mathbf{p}} \right)_v = -\frac{e_{y,v}}{p_v} \)，所以对于 v=y 是 \(-1/p_y\)，其余 0。

#### 4. 计算 Softmax 的雅可比矩阵 \( \mathbf{J} \)
现在，推导 \( \frac{\partial p_v}{\partial o_u} \)：

- \(p_v\) = \(exp(o_v) / Z\)，其中 \(Z = ∑ exp(o_u)\)。
- 如果 v = u：\( \frac{\partial p_v}{\partial o_u} = \frac{\exp(o_v) Z - \exp(o_v) \exp(o_v)}{Z^2} = p_v (1 - p_v) \)。
- 如果 v ≠ u：\( \frac{\partial p_v}{\partial o_u} = \frac{0 - \exp(o_v) \exp(o_u)}{Z^2} = -p_v p_u \)。
- 统一形式：\( \frac{\partial p_v}{\partial o_u} = p_v (\delta_{vu} - p_u) \)，其中 \(δ_{vu} = 1\) if v=u else 0。

这就是雅可比矩阵：\(J_{vu} = p_v (δ_{vu} - p_u)\)。它是一个对角矩阵减去外积：\(J = diag(p) - p p^\top\)。

#### 5. 组合得到 \( \frac{\partial \ell}{\partial \mathbf{o}} \)
现在，代入链式法则：

\[
\frac{\partial \ell}{\partial \mathbf{o}} = \mathbf{J}^\top \frac{\partial \ell}{\partial \mathbf{p}} = [\text{diag}(\mathbf{p}) - \mathbf{p} \mathbf{p}^\top]^\top \left( -\frac{\mathbf{e}_y}{\mathbf{p}} \right)
\]

由于 \(diag(p)\) 是对称的，且 \((p p^\top)^\top = p p^\top\)，所以 \(J^\top = diag(p) - p p^\top \)同 \(J\)。

但让我们直接计算（因为 V 可能大，但这是理论推导）：

\[
\frac{\partial \ell}{\partial o_u} = \sum_v \frac{\partial p_v}{\partial o_u} \cdot \frac{\partial \ell}{\partial p_v} = \sum_v p_v (\delta_{vu} - p_u) \left( -\frac{e_{y,v}}{p_v} \right) = -\sum_v (\delta_{vu} - p_u) e_{y,v}
\]

简化：

- \(∑_v δ_{vu} e_{y,v} = e_{y,u}\)（因为 δ 挑选 u=v）。
- \(∑_v -p_u e_{y,v} = -p_u ∑_v e_{y,v} = -p_u\)（因为 one-hot 和为1）。

所以，\( \frac{\partial \ell}{\partial o_u} = - (e_{y,u} - p_u) = p_u - e_{y,u} \)。  

向量形式：\( \frac{\partial \ell}{\partial \mathbf{o}} = \mathbf{p} - \mathbf{e}_y \)。  

这就是最终简化的公式！我们通过雅可比矩阵推导出来了，但实际实现中可以直接用这个向量，而不用显式构建 VxV 矩阵（因为 V 可能上万，构建矩阵太耗内存，直接计算这个简化形式更高效）。

#### 6. 为什么需要雅可比，即使 loss 是标量？
- Loss 是标量，但梯度是相对于向量的（o 是向量），所以必须处理向量函数的导数。
- 如果没有中间向量函数（如 sigmoid 在二分类，输出是标量），就不需要雅可比。但在多分类，softmax 输出向量，loss 输入向量，所以雅可比不可避免。
- 在自动微分框架（如 PyTorch）中，这一切是隐式的：框架自动计算雅可比并应用链式法则，你只需写 forward，backward 自动处理。

这个推导在 GPT 等模型中至关重要，因为每个位置的输出都是 softmax over V，V 很大（e.g., GPT-3 有 50k+ tokens）。如果有疑问，可以问具体哪步！