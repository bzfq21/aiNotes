# 深度学习基础

## 什么是深度学习

深度学习是机器学习的一个分支，它使用多层人工神经网络来模拟人脑神经元的工作方式，通过从训练数据中学习特征表示，最终实现模式识别和预测任务。

深度学习网络的理论基础是**通用近似定理**：由于激活函数的存在，单层神经网络理论上可以拟合任何连续函数，多层神经网络则可以学习更复杂的特征层次结构。

但实际应用中，深度学习面临两个主要挑战：
- **数据需求**：需要大量标注数据才能避免过拟合
- **计算资源**：深层网络需要强大的计算能力进行训练

## 问题类型与模型设计

深度学习模型的设计必须与实际问题相匹配，不同类型的问题需要采用不同的网络结构和损失函数。常见问题类型包括：

### 1. 分类问题（Classification）

**问题描述**：预测输入属于哪个离散类别
**典型应用**：图像分类、文本分类、情感分析

**损失函数详细推导过程**：

### 步骤1：信息论基础

**熵的定义**：衡量随机变量的不确定性
$$H(P) = -\sum_{i=1}^{k} P(i)\log P(i)$$

**相对熵（KL散度）**：衡量两个概率分布的差异
$$D_{KL}(P||Q) = \sum_{i=1}^{k} P(i)\log\frac{P(i)}{Q(i)}$$

**交叉熵**：$H(P,Q) = H(P) + D_{KL}(P|Q) = -\sum_{i=1}^{k} P(i)\log Q(i)$

### 步骤2：最大似然估计框架

**问题设定**：
- 输入样本：$x_i \in \mathbb{R}^d$，真实标签：$y_i \in \{1,2,...,k\}$
- 网络输出：$\hat{y}_i = [p_1, p_2, ..., p_k]$，其中 $p_j = P(y=j|x_i; \theta)$
- 约束：$\sum_{j=1}^{k} p_j = 1$

**独热编码（One-hot Encoding）**：
真实标签 $y_i$ 编码为向量 $[y_{i1}, y_{i2}, ..., y_{ik}]$，其中：
$$y_{ij} = \begin{cases} 
1 & \text{if } y_i = j \\
0 & \text{otherwise}
\end{cases}$$

### 步骤3：似然函数推导

**单个样本的概率**：
$$p(y_i|x_i; \theta) = \prod_{j=1}^{k} \hat{y}_{ij}^{y_{ij}}$$

这是因为：当 $y_{ij} = 1$ 时，该项等于 $\hat{y}_{ij}$；当 $y_{ij} = 0$ 时，该项等于 1。

**全部样本的似然函数**（假设独立同分布）：
$$L(\theta) = \prod_{i=1}^{n} p(y_i|x_i; \theta) = \prod_{i=1}^{n} \prod_{j=1}^{k} \hat{y}_{ij}^{y_{ij}}$$

### 步骤4：对数似然函数

**取对数**（将乘法转为加法，简化计算）：
$$\log L(\theta) = \sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log \hat{y}_{ij}$$

**最大化对数似然**：
$$\max_{\theta} \sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log \hat{y}_{ij}$$

### 步骤5：损失函数转换

**机器学习习惯**：通常最小化损失函数，而非最大化似然
$$\min_{\theta} -\sum_{i=1}^{n} \sum_{j=1}^{k} y_{ij} \log \hat{y}_{ij}$$

**归一化**：除以样本数量 $n$，得到平均损失
$$L_{CE}(\theta) = -\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{k} y_{ij}\log(\hat{y}_{ij})$$

### 步骤6：与交叉熵的等价性证明

**真实分布**：$P_i = [y_{i1}, y_{i2}, ..., y_{ik}]$（独热向量）
**预测分布**：$Q_i = [\hat{y}_{i1}, \hat{y}_{i2}, ..., \hat{y}_{ik}]$

**交叉熵计算**：
$$H(P_i, Q_i) = -\sum_{j=1}^{k} y_{ij} \log \hat{y}_{ij}$$

**总损失**：所有样本交叉熵的平均
$$L_{CE} = \frac{1}{n}\sum_{i=1}^{n} H(P_i, Q_i)$$

因此，**交叉熵损失 = 负对数似然**，两者完全等价。

### 2. 回归问题（Regression）

**问题描述**：预测连续数值
**典型应用**：房价预测、股票价格预测、温度预测

**损失函数详细推导过程**：

### 步骤1：概率建模假设

**线性回归模型**：
$$y = f(x; \theta) + \epsilon = \theta^T x + \epsilon$$

**噪声假设**：观测误差服从高斯分布
$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

**条件概率**：给定输入 $x$，输出 $y$ 的概率分布
$$p(y|x; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \theta^T x)^2}{2\sigma^2}\right)$$

### 步骤2：似然函数构建

**独立同分布假设**：$n$ 个训练样本 $\{(x_i, y_i)\}_{i=1}^n$

**联合概率密度**：
$$p(y_1, y_2, ..., y_n|x_1, x_2, ..., x_n; \theta) = \prod_{i=1}^{n} p(y_i|x_i; \theta)$$

**似然函数**（作为 $\theta$ 的函数）：
$$L(\theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \theta^T x_i)^2}{2\sigma^2}\right)$$

### 步骤3：对数似然函数

**取自然对数**：
$$\begin{align}
\log L(\theta) &= \sum_{i=1}^{n} \log\left[\frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - \theta^T x_i)^2}{2\sigma^2}\right)\right] \\
&= \sum_{i=1}^{n} \left[-\frac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - \theta^T x_i)^2}{2\sigma^2}\right] \\
&= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \theta^T x_i)^2
\end{align}$$

### 步骤4：最大化对数似然

**目标函数**：
$$\max_{\theta} \log L(\theta) = \max_{\theta} \left[-\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \theta^T x_i)^2\right]$$

**简化**：第一项与 $\theta$ 无关，第二项系数为负
$$\max_{\theta} \log L(\theta) \equiv \min_{\theta} \sum_{i=1}^{n}(y_i - \theta^T x_i)^2$$

### 步骤5：损失函数形式化

**最小化残差平方和**：
$$J(\theta) = \sum_{i=1}^{n}(y_i - f(x_i; \theta))^2$$

**归一化版本**（常用形式）：
$$L_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

其中 $\hat{y}_i = f(x_i; \theta)$ 是模型预测值。

### 步骤6：与最小二乘法的联系

**最小二乘法**：寻找 $\theta$ 使残差平方和最小
$$\theta^* = \arg\min_{\theta} \sum_{i=1}^{n}(y_i - \theta^T x_i)^2$$

**闭式解**（当 $f(x) = \theta^T x$ 为线性函数时）：
$$\theta^* = (X^T X)^{-1} X^T y$$

**重要结论**：在噪声服从高斯分布的假设下，**最小二乘法 = 最大似然估计**，两者完全等价。

### 3. 聚类问题（Clustering）

**问题描述**：无监督地发现数据中的自然分组
**典型应用**：客户分群、文档主题聚类

**损失函数详细推导过程**（以K-Means为例）：

### 步骤1：问题形式化定义

**输入**：数据集 $\{x_1, x_2, ..., x_n\}$，其中 $x_i \in \mathbb{R}^d$
**目标**：将数据分为 $k$ 个簇，找到最优聚类中心 $\{\mu_1, \mu_2, ..., \mu_k\}$
**分配变量**：$c_i \in \{1, 2, ..., k\}$ 表示样本 $x_i$ 所属的簇

### 步骤2：目标函数构建

**簇内方差定义**：衡量簇内数据点的紧密程度
$$W(C) = \sum_{j=1}^{k} \sum_{i \in C_j} ||x_i - \mu_j||^2$$

其中：
- $C_j$：第 $j$ 个簇的样本集合
- $\mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i$：第 $j$ 个簇的中心

### 步骤3：优化问题的等价形式

**引入指示函数**：
$$r_{ij} = \begin{cases} 
1 & \text{如果样本 } x_i \text{ 属于簇 } j \\
0 & \text{otherwise}
\end{cases}$$

**约束条件**：
- 每个样本只属于一个簇：$\sum_{j=1}^{k} r_{ij} = 1, \forall i$
- 每个簇至少有一个样本：$\sum_{i=1}^{n} r_{ij} > 0, \forall j$

**目标函数重写**：
$$J(\{r_{ij}\}, \{\mu_j\}) = \sum_{i=1}^{n}\sum_{j=1}^{k} r_{ij} ||x_i - \mu_j||^2$$

### 步骤4：交替优化算法（EM思想）

**E步（Expectation）**：固定聚类中心，更新样本分配
$$r_{ij}^{(t+1)} = \begin{cases} 
1 & \text{if } j = \arg\min_{l} ||x_i - \mu_l^{(t)}||^2 \\
0 & \text{otherwise}
\end{cases}$$

**M步（Maximization）**：固定样本分配，更新聚类中心
$$\mu_j^{(t+1)} = \frac{\sum_{i=1}^{n} r_{ij}^{(t+1)} x_i}{\sum_{i=1}^{n} r_{ij}^{(t+1)}}$$

### 步骤5：收敛性证明

**单调性**：每一步都减少或保持目标函数值
$$J^{(t+1)} \leq J^{(t)}$$

**有限性**：可能的分配方式有限，算法必在有限步内收敛

### 步骤6：评估指标推导（轮廓系数）

**簇内凝聚度**：样本 $i$ 与同簇其他样本的平均距离
$$a(i) = \frac{1}{|C_{c_i}| - 1}\sum_{j \in C_{c_i}, j \neq i} ||x_i - x_j||$$

**簇间分离度**：样本 $i$ 与最近簇的平均距离
$$b(i) = \min_{k \neq c_i} \frac{1}{|C_k|}\sum_{j \in C_k} ||x_i - x_j||$$

**轮廓系数**：
$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}$$

**全局轮廓系数**：所有样本轮廓系数的平均值
$$\text{Silhouette Score} = \frac{1}{n}\sum_{i=1}^{n} s(i)$$

**注意**：轮廓系数是**评估指标**，不是要优化的损失函数，用于衡量聚类质量的好坏。

### 4. 降维问题（Dimensionality Reduction）

**问题描述**：将高维数据映射到低维空间，保留关键信息
**典型应用**：数据可视化、特征提取、噪声消除

**不同方法的损失函数详细推导**：

### 1. 自编码器（Autoencoder）

**网络结构**：
- 编码器：$h = f(x) = \sigma(Wx + b)$，其中 $h \in \mathbb{R}^d$（低维表示）
- 解码器：$\hat{x} = g(h) = \sigma(W'h + b')$，其中 $\hat{x} \in \mathbb{R}^D$（重构）

**重构误差推导**：

**步骤1：MSE损失函数**
$$L_{recon} = \frac{1}{n}\sum_{i=1}^{n} ||x_i - \hat{x}_i||^2 = \frac{1}{n}\sum_{i=1}^{n} ||x_i - g(f(x_i))||^2$$

**步骤2：概率解释**
假设每个维度独立，噪声服从高斯分布：
$$p(x|\hat{x}) = \prod_{d=1}^{D} \mathcal{N}(x_d; \hat{x}_d, \sigma^2)$$

**步骤3：最大似然估计**
$$\max \sum_{i=1}^{n} \log p(x_i|\hat{x}_i) \equiv \min \sum_{i=1}^{n} ||x_i - \hat{x}_i||^2$$

**步骤4：正则化增强**
$$L_{total} = L_{recon} + \lambda L_{reg}$$

其中 $L_{reg}$ 可以是：
- 稀疏性正则：$L_{reg} = ||h||_1$（鼓励稀疏表示）
- 正交正则：$L_{reg} = ||WW^T - I||_F^2$（鼓励编码器的正交性）

### 2. 主成分分析（PCA）

**数据准备**：数据中心化 $\tilde{X} = X - \mu\mathbf{1}^T$

**步骤1：协方差矩阵**
$$C = \frac{1}{n}\tilde{X}\tilde{X}^T \in \mathbb{R}^{D \times D}$$

**步骤2：特征值分解**
找到前 $d$ 个最大特征值对应的特征向量：
$$Cw_j = \lambda_j w_j, \quad j = 1, 2, ..., d$$

**投影矩阵**：$W = [w_1, w_2, ..., w_d] \in \mathbb{R}^{D \times d}$

**步骤3：投影目标**
低维表示：$Z = \tilde{X}^TW$

**步骤4：重构**
$$\hat{X} = ZW^T + \mu\mathbf{1}^T$$

**步骤5：投影误差**
$$J(W) = ||\tilde{X} - \tilde{X}WW^T||_F^2$$

**步骤6：Frobenius范数展开**
$$\begin{align}
||\tilde{X} - \tilde{X}WW^T||_F^2 &= \text{tr}(\tilde{X}^T\tilde{X}) - 2\text{tr}(\tilde{X}^T\tilde{X}WW^T) + \text{tr}(W^T\tilde{X}^T\tilde{X}WW^T)\\
&= \text{tr}(\tilde{X}^T\tilde{X}) - 2\text{tr}(W^T\tilde{X}^T\tilde{X}W) + \text{tr}(W^T\tilde{X}^T\tilde{X}W)\\
&= \text{tr}(\tilde{X}^T\tilde{X}) - \text{tr}(W^T\tilde{X}^T\tilde{X}W)
\end{align}$$

**步骤7：最优化**（约束 $W^TW = I$）
$$\max_W \text{tr}(W^T\tilde{X}^T\tilde{X}W)$$

这等价于找到协方差矩阵 $\tilde{X}^T\tilde{X}$ 的前 $d$ 个最大特征值。

### 3. t-SNE（t-Distributed Stochastic Neighbor Embedding）

**步骤1：相似度定义**
原始空间中点对 $(i,j)$ 的相似度：
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2/2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2/2\sigma_i^2)}$$

**对称化**：
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**步骤2：低维相似度**
低维空间中点对 $(i,j)$ 的相似度：
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

**步骤3：KL散度损失**
$$L(Y) = \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}}$$

**步骤4：梯度推导**
对损失函数关于低维坐标 $y_i$ 求偏导：
$$\frac{\partial L}{\partial y_i} = 4\sum_{j \neq i} (p_{ij} - q_{ij})(1 + ||y_i - y_j||^2)^{-1}(y_i - y_j)$$

**步骤5：优化目标**
$$\min_Y \sum_{i \neq j} p_{ij} \log\frac{p_{ij}}{q_{ij}}$$

**参数 $\sigma_i$ 的选择**：通过二元搜索找到合适的 $\sigma_i$ 使得条件熵 $H(P_i) = -\sum_j p_{j|i}\log p_{j|i}$ 等于预定义的困惑度。

## 模型选择指导原则

1. **先理解问题本质**：有监督vs无监督，离散vs连续
2. **选择合适架构**：CNN（图像）、RNN（序列）、Transformer（注意力）
3. **设计损失函数**：基于概率假设和信息论原理
4. **考虑计算约束**：模型复杂度vs效果权衡

深度学习的核心思想是**端到端学习**：通过梯度下降直接优化最终的损失函数，让网络自动学习最适合的特征表示。