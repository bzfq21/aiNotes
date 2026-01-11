# PPO (Proximal Policy Optimization) 算法详解

## 1. 数学理论基础

### 1.1 强化学习基础框架

在强化学习中，我们考虑一个马尔可夫决策过程 (MDP) 定义为五元组：

$$M = (\mathcal{S}, \mathcal{A}, P, r, \gamma)$$

其中：
- $\mathcal{S}$ 是状态空间
- $\mathcal{A}$ 是动作空间  
- $P(s'|s,a)$ 是状态转移概率
- $r(s,a)$ 是即时奖励函数
- $\gamma \in [0,1]$ 是折扣因子

### 1.2 策略$\pi_\theta$的数学定义与特性

**策略 $\pi_\theta$ 的基本定义**:
- **$\pi_\theta(a|s)$**: 参数为 θ 的随机策略，表示在状态 s 下采取动作 a 的概率
- **θ ∈ ℝ^d**: 策略参数向量，通常通过神经网络实现
- **约束条件**: $\sum_{a\in\mathcal{A}} \pi_\theta(a|s) = 1$, $∀s \in \mathcal{S}$

**常见实现形式**:

1. **Softmax策略（分类动作空间）**:
   $$\pi_\theta(a|s) = \frac{\exp(f_\theta(s,a))}{\sum_{a'\in\mathcal{A}} \exp(f_\theta(s,a'))}$$
   其中 $f_\theta(s,a)$ 是神经网络输出

2. **高斯策略（连续动作空间）**:
   $$\pi_\theta(a|s) = \mathcal{N}(a | \mu_\theta(s), \sigma_\theta^2(s))$$
   其中 $\mu_\theta(s)$ 和 $\sigma_\theta(s)$ 由神经网络输出

3. **策略梯度中的对数导数**:
   $$\nabla_\theta \log \pi_\theta(a|s) = \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)}$$

### 1.3 策略梯度定理的完整推导

#### 1.3.1 期望回报的定义

策略梯度方法的目标是最大化期望回报。明确定义期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

其中 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ 是轨迹，$R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$ 是轨迹的折扣回报

轨迹概率分布为：

$$P(\tau|\theta) = \rho_0(s_0) \prod_{t=0}^{\infty} P(s_{t+1}|s_t,a_t) \pi_\theta(a_t|s_t)$$

其中 $\rho_0(s_0)$ 是初始状态分布，$P(s_{t+1}|s_t,a_t)$ 是环境转移概率。

#### 1.3.2 梯度推导：从基础定义开始

对 $\theta$ 求梯度：

$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau|\theta) R(\tau) d\tau$$

由于积分和微分可交换（在适当正则条件下）：

$$\nabla_\theta J(\theta) = \int \nabla_\theta P(\tau|\theta) R(\tau) d\tau$$

使用对数导数技巧：$\nabla f = f \nabla \log f$，得到：

$$\nabla_\theta P(\tau|\theta) = P(\tau|\theta) \nabla_\theta \log P(\tau|\theta)$$

因此：

$$\nabla_\theta J(\theta) = \int P(\tau|\theta) \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau|\theta) R(\tau)]$$

#### 1.3.3 展开轨迹概率的对数

展开 $\log P(\tau|\theta)$：

$$\log P(\tau|\theta) = \log \rho_0(s_0) + \sum_{t=0}^{\infty} [\log P(s_{t+1}|s_t,a_t) + \log \pi_\theta(a_t|s_t)]$$

对 $\theta$ 求导时，只有 $\pi_\theta(a_t|s_t)$ 依赖于 $\theta$，因此：

$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

#### 1.3.4 得到初步梯度表达式

#### 1.3.5 利用因果结构简化

利用因果结构，我们知道 $r_k$ 只依赖于 $(s_t,a_t)$ 对于 $t \leq k$。因此可以写成：

$$\nabla_\theta J(\theta) = \sum_{t=0}^{\infty} \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \mathbb{E}[\sum_{k=t}^{\infty} \gamma^k r_k | s_t, a_t]]$$

定义状态-动作价值函数：

$$Q^{\pi_\theta}(s,a) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{k=0}^{\infty} \gamma^k r_{t+k} \Big| s_t=s, a_t=a\right]$$

因此：

$$\nabla_\theta J(\theta) = \sum_{t=0}^{\infty} \mathbb{E}_{s_t,a_t \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t Q^{\pi_\theta}(s_t,a_t)]$$

#### 1.3.6 引入基线减除降低方差

注意到对于任意函数 $b(s)$ 不依赖于 $a$：

$$\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}[\nabla_\theta \log \pi_\theta(a|s) b(s)] = b(s) \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0$$

因为 $\sum_{a \in \mathcal{A}} \pi_\theta(a|s) = 1$。

选择基线为状态价值函数：

$$V^{\pi_\theta}(s) = \sum_{a \in \mathcal{A}} \pi_\theta(a|s) Q^{\pi_\theta}(s,a)$$

因此可以写成：

$$\nabla_\theta J(\theta) = \sum_{t=0}^{\infty} \mathbb{E}_{s_t,a_t \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \gamma^t (Q^{\pi_\theta}(s_t,a_t) - V^{\pi_\theta}(s_t))]$$

#### 1.3.7 最终策略梯度定理

定义优势函数：

$$A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)$$

得到策略梯度定理的最终形式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

#### 1.3.8 离散时间有限范围版本

对于有限时间范围 $T$，策略梯度定理简化为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t)\right]$$

#### 1.3.9 连续时间版本

对于连续时间，考虑随机策略梯度：

$$\nabla_\theta J(\theta) = \int_0^{T} \gamma^t \mathbb{E}_{s_t \sim \pi_\theta}\left[\int_{\mathcal{A}} \nabla_\theta \pi_\theta(a|s_t) Q^{\pi_\theta}(s_t,a) da\right] dt$$

使用平稳分布 $\rho^{\pi_\theta}(s)$：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^{\pi_\theta}, a \sim \pi_\theta(\cdot|s)}[\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)]$$

#### 1.3.10 推导总结与关键洞察

策略梯度推导的关键洞察：

1. **对数导数技巧**：$\nabla f = f \nabla \log f$ 将梯度转换为对数形式
2. **因果分解**：利用马尔可夫性质，将未来奖励与当前决策分离
3. **基线减除**：通过减去状态价值函数降低方差，不改变期望
4. **期望表示**：将梯度表示为期望形式，便于蒙特卡洛估计

最终，策略梯度定理的核心是：**策略梯度等于对数策略梯度与优势函数的期望乘积**。

## 2. PPO 目标函数的数学推导

### 2.1 重要性采样 (Importance Sampling)

PPO 使用重要性采样来处理策略更新中的分布偏移问题。定义重要性权重：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

### 2.2 替代目标函数

PPO 的核心思想是构造一个替代目标函数，限制策略更新步长。定义替代目标：

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

其中 $\epsilon$ 是一个超参数（通常取 0.1-0.3），$\hat{A}_t$ 是估计的优势函数。

### 2.3 数学推导过程

考虑策略 $\pi_\theta$ 和 $\pi_{\theta_{\text{old}}}$ 之间的 KL 散度约束优化问题：

$$\begin{align}
&\max_\theta \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a)\right] \\
&\text{s.t. } \mathbb{E}_{s \sim \pi_{\theta_{\text{old}}}}\left[\text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right] \leq \delta
\end{align}$$

使用拉格朗日乘子法，我们得到：

$$\max_\theta \mathbb{E}_{s,a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s,a) - \beta \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))\right]$$

## 3. PPO 算法的核心数学公式

### 3.1 替代目标函数

完整的 PPO 目标函数：

$$L^{\text{PPO}}(\theta) = \mathbb{E}_t\left[L^{\text{CLIP}}(\theta) - c_1 L_t^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t)\right]$$

其中：
- $L^{\text{CLIP}}(\theta)$ 是剪切替代目标
- $L_t^{\text{VF}}(\theta) = (V_\theta(s_t) - V_t^{\text{targ}})^2$ 是价值函数损失
- $S[\pi_\theta](s_t)$ 是策略熵奖励
- $c_1, c_2$ 是超参数

### 3.2 剪切机制详解

剪切函数 $\text{clip}(r, 1-\epsilon, 1+\epsilon)$ 的数学表达：

$$\text{clip}(r, 1-\epsilon, 1+\epsilon) = 
\begin{cases}
1-\epsilon, & r < 1-\epsilon \\
r, & 1-\epsilon \leq r \leq 1+\epsilon \\
1+\epsilon, & r > 1+\epsilon
\end{cases}$$

### 3.3 广义优势估计 (GAE)

优势函数的估计使用 GAE：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}^{V}$$

其中 $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

## 4. 逐步算法推导

### 4.1 策略更新步骤

1. **收集经验**：使用当前策略 $\pi_{\theta_{\text{old}}}$ 收集轨迹 $\tau = \{(s_t, a_t, r_t)\}_{t=0}^{T-1}$

2. **计算优势**：
   - 计算 TD 误差：$\delta_t = r_t + \gamma V_{\phi}(s_{t+1}) - V_{\phi}(s_t)$
   - 计算 GAE：$\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}$
   - 标准化优势：$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu}{\sigma}$

3. **优化策略**：
   - 对于 $K$ 个 epoch：
     - 计算重要性权重：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
     - 计算目标：$L^{\text{CLIP}}(\theta) = \frac{1}{T}\sum_{t=0}^{T-1}\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)$
     - 更新 $\theta$ 使用梯度上升

4. **更新价值函数**：
   - 计算目标价值：$V_t^{\text{targ}} = \hat{A}_t + V_{\phi}(s_t)$
   - 最小化价值函数损失：$L^{\text{VF}}(\phi) = \frac{1}{T}\sum_{t=0}^{T-1}(V_{\phi}(s_t) - V_t^{\text{targ}})^2$

### 4.2 收敛性分析

PPO 的收敛性由以下不等式保证：

$$J(\theta) - J(\theta_{\text{old}}) \geq \frac{1}{1-\gamma}\mathbb{E}_{s \sim \rho^{\pi_{\theta_{\text{old}}}}}\left[\sum_a \pi_\theta(a|s) A^{\pi_{\theta_{\text{old}}}}(s,a)\right] - \frac{2\gamma\epsilon^{\text{max}}}{(1-\gamma)^2}\alpha^2$$

其中 $\epsilon^{\text{max}} = \max_{s,a} |A^{\pi_{\theta_{\text{old}}}}(s,a)|$，$\alpha = \max_s \text{TV}(\pi_\theta(\cdot|s), \pi_{\theta_{\text{old}}}(\cdot|s))$

## 5. 数学符号说明

| 符号 | 含义 |
|------|------|
| $\pi_\theta(a\|s)$ | 参数为 $\theta$ 的策略 |
| $V_\phi(s)$ | 参数为 $\phi$ 的价值函数 |
| $A^{\pi}(s,a)$ | 策略 $\pi$ 下的优势函数 |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE 参数 |
| $\epsilon$ | PPO 剪切参数 |
| $r_t(\theta)$ | 重要性权重 |
| $\delta_t$ | TD 误差 |
| $\text{KL}(\cdot\|\cdot)$ | KL 散度 |

## 6. 算法伪代码

```
输入：环境 Env，策略网络 π_θ，价值网络 V_φ
超参数：γ, λ, ε, K, B, c₁, c₂

循环直到收敛：
    1. 收集经验：
       使用 π_{θ_old} 收集轨迹 {(s_t, a_t, r_t)}_{t=0}^{T-1}
    
    2. 计算优势：
       δ_t ← r_t + γ V_φ(s_{t+1}) - V_φ(s_t)
       Â_t ← Σ_{l=0}^{T-t-1}(γλ)^l δ_{t+l}
       Â_t ← (Â_t - μ_A)/σ_A
    
    3. 策略更新：
       对于 epoch = 1 到 K：
           对于 minibatch = 1 到 B：
               r_t(θ) ← π_θ(a_t|s_t)/π_{θ_old}(a_t|s_t)
               L^{CLIP} ← min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)
               θ ← θ + η ∇_θ L^{CLIP}
    
    4. 价值更新：
       V_t^{targ} ← Â_t + V_φ(s_t)
       φ ← φ - η_φ ∇_φ (V_φ(s_t) - V_t^{targ})^2
    
    5. 更新旧策略：
       θ_old ← θ
```

## 7. 理论保证

### 7.1 策略改进下界

PPO 的理论保证基于策略改进下界理论。考虑KL散度约束下的策略优化问题：

**定理**（PPO 策略改进下界）：对于任意策略 $\pi_{\theta}$ 和 $\pi_{\theta_{\text{old}}}$，如果满足：

$$\max_s \text{KL}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s)) \leq \delta$$

则有策略改进下界：

$$J(\theta) - J(\theta_{\text{old}}) \geq \frac{1}{1-\gamma}\mathbb{E}_{s \sim \rho^{\pi_{\theta_{\text{old}}}}}\left[\sum_a \pi_\theta(a|s) A^{\pi_{\theta_{\text{old}}}}(s,a)\right] - \frac{2\gamma}{(1-\gamma)^2}\epsilon_{\text{max}}\delta$$

其中 $\epsilon_{\text{max}} = \max_{s,a} |A^{\pi_{\theta_{\text{old}}}}(s,a)|$ 是最大优势值，$\rho^{\pi_{\theta_{\text{old}}}}(s)$ 是折扣状态分布。

### 7.2 收敛性保证

**收敛条件**：
为确保策略改进，需要满足：
$$\mathbb{E}_{s \sim \rho^{\pi_{\theta_{\text{old}}}}}\left[\sum_a \pi_\theta(a|s) A^{\pi_{\theta_{\text{old}}}}(s,a)\right] \geq \frac{2\gamma\epsilon_{\text{max}}}{1-\gamma}\delta$$

**单调性保证**：
在满足KL约束 $\delta$ 的条件下，PPO算法保证策略性能单调不减，即：
$$J(\theta_{k+1}) \geq J(\theta_k)$$

### 7.3 样本复杂度

PPO的样本复杂度为 $\tilde{O}(\frac{1}{\epsilon^2(1-\gamma)^2})$，其中 $\epsilon$ 是目标精度，$\gamma$ 是折扣因子。