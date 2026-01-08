模型训练的标准步骤通常包括三个阶段：预训练（Pre-training）、SFT（监督微调）和 RLHF（基于人类反馈的强化学习）。

国产模型的训练流程常有以下特点：
- SFT 阶段多采用冷启动方式，训练样本规模相对较小但质量较高
  - DeepSeek R1：数千条高质量CoT数据，采用特殊结构化格式（|special_token|&lt;reasoning_process&gt;|special_token|&lt;summary&gt;）
  - Qwen：标准对话格式（&lt;|im_start|&gt;标记），数据经过两阶段严格过滤
  - GLM-4.5：标准对话格式，使用扩展CoT的小规模SFT数据
  - MiniMax：标准对话格式，多领域覆盖，数学和代码数据占比60%
  - Kimi 1.5：标准对话格式，包含规划、评估、反思和探索四种核心思考节点
- RLHF 阶段通过多轮训练迭代优化模型表现
  - DeepSeek R1采用GRPO（Group Relative Policy Optimization）算法，完全摒弃价值网络
  - Grok系列大规模强化学习训练，训练量达到前代模型的10-100倍
- 具体实现方法因模型和团队而异

DeepSeek R1的R1-Zero实验证明：跳过SFT直接使用RL激发模型推理能力是可行的，但存在语言混合、可读性差等问题。最终版本仍采用"冷启动SFT + RL"混合策略。
