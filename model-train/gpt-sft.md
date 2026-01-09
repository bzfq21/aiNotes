# GPT 监督微调 (Supervised Fine-Tuning) 详解

## 概述

监督微调 (Supervised Fine-Tuning, SFT) 是将预训练的 GPT 模型在高质量的指令-响应对数据集上进行进一步训练的过程。这是 RLHF (Reinforcement Learning from Human Feedback) 流程的第一阶段，让模型更好地理解和遵循自然语言指令。

### 重要性

- **指令遵循能力**：让模型能够理解和执行用户指令
- **任务适应性**：针对特定任务进行优化
- **安全性**：减少有害输出，提高响应质量
- **效率**：相比全量微调，使用参数高效方法 (PEFT) 降低计算成本

### SFT 在 RLHF 中的位置

```
预训练 (PT) → 监督微调 (SFT) → 奖励模型训练 (RM) → 强化学习 (RL)
```

---

## 数据准备

### 数据集格式

SFT 数据集通常包含指令-响应对：

```json
[
  {
    "instruction": "解释什么是机器学习",
    "input": "",
    "output": "机器学习是人工智能的一个分支..."
  },
  {
    "instruction": "写一首关于春天的诗",
    "input": "",
    "output": "春风拂面百花开..."
  }
]
```

### 数据质量要求

1. **多样性**：覆盖多种任务类型和领域
2. **准确性**：响应应该正确、有帮助
3. **安全性**：避免有害或偏见内容
4. **一致性**：风格和质量保持一致

### 数据处理流程

```python
def format_instruction_response(example):
    """格式化指令-响应对"""
    if example.get("input", "") == "":
        formatted_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    else:
        formatted_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"

    return {"text": formatted_text}
```

---

## 微调方法

### 1. 完全微调 (Full Fine-tuning)

更新所有模型参数，效果最好但计算成本高。

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
training_args = TrainingArguments(
    output_dir="./gpt2-sft",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_steps=500,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)
trainer.train()
```

### 2. 参数高效微调 (PEFT)

#### LoRA (Low-Rank Adaptation)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # 秩
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

#### QLoRA (Quantized LoRA)

```python
from peft import prepare_model_for_kbit_training

# 4-bit 量化
model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    load_in_4bit=True,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

### 3. 使用 SFTTrainer

Hugging Face 的 TRL 库提供了专门的 SFTTrainer：

```python
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("json", data_files="sft_data.jsonl")

# 训练配置
training_args = SFTConfig(
    output_dir="./gpt2-sft-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    max_seq_length=512,
    packing=True,  # 序列打包，加速训练
)

# LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model="gpt2",
    args=training_args,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    formatting_func=format_instruction_response,
    max_seq_length=512,
)

trainer.train()
```

---

## 训练配置

### 关键超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| learning_rate | 2e-5 ~ 5e-4 | 学习率，LoRA 可以用更高值 |
| batch_size | 4 ~ 16 | 批大小，根据显存调整 |
| num_epochs | 1 ~ 3 | 训练轮数，避免过拟合 |
| max_seq_length | 512 ~ 2048 | 最大序列长度 |
| gradient_accumulation_steps | 4 ~ 16 | 梯度累积步数 |

### 学习率调度

```python
from transformers import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=total_steps
)
```

### 混合精度训练

```python
training_args = TrainingArguments(
    fp16=True,  # 半精度训练
    bf16=False,  # 或使用 bf16 (如果支持)
)
```

---

## 评估与验证

### 评估指标

1. **困惑度 (Perplexity)**：衡量模型预测准确性
2. **ROUGE/BLEU**：针对生成任务
3. **人类评估**：指令遵循质量、安全性

### 验证策略

```python
def evaluate_model(model, eval_dataset, tokenizer):
    evaluator = Evaluator(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
    )

    results = evaluator.evaluate()
    print(f"Perplexity: {results['perplexity']}")
    return results
```

### 过拟合检测

监控训练和验证损失：

```python
training_args = TrainingArguments(
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
```

---

## 最佳实践

### 1. 数据质量优先

- 使用高质量、人工标注的数据
- 多样化指令类型和领域
- 避免数据污染（训练数据泄露到测试）

### 2. 参数高效方法

- 对于大模型，优先使用 LoRA/QLoRA
- r=8~16，lora_alpha=16~32
- 只微调 attention 和 feed-forward 层

### 3. 训练稳定性

- 使用梯度裁剪：`max_grad_norm=1.0`
- 学习率预热和衰减
- 监控训练曲线，避免过拟合

### 4. 计算优化

- 使用 DeepSpeed 或 FSDP 进行分布式训练
- 启用梯度检查点：`gradient_checkpointing=True`
- 序列打包减少 padding 开销

### 5. 模型融合

训练多个 LoRA 适配器，然后融合：

```python
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# 加载多个 LoRA 适配器
model1 = PeftModel.from_pretrained(base_model, "path/to/lora1")
model2 = PeftModel.from_pretrained(base_model, "path/to/lora2")

# 融合适配器
merged_model = model1.merge_and_unload()
merged_model = PeftModel.from_pretrained(merged_model, "path/to/lora2")
final_model = merged_model.merge_and_unload()
```

---

## 完整训练脚本

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig

def main():
    # 模型和数据配置
    model_name = "gpt2-medium"
    dataset_path = "path/to/sft_dataset.json"

    # 加载数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # LoRA 配置
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc", "c_proj"]  # GPT 架构特定
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2-sft-output",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        fp16=True,
        gradient_checkpointing=True,
    )

    # 初始化训练器
    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_instruction_response,
        max_seq_length=1024,
        packing=True,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    trainer.save_model("./gpt2-sft-final")

if __name__ == "__main__":
    main()
```

---

## 常见问题与解决方案

### Q1: 训练损失不下降？

**可能原因**：
- 学习率过低
- 数据质量差
- 模型容量不足

**解决方案**：
- 提高学习率
- 检查和清理数据
- 使用更大模型

### Q2: 出现梯度爆炸？

**解决方案**：
```python
# 添加梯度裁剪
training_args = TrainingArguments(
    max_grad_norm=1.0,
)
```

### Q3: 如何选择 LoRA 参数？

- **r**：4~64，越高效果越好但参数越多
- **lora_alpha**：通常为 r 的 2 倍
- **target_modules**：针对 attention 层最有效

### Q4: SFT vs RLHF？

- **SFT**：学习指令遵循，使用监督学习
- **RLHF**：学习人类偏好，使用强化学习
- **建议**：先做 SFT，再做 RLHF

---

## 扩展阅读

### 经典论文
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (2022)
- [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (2021)

### 相关资源
- [Hugging Face TRL 文档](https://huggingface.co/docs/trl)
- [PEFT 库](https://huggingface.co/docs/peft)
- [OpenAI SFT 指南](https://platform.openai.com/docs/guides/supervised-fine-tuning)

---

## 总结

SFT 是将预训练 GPT 模型转化为有用 AI 助手的桥梁。通过精心准备的数据和合适的训练方法，可以显著提升模型的指令遵循能力和任务表现。

关键要点：
1. **数据质量**决定微调效果
2. **PEFT 方法**降低训练成本
3. **超参数调优**需要实验验证
4. **评估完整性**确保模型可靠性

随着模型规模的增加，SFT 的重要性将进一步凸显，成为 AI 对齐的关键技术。