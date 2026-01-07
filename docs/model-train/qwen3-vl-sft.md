# Qwen3 VL 8B LoRA 微调笔记

## 概述

Qwen3 VL 是 Qwen 系列的多模态大语言模型，支持文本、图像和视频的联合理解和生成。本笔记专注于 Qwen3-VL-8B 模型的 LoRA (Low-Rank Adaptation) 微调实现。

### 模型特性

- **架构**: Transformer-based 多模态模型
- **参数量**: 8B
- **模态支持**: 文本、图像、视频
- **上下文长度**: 128K tokens
- **量化支持**: FP16, BF16, INT8, INT4

### LoRA 微调优势

- **参数效率**: 只训练少量参数 (< 1%)
- **内存友好**: 支持单卡训练 8B 模型
- **保持原能力**: 不破坏预训练知识
- **快速收敛**: 通常只需 1-3 轮训练

---

## 环境准备

### 依赖安装

```bash
# 基础环境
pip install torch torchvision torchaudio
pip install transformers accelerate peft

# Unsloth (推荐，优化速度)
pip install unsloth

# 或使用官方 Qwen 工具
pip install ms-swift

# 可选：DeepSpeed 加速
pip install deepspeed

# 可选：Flash Attention
pip install flash-attn --no-build-isolation
```

### 硬件要求

| 配置 | 推荐规格 |
|------|----------|
| GPU | RTX 3090/4090 或 A100 (24GB+) |
| RAM | 64GB+ |
| 存储 | 50GB+ (模型 + 数据) |

---

## 数据准备

### 数据集格式

Qwen3 VL 支持多模态对话数据：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "path/to/image.jpg"},
          {"type": "text", "text": "描述这张图片"}
        ]
      },
      {
        "role": "assistant",
        "content": "这是一张美丽的风景照片..."
      }
    ]
  }
]
```

### 数据集转换

```python
from datasets import Dataset, DatasetDict
import json

def convert_to_qwen_format(data_path):
    """转换自定义数据集为 Qwen 格式"""

    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    formatted_data = []
    for item in raw_data:
        # 假设原始数据格式为: {"image_path": "...", "question": "...", "answer": "..."}
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["question"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": item["answer"]
                }
            ]
        }
        formatted_data.append(message)

    return Dataset.from_list(formatted_data)

# 使用示例
train_dataset = convert_to_qwen_format("train_data.json")
```

### 数据增强

```python
from PIL import Image
import torchvision.transforms as transforms

def augment_image(image_path):
    """图像数据增强"""
    image = Image.open(image_path)

    # 定义增强变换
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    return transform(image)

# 在数据加载时应用增强
def preprocess_function(examples):
    images = []
    for img_path in examples["image_path"]:
        augmented_img = augment_image(img_path)
        images.append(augmented_img)

    return {"pixel_values": images}
```

---

## LoRA 配置

### 基础配置

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,  # LoRA 秩
    lora_alpha=32,  # LoRA alpha
    lora_dropout=0.05,  # dropout 率
    bias="none",  # 不训练 bias
    task_type="CAUSAL_LM",  # 任务类型
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention 层
        "gate_proj", "up_proj", "down_proj",     # MLP 层
        "visual.q_proj", "visual.k_proj", "visual.v_proj", "visual.o_proj"  # 视觉模块
    ],
    modules_to_save=["lm_head", "embed_tokens"]  # 保存的模块
)
```

### 视觉模块 LoRA

对于多模态模型，需要特别配置视觉模块：

```python
# 视觉模块特定的 LoRA 配置
visual_lora_config = LoraConfig(
    r=8,  # 视觉模块使用较小的秩
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "visual.blocks.*.attn.qkv",    # 视觉 attention
        "visual.blocks.*.mlp.fc1",     # 视觉 MLP
        "visual.blocks.*.mlp.fc2"
    ]
)
```

### Unsloth 优化配置

```python
from unsloth import FastVisionModel

# Unsloth 自动优化 LoRA 配置
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit=True,  # 4-bit 量化
    use_gradient_checkpointing="unsloth"  # Unsloth 梯度检查点
)

model = FastVisionModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_rslora=True,  # Rank Stabilized LoRA
    loftq_config=None,
)
```

---

## 训练实现

### 使用 Transformers + PEFT

```python
import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, PeftModel
from datasets import load_dataset

def train_qwen3_vl_lora():
    # 加载模型和处理器
    model_id = "Qwen/Qwen3-VL-8B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Flash Attention 2
    )

    processor = AutoProcessor.from_pretrained(model_id)

    # 应用 LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 查看可训练参数

    # 加载数据集
    dataset = load_dataset("json", data_files="qwen_vl_train.json")

    # 数据处理函数
    def collate_fn(batch):
        messages = [item["messages"] for item in batch]
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages
        ]

        # 处理图像
        images = []
        for item in batch:
            for content in item["messages"][0]["content"]:
                if content["type"] == "image":
                    image = processor.image_processor(content["image"])
                    images.append(image)

        inputs = processor(
            text=texts,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )

        # 创建标签 (用于监督学习)
        labels = inputs["input_ids"].clone()
        # 掩码掉指令部分，只计算响应部分的损失
        for i, msg in enumerate(messages):
            instruction_length = len(processor.apply_chat_template(
                [msg[0]], tokenize=True, add_generation_prompt=False
            )["input_ids"])
            labels[i, :instruction_length] = -100

        inputs["labels"] = labels
        return inputs

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen3-vl-lora-output",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        evaluation_strategy="steps",
        eval_steps=500,
        bf16=True,  # bfloat16 精度
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        data_collator=collate_fn,
    )

    # 开始训练
    trainer.train()

    # 保存 LoRA 权重
    trainer.save_model("./qwen3-vl-lora-final")

    return model, processor

# 运行训练
model, processor = train_qwen3_vl_lora()
```

### 使用 Unsloth (推荐)

```python
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments
from trl import SFTTrainer
import torch

def train_with_unsloth():
    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )

    # 获取 PEFT 模型
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
        loftq_config=None,
    )

    # 准备数据
    train_dataset = load_dataset("json", data_files="qwen_vl_train.json")["train"]

    # Unsloth 数据整理器
    data_collator = UnslothVisionDataCollator(
        model, tokenizer, data_collator=None
    )

    # 训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,  # 快速验证用，实际训练增加
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
    )

    # SFT 训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="messages",  # 数据集字段名
        max_seq_length=2048,
        data_collator=data_collator,
        dataset_num_proc=4,  # 数据处理并行数
        packing=False,  # Qwen VL 不支持 packing
        args=training_args,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained("qwen3_vl_lora_model")
    tokenizer.save_pretrained("qwen3_vl_lora_model")

# 运行训练
train_with_unsloth()
```

---

## 推理与部署

### 加载微调模型

```python
from unsloth import FastVisionModel
from transformers import TextStreamer
import torch

def load_finetuned_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "qwen3_vl_lora_model",
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)  # 推理模式优化
    return model, tokenizer

def inference_with_image(image_path, question):
    model, tokenizer = load_finetuned_model()

    # 准备消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }
    ]

    # 应用聊天模板
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    # 处理输入
    inputs = tokenizer(
        image_path,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")

    # 生成响应
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=128,
        use_cache=True,
        temperature=1.5,
        min_p=0.1
    )

# 使用示例
inference_with_image("sample.jpg", "描述这张图片的内容")
```

### 合并 LoRA 权重

```python
from peft import PeftModel

# 加载基础模型
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16
)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(base_model, "qwen3_vl_lora_model")

# 合并权重
merged_model = model.merge_and_unload()

# 保存完整模型
merged_model.save_pretrained("qwen3_vl_merged_model")
```

### 量化部署

```python
from transformers import BitsAndBytesConfig

# 4-bit 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 加载量化模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "qwen3_vl_merged_model",
    quantization_config=quantization_config,
    device_map="auto"
)
```

---

## 最佳实践

### 1. 数据质量

- **多样性**: 包含各种类型的图像和任务
- **质量控制**: 人工审核标注质量
- **平衡性**: 避免类别不平衡

### 2. 训练策略

- **学习率**: 2e-4 ~ 5e-4 (LoRA 可以使用较高学习率)
- **批大小**: 从小批开始，逐渐增大
- **梯度累积**: 补偿小批大小
- **早停**: 监控验证损失

### 3. 硬件优化

- **梯度检查点**: 节省显存
- **混合精度**: FP16/BF16 加速训练
- **数据并行**: 多 GPU 分布式训练

### 4. 超参数调优

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| r | 8-32 | LoRA 秩，越高拟合能力越强 |
| lora_alpha | 16-64 | 通常为 r 的 2-4 倍 |
| dropout | 0-0.1 | 防止过拟合 |
| learning_rate | 1e-4 - 5e-4 | LoRA 可以使用较高值 |

### 5. 调试技巧

```python
# 监控训练过程
from transformers import TrainerCallback

class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"Step {state.global_step}: {logs}")

trainer = Trainer(
    # ... 其他参数
    callbacks=[LoggingCallback()]
)
```

---

## 常见问题

### Q1: CUDA 内存不足

**解决方案**:
```python
# 减小批大小
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)

# 使用梯度检查点
model.gradient_checkpointing_enable()

# 使用 4-bit 量化
model = model.to(dtype=torch.float16)
```

### Q2: 训练损失不下降

**可能原因**:
- 学习率过低
- 数据质量差
- LoRA 配置不当

**解决方案**:
- 增加学习率到 5e-4
- 检查数据质量
- 增加 LoRA 秩到 32

### Q3: 推理时出现错误

**常见错误**: 图像处理失败
```python
# 检查图像路径和格式
from PIL import Image

def validate_image(image_path):
    try:
        img = Image.open(image_path)
        return img.format in ['JPEG', 'PNG', 'BMP']
    except:
        return False
```

### Q4: 如何评估微调效果

```python
from evaluate import load

# 加载评估指标
bleu = load("bleu")
rouge = load("rouge")

def evaluate_model(model, eval_dataset):
    predictions = []
    references = []

    for item in eval_dataset:
        # 生成预测
        pred = generate_response(model, item["input"])
        predictions.append(pred)
        references.append(item["output"])

    # 计算指标
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    return {"bleu": bleu_score, "rouge": rouge_score}
```

---

## 性能对比

| 方法 | 训练参数 | 显存占用 | 相对速度 |
|------|----------|----------|----------|
| 全量微调 | 8B | ~64GB | 1x |
| LoRA (r=16) | ~3.5M | ~16GB | 2-3x |
| QLoRA (4-bit) | ~3.5M | ~12GB | 3-4x |
| Unsloth | ~3.5M | ~10GB | 4-5x |

---

## 扩展阅读

### 官方资源
- [Qwen3 VL 官方文档](https://github.com/QwenLM/Qwen3)
- [Unsloth Qwen3 VL 指南](https://unsloth.ai/docs/models/qwen3-vl-how-to-run-and-fine-tune)

### 相关论文
- [Qwen3 Technical Report](https://arxiv.org/abs/2412.XXX) (即将发布)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### 社区资源
- [Hugging Face Qwen3 VL](https://huggingface.co/Qwen/Qwen3-VL-8B)
- [Qwen VL 微调示例](https://github.com/QwenLM/Qwen-VL)

---

## 总结

Qwen3 VL 8B 的 LoRA 微调提供了一种高效的多模态模型定制方案。通过合理的数据准备、LoRA 配置和训练策略，可以在保持模型通用能力的同时显著提升特定任务的表现。

关键要点：
1. **多模态数据**: 正确格式化图像-文本对
2. **LoRA 配置**: 针对视觉和语言模块优化
3. **Unsloth 加速**: 显著提升训练效率
4. **推理优化**: 量化部署降低资源需求

随着多模态应用的普及，LoRA 微调将成为定制 VLMs 的标准方法。