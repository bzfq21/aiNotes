# Qwen3 VL 计算机使用代理微调

**核心任务**: 从屏幕截图中识别和定位可交互UI元素，支持图片下载等操作。

## 任务描述
- **输入**: 截图 + 指令（如"下载小猫图片"）
- **输出**: 元素定位信息 → 点击坐标
- **关键能力**: 理解图片内容，识别可点击元素，生成操作序列

## 简化训练方案

### 核心思路
**输入图片 + 指令 → 输出操作**，大幅降低标注复杂度。

### 推荐格式：单图+指令（最实用）
```json
{
  "instruction": "在浏览器中搜索'天气预报'",
  "initial_screenshot": "screenshot.png",
  "next_action": {
    "action": "click",
    "x": 300, "y": 80,
    "reasoning": "点击搜索框"
  }
}
```

### 少样本学习（推荐）
- **数据量**: 10-50 个样本
- **标注时间**: 每个样本 1-2 分钟
- **优势**: 快速原型，针对特定任务优化

## 数据准备

### 核心数据格式
```json
{
  "instruction": "下载小猫图片",
  "screenshot": "browser_screenshot.png",
  "action": {
    "type": "click",
    "x": 300, "y": 400,
    "target": "cat_image"
  }
}
```

### 数据收集
- **合成数据**: 生成简单UI界面，标注交互元素
- **真实数据**: 记录用户操作轨迹
- **少样本**: 10-50个精心设计的样本即可开始

## LoRA 配置

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,  # 定位任务使用16-32
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)
```

## 训练脚本

```python
from unsloth import FastVisionModel
from trl import SFTTrainer
from transformers import TrainingArguments

# 加载模型
model, tokenizer = FastVisionModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    load_in_4bit=True
)

# 应用 LoRA
model = FastVisionModel.get_peft_model(model, lora_config)

# 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        output_dir="./output"
    )
)

trainer.train()
```

## 推理使用

```python
from unsloth import FastVisionModel

# 加载模型
model, tokenizer = FastVisionModel.from_pretrained("qwen3_vl_computer_use_lora")

def analyze_screenshot(image_path, instruction):
    """分析截图并生成操作"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction}
            ]
        }
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(image_path, input_text, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 解析操作
    return parse_action_response(response)

# 使用示例
action = analyze_screenshot("browser.png", "下载小猫图片")
print(action)  # {"action": "click", "x": 300, "y": 400}
```

## 评估

```python
def evaluate_actions(predictions, ground_truths):
    """评估操作生成准确性"""
    correct_actions = 0
    total_samples = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        # 比较操作类型和坐标（简化版）
        if (pred.get("action") == gt.get("action") and
            abs(pred.get("x", 0) - gt.get("x", 0)) < 50 and
            abs(pred.get("y", 0) - gt.get("y", 0)) < 50):
            correct_actions += 1

    return correct_actions / total_samples if total_samples > 0 else 0
```

## 最佳实践

- **数据**: 从10-50个少样本开始，逐步增加复杂度
- **标注**: 优先标注核心交互元素（按钮、输入框、图片）
- **训练**: LoRA r=16-32，学习率2e-4，1-3轮训练
- **推理**: 低温度(0.1)，后处理解析操作坐标

## 总结

**核心工作流**:
1. 收集截图+指令数据（少样本即可）
2. LoRA微调Qwen3 VL
3. 推理时输入截图+指令，输出操作坐标
4. 支持图片下载、按钮点击、表单填写等任务

**优势**: 快速上手，效果显著，特别适合图片定位任务。