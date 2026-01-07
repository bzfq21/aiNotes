# Qwen3 VL 计算机使用代理微调：截图交互元素定位

## 概述

本笔记专注于使用 Qwen3 VL 8B 模型微调计算机使用代理 (Computer Use Agent)，具体任务是从屏幕截图中识别、描述和定位所有可交互的UI元素（不仅仅是文本元素）。这对于构建能够理解和操作GUI界面的AI代理至关重要。

### 任务描述

**输入**: 计算机屏幕截图 + 用户指令  
**输出**: JSON格式的UI元素列表，包含：
- 元素文本内容（如果有）
- 元素描述和意图
- 边界框坐标 (bounding box)
- 交互属性：可点击性、可输入性
- 元素类型：button, input, link, select, checkbox, image等

#### 具体示例场景
**场景**: 用户说"下载这张小猫图片"，截图显示浏览器中有一张小猫图片  
**模型需要识别**: 图片元素的位置、大小、内容描述  
**输出用于**: 生成点击坐标来下载图片

### 目标元素类型

| 元素类型 | 描述 | 交互属性 |
|----------|------|----------|
| button | 按钮 | clickable: true |
| input | 输入框 | inputable: true |
| textarea | 多行文本框 | inputable: true |
| select | 下拉选择 | clickable: true, expandable: true |
| link | 超链接 | clickable: true |
| checkbox | 复选框 | clickable: true |
| radio | 单选按钮 | clickable: true |
| menu | 菜单项 | clickable: true |
| icon | 可点击图标 | clickable: true |
| image | 可点击图片 | clickable: true, content_description: "图片内容描述" |

### 应用场景

- **GUI自动化**: 识别和操作所有交互元素
- **Web自动化**: 完整表单填写和导航
- **桌面应用控制**: 全面的界面理解和操作
- **内容识别**: 识别图片、视频等媒体内容并执行相应操作
- **下载管理**: 定位可下载的图片、文件等资源
- **辅助技术**: 为残障用户提供完整的界面访问

---

## 简化训练方案

考虑到完整标注的复杂性，这里提供几种简化方案，可以大幅降低数据准备成本：

### 方案1：最小化属性标注

**只标注核心属性**，减少标注时间：

```json
{
  "elements": [
    {
      "type": "image",
      "bbox": [100, 400, 500, 600],
      "description": "cat picture"
    },
    {
      "type": "button",
      "bbox": [200, 280, 300, 310],
      "text": "Download"
    }
  ]
}
```

**优势**：
- 标注速度提升 3-5 倍
- 模型仍能学习定位和基本分类
- 适合快速原型开发

### 方案2：任务导向的少样本学习

**针对特定任务（如图片下载）进行少样本微调**：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "screenshot1.png"},
          {"type": "text", "text": "Find the cat image and tell me its location"}
        ]
      },
      {
        "role": "assistant",
        "content": "The cat image is located at coordinates [100, 400, 500, 600]"
      }
    ]
  }
]
```

**优势**：
- 数据量减少到 10-50 个样本
- 直接针对具体任务优化
- 利用模型已有能力

### 方案3：两阶段微调

**阶段1：元素定位（粗定位）**
- 只训练模型识别和定位所有可点击元素
- 输出：边界框列表

**阶段2：属性分类（精分类）**
- 在定位基础上，添加元素类型和属性
- 可以用更少的标注数据

### 方案4：弱监督学习

**利用现有工具自动生成标注**：
1. 使用 OCR 工具识别文本元素
2. 使用图像分类模型预标注图片内容
3. 人工审核和修正少量错误

**数据生成脚本**（简化版）：

```python
def generate_simple_training_data():
    """简化版训练数据生成"""
    samples = []

    # 预定义的简单场景
    scenarios = [
        {
            "image": "browser_with_cat.png",
            "instruction": "Locate the cat image",
            "response": "Image located at [200, 300, 600, 500] - small cat with white fur"
        },
        {
            "image": "download_page.png",
            "instruction": "Find the download button",
            "response": "Button at [400, 250, 500, 280] with text 'Download'"
        }
    ]

    for scenario in scenarios:
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": scenario["image"]},
                        {"type": "text", "text": scenario["instruction"]}
                    ]
                },
                {
                    "role": "assistant",
                    "content": scenario["response"]
                }
            ]
        }
        samples.append(sample)

    return samples

# 只需10-20个精心设计的样本
simple_data = generate_simple_training_data()
```

---

## 数据准备

### 数据格式

训练数据采用多模态对话格式：

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": "screenshots/web_form.png"},
          {"type": "text", "text": "Identify and locate all interactive UI elements in this screenshot. Return the result as JSON with detailed properties for each element."}
        ]
      },
      {
        "role": "assistant",
        "content": "{\n  \"elements\": [\n    {\n      \"id\": \"username_input\",\n      \"text\": \"\",\n      \"description\": \"username input field\",\n      \"bbox\": [150, 200, 450, 230],\n      \"clickable\": true,\n      \"inputable\": true,\n      \"type\": \"input\",\n      \"placeholder\": \"Enter username\",\n      \"required\": true\n    },\n    {\n      \"id\": \"login_button\",\n      \"text\": \"Login\",\n      \"description\": \"submit login form\",\n      \"bbox\": [200, 280, 300, 310],\n      \"clickable\": true,\n      \"inputable\": false,\n      \"type\": \"button\",\n      \"intent\": \"authenticate_user\"\n    },\n    {\n      \"id\": \"remember_checkbox\",\n      \"text\": \"Remember me\",\n      \"description\": \"stay logged in option\",\n      \"bbox\": [150, 250, 170, 270],\n      \"clickable\": true,\n      \"inputable\": false,\n      \"type\": \"checkbox\",\n      \"checked\": false\n    },\n    {\n      \"id\": \"forgot_link\",\n      \"text\": \"Forgot password?\",\n      \"description\": \"password recovery link\",\n      \"bbox\": [320, 285, 420, 305],\n      \"clickable\": true,\n      \"inputable\": false,\n      \"type\": \"link\",\n      \"href\": \"#forgot\"\n    },\n    {\n      \"id\": \"cat_image\",\n      \"text\": \"\",\n      \"description\": \"cute kitten image, suitable for download\",\n      \"bbox\": [100, 400, 500, 600],\n      \"clickable\": true,\n      \"inputable\": false,\n      \"type\": \"image\",\n      \"content_description\": \"small cat with white and orange fur\",\n      \"downloadable\": true,\n      \"src\": \"cat.jpg\"\n    }\n  ]\n}"
      }
    ]
  }
]
```

### 数据收集策略

#### 1. 合成数据生成

```python
import json
from PIL import Image, ImageDraw, ImageFont
import random

def generate_synthetic_ui_screenshot():
    """生成包含多种UI元素的合成截图数据"""
    # 创建空白图像
    img = Image.new('RGB', (1920, 1080), color='lightgray')
    draw = ImageDraw.Draw(img)

    # 定义字体
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    elements = []

    # 生成登录表单
    form_x, form_y = 400, 200
    form_width, form_height = 400, 300

    # 表单背景
    draw.rectangle([form_x, form_y, form_x+form_width, form_y+form_height],
                  fill='white', outline='black', width=2)

    # 标题
    title_y = form_y + 20
    draw.text((form_x + 150, title_y), "Login Form", fill='black', font=font)
    elements.append({
        "id": "form_title",
        "text": "Login Form",
        "description": "form title",
        "bbox": [form_x + 150, title_y, form_x + 250, title_y + 20],
        "clickable": False,
        "inputable": False,
        "type": "label"
    })

    # 用户名输入框
    input_y = title_y + 50
    draw.rectangle([form_x + 50, input_y, form_x + 350, input_y + 30],
                  fill='white', outline='gray')
    draw.text((form_x + 55, input_y + 5), "Enter username", fill='gray', font=small_font)
    elements.append({
        "id": "username_input",
        "text": "",
        "description": "username input field",
        "bbox": [form_x + 50, input_y, form_x + 350, input_y + 30],
        "clickable": True,
        "inputable": True,
        "type": "input",
        "placeholder": "Enter username",
        "required": True
    })

    # 密码输入框
    pwd_y = input_y + 50
    draw.rectangle([form_x + 50, pwd_y, form_x + 350, pwd_y + 30],
                  fill='white', outline='gray')
    draw.text((form_x + 55, pwd_y + 5), "••••••••", fill='black', font=small_font)
    elements.append({
        "id": "password_input",
        "text": "",
        "description": "password input field",
        "bbox": [form_x + 50, pwd_y, form_x + 350, pwd_y + 30],
        "clickable": True,
        "inputable": True,
        "type": "input",
        "input_type": "password",
        "required": True
    })

    # 记住我复选框
    checkbox_y = pwd_y + 40
    draw.rectangle([form_x + 50, checkbox_y, form_x + 65, checkbox_y + 15],
                  fill='white', outline='black')
    draw.text((form_x + 75, checkbox_y), "Remember me", fill='black', font=small_font)
    elements.append({
        "id": "remember_checkbox",
        "text": "Remember me",
        "description": "remember login option",
        "bbox": [form_x + 50, checkbox_y, form_x + 160, checkbox_y + 15],
        "clickable": True,
        "inputable": False,
        "type": "checkbox",
        "checked": False
    })

    # 登录按钮
    btn_y = checkbox_y + 30
    draw.rectangle([form_x + 150, btn_y, form_x + 250, btn_y + 35],
                  fill='blue', outline='darkblue')
    draw.text((form_x + 190, btn_y + 8), "Login", fill='white', font=font)
    elements.append({
        "id": "login_button",
        "text": "Login",
        "description": "submit login credentials",
        "bbox": [form_x + 150, btn_y, form_x + 250, btn_y + 35],
        "clickable": True,
        "inputable": False,
        "type": "button",
        "intent": "authenticate_user"
    })

    # 忘记密码链接
    link_y = btn_y + 50
    draw.text((form_x + 160, link_y), "Forgot password?", fill='blue', font=small_font)
    elements.append({
        "id": "forgot_link",
        "text": "Forgot password?",
        "description": "password recovery link",
        "bbox": [form_x + 160, link_y, form_x + 280, link_y + 15],
        "clickable": True,
        "inputable": False,
        "type": "link",
        "href": "#forgot"
    })

    # 添加图片元素
    img_y = form_y + form_height + 50
    draw.rectangle([form_x + 100, img_y, form_x + 300, img_y + 200],
                  fill='lightgray', outline='black', width=2)
    # 模拟图片内容描述
    draw.text((form_x + 120, img_y + 80), "[Cat Image]", fill='black', font=font)
    elements.append({
        "id": "sample_image",
        "text": "",
        "description": "sample image showing a small cat",
        "bbox": [form_x + 100, img_y, form_x + 300, img_y + 200],
        "clickable": True,
        "inputable": False,
        "type": "image",
        "content_description": "cute small cat with fluffy fur",
        "downloadable": True,
        "alt": "Small cat image"
    })

    # 随机添加其他元素
    for i in range(random.randint(2, 5)):
        elem_types = ["button", "link", "icon", "image"]
        elem_type = random.choice(elem_types)

        x1 = random.randint(50, 1800)
        y1 = random.randint(50, 1000)
        width = random.randint(60, 200)
        height = random.randint(25, 50)

        if elem_type == "button":
            draw.rectangle([x1, y1, x1+width, y1+height], fill='lightblue', outline='blue')
            text = random.choice(["Save", "Cancel", "Delete", "Edit"])
            draw.text((x1+10, y1+10), text, fill='black', font=small_font)
            elements.append({
                "id": f"random_button_{i}",
                "text": text,
                "description": f"{text.lower()} action button",
                "bbox": [x1, y1, x1+width, y1+height],
                "clickable": True,
                "inputable": False,
                "type": "button"
            })
        elif elem_type == "link":
            text = random.choice(["Help", "About", "Contact", "Privacy"])
            draw.text((x1, y1), text, fill='blue', font=small_font)
            elements.append({
                "id": f"random_link_{i}",
                "text": text,
                "description": f"{text.lower()} page link",
                "bbox": [x1, y1, x1+len(text)*8, y1+15],
                "clickable": True,
                "inputable": False,
                "type": "link"
            })
        elif elem_type == "icon":
            draw.rectangle([x1, y1, x1+height, y1+height], fill='gray', outline='black')
            elements.append({
                "id": f"random_icon_{i}",
                "text": "",
                "description": "clickable icon",
                "bbox": [x1, y1, x1+height, y1+height],
                "clickable": True,
                "inputable": False,
                "type": "icon"
            })
        elif elem_type == "image":
            draw.rectangle([x1, y1, x1+width, y1+height], fill='lightblue', outline='black')
            img_content = random.choice(["[Photo]", "[Image]", "[Picture]"])
            draw.text((x1+10, y1+height//2-10), img_content, fill='black', font=small_font)
            elements.append({
                "id": f"random_image_{i}",
                "text": "",
                "description": f"image element {img_content.lower()}",
                "bbox": [x1, y1, x1+width, y1+height],
                "clickable": True,
                "inputable": False,
                "type": "image",
                "content_description": f"image showing {random.choice(['nature', 'animal', 'object', 'person'])}",
                "downloadable": random.random() > 0.3
            })

    return img, elements
```

#### 2. 真实数据标注

使用工具如 LabelImg 或自定义脚本来标注真实截图：

```python
import cv2
import json

def annotate_screenshot(image_path):
    """手动标注截图"""
    # 使用 OpenCV 或其他工具进行标注
    # 返回标注结果
    pass

# 批量标注
real_screenshots = ["screenshot1.png", "screenshot2.png"]
annotated_data = []

for screenshot in real_screenshots:
    annotation = annotate_screenshot(screenshot)
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": screenshot},
                    {"type": "text", "text": "Identify and locate all text elements in this screenshot. Return the result as JSON."}
                ]
            },
            {
                "role": "assistant",
                "content": json.dumps(annotation, indent=2)
            }
        ]
    }
    annotated_data.append(sample)
```

### 数据增强

```python
from torchvision import transforms

def augment_screenshot(image, elements):
    """数据增强"""
    # 定义增强变换
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])

    augmented_img = transform(image)

    # 相应调整边界框坐标
    # 注意：这里需要实现坐标变换逻辑
    augmented_elements = adjust_bboxes(elements, transform)

    return augmented_img, augmented_elements

def adjust_bboxes(elements, transform):
    """根据变换调整边界框"""
    # 实现坐标变换逻辑
    return elements
```

---

## LoRA 配置优化

### 定位任务专用配置

```python
from peft import LoraConfig

# 针对GUI定位优化的LoRA配置
localization_lora_config = LoraConfig(
    r=32,  # 更高秩以捕捉空间信息
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # 视觉模块 - 重点关注
        "visual.blocks.*.attn.qkv",    # 视觉注意力
        "visual.blocks.*.attn.proj",
        "visual.blocks.*.mlp.fc1",
        "visual.blocks.*.mlp.fc2",

        # 语言模块 - 输出结构化文本
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",

        # 输出层 - 学习JSON格式
        "lm_head"
    ],
    modules_to_save=["embed_tokens"]  # 保存位置嵌入
)
```

### Unsloth 优化配置

```python
from unsloth import FastVisionModel

# Unsloth 配置，针对定位任务
model = FastVisionModel.get_peft_model(
    model,
    r=32,  # 增加秩
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_rslora=True,  # Rank Stabilized LoRA
    loftq_config=None,
    # 针对视觉任务的额外配置
    visual_target_modules=[
        "visual.blocks.*.attn.qkv",
        "visual.blocks.*.mlp.fc1"
    ]
)
```

---

## 训练实现

### 数据处理函数

```python
from datasets import Dataset
import json

def preprocess_computer_use_data(examples):
    """处理计算机使用数据"""
    processed_data = []

    for example in examples:
        messages = example["messages"]
        user_content = messages[0]["content"]
        assistant_content = messages[1]["content"]

        # 提取图像路径
        image_path = None
        text_prompt = ""
        for content_item in user_content:
            if content_item["type"] == "image":
                image_path = content_item["image"]
            elif content_item["type"] == "text":
                text_prompt = content_item["text"]

        # 验证JSON格式响应
        try:
            json.loads(assistant_content)
            is_valid = True
        except:
            is_valid = False

        if image_path and is_valid:
            processed_data.append({
                "image_path": image_path,
                "prompt": text_prompt,
                "response": assistant_content
            })

    return processed_data

# 加载和处理数据集
dataset = Dataset.from_json("computer_use_train.json")
processed_dataset = dataset.map(preprocess_computer_use_data, batched=True)
```

### 训练脚本

```python
import torch
from unsloth import FastVisionModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

def train_computer_use_agent():
    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )

    # 应用 LoRA
    model = FastVisionModel.get_peft_model(
        model,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_rslora=True,
    )

    # 加载数据集
    train_dataset = load_dataset(
        "json",
        data_files="computer_use_train.json",
        split="train"
    )

    # 数据整理器
    class ComputerUseDataCollator:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def __call__(self, batch):
            # 处理图像和文本
            processed_batch = []
            for item in batch:
                # 这里实现具体的处理逻辑
                # 包括图像处理、文本tokenization等
                pass
            return processed_batch

    data_collator = ComputerUseDataCollator(model, tokenizer)

    # 训练参数
    training_args = TrainingArguments(
        per_device_train_batch_size=1,  # 小批次以适应图像
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=1000,  # 根据数据集大小调整
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="qwen3_vl_computer_use_output",
        report_to="none",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="messages",
        max_seq_length=2048,
        data_collator=data_collator,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    # 开始训练
    trainer.train()

    # 保存模型
    model.save_pretrained("qwen3_vl_computer_use_lora")
    tokenizer.save_pretrained("qwen3_vl_computer_use_lora")

    return model, tokenizer

# 运行训练
model, tokenizer = train_computer_use_agent()
```

### 自定义损失函数

为了更好地学习JSON格式输出，可以使用自定义损失：

```python
import torch.nn as nn
from transformers import Trainer

class ComputerUseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits

        # 自定义损失：鼓励JSON格式输出
        # 这里可以实现更复杂的逻辑，如解析JSON并计算定位准确性

        # 基础语言建模损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            inputs["labels"][:, 1:].contiguous().view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# 使用自定义训练器
trainer = ComputerUseTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)
```

---

## 推理与部署

### 加载微调模型

```python
from unsloth import FastVisionModel
import json

def load_computer_use_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "qwen3_vl_computer_use_lora",
        load_in_4bit=True,
    )
    FastVisionModel.for_inference(model)
    return model, tokenizer

def localize_ui_elements(image_path, model, tokenizer):
    """定位截图中的所有UI交互元素"""

    # 准备输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Identify and locate all interactive UI elements in this screenshot including buttons, inputs, links, checkboxes, images, and other clickable elements. For images, provide content descriptions. For each element, provide: id, text content, description, bbox coordinates [x1,y1,x2,y2], clickable status, inputable status, element type, and relevant properties like content_description for images. Return the result as JSON with an 'elements' array."}
            ]
        }
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
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            use_cache=True,
            temperature=0.1,  # 低温度以获得确定性输出
            do_sample=False,
        )

    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("Assistant:")[-1].strip()

    # 解析JSON
    try:
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试提取
        print(f"JSON parsing failed, raw response: {response}")
        return {"elements": []}

# 使用示例
model, tokenizer = load_computer_use_model()
result = localize_ui_elements("test_screenshot.png", model, tokenizer)
print(json.dumps(result, indent=2))

# 图像定位特定示例
def find_downloadable_image(image_path, description_keywords, model, tokenizer):
    """查找符合描述的可下载图片"""
    elements = localize_ui_elements(image_path, model, tokenizer)

    for element in elements.get("elements", []):
        if element.get("type") == "image" and element.get("downloadable", False):
            content_desc = element.get("content_description", "").lower()
            element_desc = element.get("description", "").lower()

            # 检查是否包含关键词
            if any(keyword.lower() in content_desc or keyword.lower() in element_desc
                   for keyword in description_keywords):
                return element

    return None

# 示例：查找小猫图片
cat_image = find_downloadable_image("browser_screenshot.png", ["cat", "kitten"], model, tokenizer)
if cat_image:
    print(f"Found cat image at: {cat_image['bbox']}")
    # 可以在这里生成点击坐标
    center_x = (cat_image['bbox'][0] + cat_image['bbox'][2]) // 2
    center_y = (cat_image['bbox'][1] + cat_image['bbox'][3]) // 2
    print(f"Click coordinates: ({center_x}, {center_y})")
else:
    print("No downloadable cat image found")
```

### 批量推理

```python
import os
from tqdm import tqdm

def batch_localize_screenshots(image_dir, output_file):
    """批量处理截图"""

    model, tokenizer = load_computer_use_model()
    results = {}

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        try:
            result = localize_ui_elements(image_path, model, tokenizer)
            results[image_file] = result
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            results[image_file] = {"elements": []}

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

# 批量处理
batch_localize_screenshots("test_screenshots/", "localization_results.json")
```

### 集成到代理系统

```python
class ComputerUseAgent:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def analyze_screen(self, screenshot_path):
        """分析屏幕截图"""
        elements = localize_text_elements(screenshot_path, self.model, self.tokenizer)
        return elements

    def find_interactive_element(self, screenshot_path, criteria):
        """查找符合条件的交互元素"""
        elements = self.analyze_screen(screenshot_path)

        for element in elements.get("elements", []):
            matches = True

            # 检查类型
            if "type" in criteria and element.get("type") != criteria["type"]:
                matches = False

            # 检查文本内容
            if "text" in criteria and criteria["text"].lower() not in element.get("text", "").lower():
                matches = False

            # 检查可点击性
            if "clickable" in criteria and element.get("clickable") != criteria["clickable"]:
                matches = False

            # 检查可输入性
            if "inputable" in criteria and element.get("inputable") != criteria["inputable"]:
                matches = False

            # 检查意图
            if "intent" in criteria and criteria["intent"].lower() not in element.get("description", "").lower():
                matches = False

            if matches:
                return element

        return None

    def generate_action(self, screenshot_path, goal):
        """根据目标生成动作"""
        elements = self.analyze_screen(screenshot_path)

        # 解析用户目标
        goal_lower = goal.lower()

        if "download" in goal_lower and ("image" in goal_lower or "picture" in goal_lower):
            # 下载图片任务
            keywords = []
            if "cat" in goal_lower or "kitten" in goal_lower:
                keywords = ["cat", "kitten"]
            elif "dog" in goal_lower:
                keywords = ["dog", "puppy"]
            # 添加更多关键词解析逻辑

            image_element = self.find_image_by_description(screenshot_path, keywords)
            if image_element:
                center_x = (image_element['bbox'][0] + image_element['bbox'][2]) // 2
                center_y = (image_element['bbox'][1] + image_element['bbox'][3]) // 2
                return {
                    "action": "click",
                    "element": image_element,
                    "coordinates": (center_x, center_y),
                    "reason": f"Clicking on {image_element.get('description', 'image')} to download"
                }

        # 其他动作类型...

        return {"action": "unknown", "goal": goal}

    def find_image_by_description(self, screenshot_path, keywords):
        """根据描述关键词查找图片"""
        elements = self.analyze_screen(screenshot_path)

        for element in elements.get("elements", []):
            if element.get("type") == "image" and element.get("downloadable", False):
                content_desc = element.get("content_description", "").lower()
                element_desc = element.get("description", "").lower()

                if any(keyword.lower() in content_desc or keyword.lower() in element_desc
                       for keyword in keywords):
                    return element

        return None

# 使用代理
agent = ComputerUseAgent(model, tokenizer)
action = agent.generate_action("current_screen.png", "login to the system")
print(action)
```

---

## 评估方法

### 定位准确性评估

```python
import json
from shapely.geometry import Polygon

def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
    poly1 = Polygon([(box1[0], box1[1]), (box1[2], box1[1]),
                     (box1[2], box1[3]), (box1[0], box1[3])])
    poly2 = Polygon([(box2[0], box2[1]), (box2[2], box2[1]),
                     (box2[2], box2[3]), (box2[0], box2[3])])

    if not poly1.intersects(poly2):
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union if union > 0 else 0.0

def evaluate_localization(predictions, ground_truths, iou_threshold=0.5):
    """评估定位性能"""

    total_predictions = 0
    total_ground_truths = 0
    true_positives = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_elements = pred.get("elements", [])
        gt_elements = gt.get("elements", [])

        total_predictions += len(pred_elements)
        total_ground_truths += len(gt_elements)

        # 为每个ground truth找到最佳匹配
        for gt_elem in gt_elements:
            best_iou = 0
            best_match = None

            for pred_elem in pred_elements:
                if pred_elem["text"] == gt_elem["text"]:
                    iou = calculate_iou(pred_elem["bbox"], gt_elem["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_elem

            if best_iou >= iou_threshold:
                true_positives += 1

    # 计算指标
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_predictions": total_predictions,
        "total_ground_truths": total_ground_truths,
        "true_positives": true_positives
    }

# 评估示例
with open("predictions.json", "r") as f:
    predictions = json.load(f)

with open("ground_truths.json", "r") as f:
    ground_truths = json.load(f)

metrics = evaluate_localization(predictions.values(), ground_truths.values())
print(f"Localization F1 Score: {metrics['f1_score']:.3f}")
```

### 元素类型分类准确性

```python
def evaluate_element_classification(predictions, ground_truths):
    """评估元素类型分类准确性"""
    correct_types = 0
    total_elements = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_elements = pred.get("elements", [])
        gt_elements = gt.get("elements", [])

        # 基于bbox匹配元素
        for gt_elem in gt_elements:
            best_match = None
            best_iou = 0

            for pred_elem in pred_elements:
                iou = calculate_iou(pred_elem["bbox"], gt_elem["bbox"])
                if iou > best_iou and iou > 0.5:  # IoU阈值
                    best_iou = iou
                    best_match = pred_elem

            if best_match and best_match["type"] == gt_elem["type"]:
                correct_types += 1

        total_elements += len(gt_elements)

    accuracy = correct_types / total_elements if total_elements > 0 else 0
    return accuracy

type_accuracy = evaluate_element_classification(predictions.values(), ground_truths.values())
print(f"Element Type Classification Accuracy: {type_accuracy:.3f}")
```

### 图像内容描述准确性

```python
def evaluate_image_description(predictions, ground_truths):
    """评估图像内容描述准确性"""
    correct_descriptions = 0
    total_images = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_images = [elem for elem in pred.get("elements", []) if elem.get("type") == "image"]
        gt_images = [elem for elem in gt.get("elements", []) if elem.get("type") == "image"]

        for gt_img in gt_images:
            best_match = None
            best_iou = 0

            for pred_img in pred_images:
                iou = calculate_iou(pred_img["bbox"], gt_img["bbox"])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_match = pred_img

            if best_match:
                # 检查内容描述相似性（简化版）
                pred_desc = best_match.get("content_description", "").lower()
                gt_desc = gt_img.get("content_description", "").lower()

                # 简单关键词匹配 - 可以用更复杂的相似度计算
                pred_keywords = set(pred_desc.split())
                gt_keywords = set(gt_desc.split())

                if len(pred_keywords & gt_keywords) > 0:
                    correct_descriptions += 1

        total_images += len(gt_images)

    accuracy = correct_descriptions / total_images if total_images > 0 else 0
    return accuracy

image_desc_accuracy = evaluate_image_description(predictions.values(), ground_truths.values())
print(f"Image Description Accuracy: {image_desc_accuracy:.3f}")
```

### 交互属性识别准确性

```python
def evaluate_interaction_properties(predictions, ground_truths):
    """评估交互属性识别准确性"""
    correct_clickable = 0
    correct_inputable = 0
    total_elements = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_elements = pred.get("elements", [])
        gt_elements = gt.get("elements", [])

        for gt_elem in gt_elements:
            best_match = None
            best_iou = 0

            for pred_elem in pred_elements:
                iou = calculate_iou(pred_elem["bbox"], gt_elem["bbox"])
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_match = pred_elem

            if best_match:
                if best_match.get("clickable") == gt_elem.get("clickable"):
                    correct_clickable += 1
                if best_match.get("inputable") == gt_elem.get("inputable"):
                    correct_inputable += 1

        total_elements += len(gt_elements)

    clickable_acc = correct_clickable / total_elements if total_elements > 0 else 0
    inputable_acc = correct_inputable / total_elements if total_elements > 0 else 0

    return {"clickable_accuracy": clickable_acc, "inputable_accuracy": inputable_acc}

interaction_acc = evaluate_interaction_properties(predictions.values(), ground_truths.values())
print(f"Clickable Detection Accuracy: {interaction_acc['clickable_accuracy']:.3f}")
print(f"Inputable Detection Accuracy: {interaction_acc['inputable_accuracy']:.3f}")
```

---

## 最佳实践

### 1. 数据质量控制

#### 完整标注方案（推荐用于生产环境）
- **元素多样性**: 包含所有交互元素类型（按钮、输入框、链接、复选框等）
- **属性完整性**: 为每个元素标注所有相关属性（可点击性、可输入性、类型等）
- **意图标注**: 为元素添加语义意图描述（"登录"、"搜索"、"提交表单"、"下载图片"等）
- **图像内容标注**: 为图片元素提供详细的内容描述（"小猫图片"、"风景照片"等）
- **可下载性标注**: 标记哪些图片/文件是可下载的
- **界面多样性**: 包含不同类型的应用界面（浏览器、桌面软件、移动端等）

#### 简化标注方案（推荐用于快速原型）
- **核心属性优先**: 只标注 type、bbox、基本描述
- **任务导向**: 只标注与目标任务相关的元素
- **批量标注**: 使用半自动工具辅助标注
- **质量 vs 数量**: 选择高质量样本而非大量低质数据
- **迭代优化**: 从简单场景开始，逐步增加复杂度

**标注时间对比**：
- 完整标注：每个元素 2-3 分钟
- 简化标注：每个元素 30 秒
- 效率提升：5-10 倍

### 2. 模型调优

- **LoRA参数**: 定位任务使用更高秩 (r=32)
- **视觉模块**: 重点微调视觉注意力层
- **输出格式**: 使用少量示例强化JSON格式学习

### 3. 训练策略

- **学习率**: 2e-4 ~ 5e-4，视觉任务可以使用稍高学习率
- **批大小**: 从1开始，根据显存调整
- **数据增强**: 轻度增强以保持定位准确性

### 4. 推理优化

- **温度设置**: 定位任务使用低温度 (0.1) 获得确定性输出
- **后处理**: 实现JSON解析容错和边界框验证
- **缓存**: 对相似界面进行结果缓存

---

## 常见问题

### Q1: 模型无法生成有效JSON

**解决方案**:
```python
# 在提示中明确指定格式
prompt = """Identify and locate all text elements in this screenshot.
Return ONLY a valid JSON object with this exact structure:
{
  "elements": [
    {
      "text": "element text",
      "description": "brief description",
      "bbox": [x1, y1, x2, y2],
      "clickable": true/false,
      "type": "button/label/input/link"
    }
  ]
}"""

# 使用JSON模式（如果支持）
# 或者后处理清理响应
def clean_json_response(response):
    # 移除多余文本，只保留JSON部分
    start = response.find('{')
    end = response.rfind('}') + 1
    if start != -1 and end != -1:
        return response[start:end]
    return response
```

### Q2: 定位准确性不高

**可能原因**:
- 训练数据中元素类型不平衡
- 复杂界面布局导致混淆
- 视觉特征相似度高（如多个相似按钮）
- 边界框标注不准确

**解决方案**:
- 平衡各类元素的训练样本
- 增加元素语义描述的多样性
- 使用更高分辨率图像训练
- 实施更严格的质量控制流程
- 添加空间位置编码

### Q3: 推理速度慢

**优化方法**:
```python
# 使用Flash Attention
model = model.to(dtype=torch.float16)

# 量化部署
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = model.from_pretrained(model_path, quantization_config=quant_config)

# 批量推理
# 实现批处理逻辑
```

### Q4: 训练样本太复杂，如何简化？

**简化策略**：

1. **减少属性数量**：
   ```json
   // 复杂版
   {"type": "image", "bbox": [100,200,300,400], "description": "cute cat", "downloadable": true, "content": "small white cat"}

   // 简化版
   {"type": "image", "bbox": [100,200,300,400], "description": "cat"}
   ```

2. **使用自然语言响应**：
   ```json
   // 结构化JSON（复杂）
   {"elements": [{"type": "button", "bbox": [100,200,150,230], "text": "Download"}]}

   // 自然语言（简单）
   "Click the Download button at coordinates [100,200,150,230]"
   ```

3. **少样本学习**：用 10-50 个高质量样本代替 1000+ 个低质样本

4. **渐进式复杂化**：从简单任务开始，逐步增加属性

### Q5: 如何处理不同分辨率的截图

```python
def normalize_bbox(bbox, image_size, target_size=(1920, 1080)):
    """标准化边界框坐标"""
    img_w, img_h = image_size
    target_w, target_h = target_size

    x1, y1, x2, y2 = bbox

    # 缩放到目标尺寸
    x1_norm = int(x1 * target_w / img_w)
    y1_norm = int(y1 * target_h / img_h)
    x2_norm = int(x2 * target_w / img_w)
    y2_norm = int(y2 * target_h / img_h)

    return [x1_norm, y1_norm, x2_norm, y2_norm]
```

---

## 扩展阅读

### 相关工作
- [GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents](https://arxiv.org/abs/2506.03143)
- [R-VLM: Region-Aware Vision Language Model for Precise GUI Grounding](https://arxiv.org/abs/2507.05673)
- [ScreenSpot: A Dataset for Video GUI Understanding](https://arxiv.org/abs/2402.15935)

### 数据集
- [ScreenSpot Dataset](https://github.com/cooelf/ScreenSpot)
- [GUI-Actor Dataset](https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2.5-VL)

### 工具
- [LabelImg](https://github.com/HumanSignal/labelImg) - 图像标注工具
- [Label Studio](https://labelstud.io/) - 数据标注平台

---

## 总结

通过LoRA微调Qwen3 VL 8B，我们可以构建强大的计算机使用代理，能够准确识别和定位屏幕截图中的所有交互式UI元素。

关键要点：
1. **灵活的复杂程度**: 可根据需求选择完整标注或简化方案
2. **全面元素识别**: 不仅识别文本，还识别所有可交互元素及其属性，包括图片内容
3. **图像理解能力**: 为图片元素提供内容描述，支持基于描述的定位和操作
4. **结构化输出**: 生成包含丰富元数据的JSON格式结果
5. **视觉-语言对齐**: 重点微调视觉模块以捕捉界面布局和元素特征
6. **语义理解**: 为元素添加意图描述，实现智能交互决策
7. **下载任务支持**: 特别优化图片定位和下载操作

**复杂度选择指南**：
- **快速原型**: 使用简化方案，10-50 个样本，1-2 天完成
- **生产应用**: 使用完整方案，500+ 个样本，1-2 周完成
- **研究项目**: 结合多种方案，1000+ 个样本，持续迭代

这种方法为构建完整的GUI自动化和辅助技术系统提供了基础，可以支持复杂的表单填写、导航操作、内容下载等任务。