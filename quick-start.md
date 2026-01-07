# 🚀 快速开始指南

欢迎使用AI笔记系统！这个指南将帮助你快速上手。

## 📖 5分钟快速上手

### 1. 选择合适的模板

根据你的学习内容选择模板：

| 学习类型 | 使用模板 | 位置 |
|----------|----------|------|
| 📄 读论文 | `paper-note.md` | `templates/paper-note.md` |
| 💻 做项目 | `project-log.md` | `templates/project-log.md` |
| 🧐 学概念 | `concept-explained.md` | `templates/concept-explained.md` |
| 🧪 做实验 | `experiment-log.md` | `templates/experiment-log.md` |

### 2. 复制模板并开始记录

```bash
# 示例：记录一篇论文阅读笔记
cp templates/paper-note.md papers/latest/2027-attention.md

# 示例：记录一个项目
cp templates/project-log.md projects/in-progress/my-first-project.md
```

### 3. 按模板结构填写

打开复制的文件，按照模板的引导填写内容。

---

## 🎯 常见使用场景

### 📚 学习AI基础课程

```bash
# 创建概念笔记
cp templates/concept-explained.md docs/fundamentals/linear-algebra.md
cp templates/concept-explained.md docs/fundamentals/probability-theory.md

# 创建算法笔记
cp templates/concept-explained.md docs/machine-learning/gradient-descent.md
cp templates/concept-explained.md docs/deep-learning/backpropagation.md
```

### 📄 阅读经典论文

```bash
# 经典论文
cp templates/paper-note.md papers/classic/2015-resnet.md
cp templates/paper-note.md papers/classic/2017-transformer.md

# 最新论文
cp templates/paper-note.md papers/latest/2027-new-method.md
```

### 💻 实践项目开发

```bash
# 新项目想法
cp templates/project-log.md projects/ideas/image-classification.md

# 进行中的项目
cp templates/project-log.md projects/in-progress/nlp-chatbot.md

# 已完成的项目
cp templates/project-log.md projects/completed/data-visualization.md
```

### 🧪 算法实验记录

```bash
# 模型对比实验
cp templates/experiment-log.md experiments/models/cnn-vs-transformer.md

# 参数调优实验
cp templates/experiment-log.md experiments/models/hyperparameter-tuning.md
```

---

## 📝 命名规范

### 文件命名
- **英文小写 + 连字符**：`transformer-architecture.md`
- **论文笔记**：`[年份]-[简称].md`：`2027-attention.md`
- **项目记录**：`[项目名]-[日期].md`：`chatbot-20270107.md`
- **实验记录**：`exp-[日期]-[序号].md`：`exp-20270107-01.md`

### 目录选择
```bash
# 理论知识 → docs/
docs/fundamentals/          # 数学基础
docs/machine-learning/      # 机器学习
docs/deep-learning/         # 深度学习
docs/nlp/                   # 自然语言处理
docs/cv/                    # 计算机视觉

# 实践内容 → projects/
projects/ideas/             # 项目想法
projects/in-progress/       # 进行中
projects/completed/         # 已完成

# 论文阅读 → papers/
papers/classic/             # 经典论文
papers/latest/              # 最新论文

# 实验记录 → experiments/
experiments/models/          # 模型实验
experiments/datasets/       # 数据集实验
```

---

## 🛠️ 推荐工具配置

### VS Code 插件推荐
```json
{
  "recommendations": [
    "yzhang.markdown-all-in-one",      // Markdown预览
    "shd101wyy.markdown-preview-enhanced", // 增强预览
    "bierner.markdown-mermaid",        // Mermaid图表
    "ms-python.python",                // Python支持
    "ms-toolsai.jupyter"              // Jupyter支持
  ]
}
```

### Git 配置
```bash
# 设置用户信息
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 创建 .gitignore
echo "*.pyc\n__pycache__/\n.env\n.DS_Store" > .gitignore
```

---

## 📊 学习路径建议

### 🔰 初学者路径 (第1-3个月)
1. **数学基础** → `docs/fundamentals/`
   - 线性代数 (2周)
   - 概率统计 (2周)
   - 微积分基础 (1周)

2. **编程基础** → `docs/tools/`
   - Python基础 (1周)
   - NumPy/Pandas (1周)
   - Matplotlib可视化 (1周)

3. **机器学习入门** → `docs/machine-learning/`
   - 监督学习 (2周)
   - 无监督学习 (1周)
   - 模型评估 (1周)

### 🎯 进阶路径 (第4-6个月)
1. **深度学习** → `docs/deep-learning/`
   - 神经网络基础 (2周)
   - CNN计算机视觉 (3周)
   - RNN序列建模 (3周)

2. **专业方向选择**
   - NLP方向：`docs/nlp/`
   - CV方向：`docs/cv/`
   - 强化学习：`docs/reinforcement/`

### 🚀 实战路径 (持续)
1. **小型项目** → `projects/`
   - 每月完成1个项目
   - 记录完整开发过程

2. **论文阅读** → `papers/`
   - 每周精读1篇经典论文
   - 跟踪领域最新进展

---

## 🔄 日常使用流程

### 📅 每日学习记录
```bash
# 1. 创建当日学习笔记
cp templates/concept-explained.md docs/machine-learning/[今日主题].md

# 2. 记录学习内容
# 3. 提交到git
git add .
git commit -m "feat: add notes for [今日主题]"
```

### 📄 每周论文阅读
```bash
# 1. 选择论文模板
cp templates/paper-note.md papers/latest/[年份]-[论文简称].md

# 2. 按模板完成笔记
# 3. 添加到阅读列表
```

### 💻 每月项目实践
```bash
# 1. 创建项目记录
cp templates/project-log.md projects/in-progress/[项目名].md

# 2. 持续更新项目进展
# 3. 完成后移到completed目录
```

---

## ❓ 常见问题

### Q: 如何组织大量笔记？
A: 
- 使用标签系统：`#基础 #进阶 #实战`
- 建立索引文件：在各个目录创建README.md
- 使用软链接：相关笔记可以互相引用

### Q: 如何保证笔记质量？
A:
- 每个笔记都要有个人思考部分
- 定期回顾和更新旧笔记
- 实践验证理论知识点

### Q: 如何与Git集成？
A:
```bash
# 常用Git命令
git add .                    # 添加所有更改
git commit -m "描述"        # 提交
git push                    # 推送到远程
git log --oneline          # 查看提交历史
```

---

## 🎯 下一步行动

1. **立即开始**：选择一个主题，复制对应的模板开始记录
2. **设置环境**：安装推荐的VS Code插件
3. **制定计划**：根据建议的学习路径制定个人计划
4. **养成习惯**：坚持每天记录学习内容

---

需要帮助？查看 `README.md` 获取更多信息，或者在相应目录下查看示例笔记。

📧 有问题或建议？欢迎提交issue或PR！

---

**开始日期**：[今天的日期]  
**目标**：[你的学习目标]  
**计划完成时间**：[目标完成时间]