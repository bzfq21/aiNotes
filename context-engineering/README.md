# Context Engineering

系统性阐述上下文工程——优化 AI 系统输入以获得价值输出

## 概述

上下文工程是设计、构建和优化 AI 系统（特别是大语言模型）输入上下文的实践，以实现可预测、高质量的输出。这是有效提示词设计、信息组织和系统编排的艺术与科学。

## 目录

- [概述](#概述)
- [为什么上下文工程很重要](#为什么上下文工程很重要)
- [核心概念](#核心概念)
- [方法论](#方法论)
- [实现模式](#实现模式)
- [评估指标](#评估指标)
- [最佳实践](#最佳实践)
- [工具与框架](#工具与框架)
- [高级主题](#高级主题)
  - [Skills（应用层能力单元）](#1-skills-应用层能力单元）
  - [动态上下文选择](#2-动态上下文选择）
  - [多智能体上下文系统](#3-多智能体上下文系统）
  - [上下文缓存策略](#4-上下文缓存策略）
  - [流式上下文](#5-流式上下文）
- [标准化与协议](#标准化与协议)
  - [规范语言（Spec）](#规范语言（spec））
    - [单智能体上下文规范](#1-single-agent-context-specification）
    - [多智能体边界规范](#2-multi-agent-boundary-specification）
    - [工具调用规范](#3-tool-invocation-specification）
    - [提示词工程规范](#4-prompt-engineering-specification）
  - [模型上下文协议（MCP）](#模型上下文协议（mcp））
- [常见陷阱](#常见陷阱)
- [案例研究](#案例研究)
- [资源](#资源)
- [贡献](#贡献)
- [许可证](#许可证)

## 为什么上下文工程很重要

### 上下文窗口问题
- **容量有限**：所有 AI 模型的上下文窗口（可处理的 token 数量）都是有限的
- **信息过载**：过多上下文可能导致注意力分散和质量下降
- **检索相关性**：只应包含最相关的信息
- **结构很重要**：信息的组织方式影响模型理解

### 业务影响
- **高质量输出**：更好的上下文 → 更好的答案、更准确的响应
- **成本效率**：优化的上下文使用减少 token 消耗
- **可靠性**：一致、可预测的 AI 行为
- **可扩展性**：能够有效处理复杂、多步骤任务的系统

## 核心概念

### 1. 上下文架构

```
用户查询
    ↓
上下文组装
    ↓
提示词构建
    ↓
模型推理
    ↓
响应处理
```

### 2. 上下文类型

| 类型 | 目的 | 示例 |
|------|------|------|
| **系统上下文** | 定义行为、人格、约束 | 角色定义、输出格式、安全准则 |
| **领域上下文** | 领域特定知识 | API 文档、代码库、行业标准 |
| **任务上下文** | 特定任务的指令 | 分步指南、示例、模板 |
| **对话上下文** | 对话历史 | 之前的消息、用户偏好、会话状态 |
| **工具上下文** | 可用工具和功能 | 函数定义、工具使用模式 |

### 3. 上下文窗口层级

1. **关键信息**（必须包含）
   - 用户的明确指令
   - 必要的领域知识
   - 必需的输出格式
   - 安全约束

2. **重要信息**（空间允许时包含）
   - 相关示例
   - 历史上下文
   - 支持性文档

3. **锦上添花**（最后包含或省略）
   - 补充细节
   - 扩展解释
   - 冗余信息

## 方法论

### 1. 检索增强生成（RAG）

**适用场景**：大型知识库、频繁变化的信息

```
查询 → 语义搜索 → Top-k 文档 → 上下文组装 → LLM → 响应
```

**最佳实践**：
- 使用语义嵌入，而非关键词搜索
- 按相关性重新排序检索的文档
- 在上下文中包含来源引用
- 实现上下文窗口管理（滑动窗口、选择性包含）

### 2. 思维链（Chain-of-Thought，CoT）

**适用场景**：复杂推理任务、多步骤问题

```
任务 → 逐步推理 → 中间步骤 → 最终答案
```

**技巧**：
- 显式推理提示词："让我们一步步思考"
- 零样本 CoT："回答前仔细思考"
- 少样本 CoT：提供推理示例
- 自我一致性：生成多个推理路径并聚合

### 3. 思维树（Tree-of-Thought，ToT）

**适用场景**：探索任务、创造性问题解决、规划

```
任务 → 生成多种方法 → 探索每种方法 → 选择最佳路径
```

**应用**：
- 代码生成策略
- 研究规划
- 游戏策略
- 创意写作

### 4. 上下文分块策略

| 策略 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| **固定大小分块** | 简单文档 | 易于实现 | 可能破坏语义单元 |
| **语义分块** | 自然语言文本 | 保留含义 | 需要预处理 |
| **递归分块** | 层次化文档 | 保持结构 | 实现复杂 |
| **基于文档的分块** | 多文档上下文 | 自然边界 | 可能丢失跨文档链接 |
| **混合方法** | 复杂系统 | 多种方法结合 | 复杂度更高 |

## 实现模式

### 1. "上下文构建器"模式

```typescript
class ContextBuilder {
  private context: string[] = [];

  addSystemPrompt(prompt: string): this {
    this.context.push(prompt);
    return this;
  }

  addKnowledge(docs: Document[], maxTokens: number): this {
    const relevant = this.rankByRelevance(docs);
    this.context.push(...this.takeTokens(relevant, maxTokens));
    return this;
  }

  addExamples(examples: Example[]): this {
    this.context.push(this.formatExamples(examples));
    return this;
  }

  build(): string {
    return this.context.join('\n\n');
  }
}
```

### 2. "上下文窗口管理器"模式

```typescript
class ContextWindowManager {
  constructor(private maxTokens: number) {}

  prioritize(
    contexts: ContextItem[],
    query: string
  ): ContextItem[] {
    const scored = contexts.map(ctx => ({
      ...ctx,
      score: this.calculateRelevance(ctx, query)
    }));

    return scored
      .sort((a, b) => b.score - a.score)
      .reduce((acc, ctx) => {
        if (acc.tokens + ctx.tokens <= this.maxTokens) {
          acc.items.push(ctx);
          acc.tokens += ctx.tokens;
        }
        return acc;
      }, { items: [], tokens: 0 })
      .items;
  }
}
```

### 3. "上下文压缩"模式

```typescript
class ContextCompressor {
  compress(
    contexts: string[],
    targetSize: number
  ): string[] {
    // Remove redundancy
    const deduped = this.removeDuplicates(contexts);

    // Extract key information
    const summarized = deduped.map(c => this.summarize(c));

    // Prioritize by information density
    return this.selectByDensity(summarized, targetSize);
  }
}
```

## 评估指标

### 上下文质量指标

| 指标 | 定义 | 如何衡量 |
|------|------|----------|
| **相关性** | 上下文与查询的关联度 | 嵌入相似度、LLM 评分 |
| **覆盖率** | 信息完整性 | 与基准真值比较 |
| **简洁性** | 信息密度 | 单位信息的 token 数 |
| **连贯性** | 上下文的逻辑流 | LLM 评分、人工评估 |
| **多样性** | 避免冗余 | 熵度量、重复检测 |

### 输出质量指标

| 指标 | 定义 | 工具 |
|------|------|------|
| **准确性** | 答案的正确性 | 基准真值比较 |
| **忠实性** | 对所提供上下文的遵守程度 | 幻觉检测 |
| **有用性** | 对用户的实用性 | 用户反馈、评分 |
| **完整性** | 完整回答问题 | 清单评估 |
| **可读性** | 写作质量 | 可读性评分 |

## 最佳实践

### 应该做 ✓

1. **分层设计上下文**
   - 系统指令优先
   - 领域知识次之
   - 特定任务上下文最后

2. **使用结构化格式**
   ```markdown
   ## 上下文：API 文档
   ### 函数：createUser
   - 参数：name (string), email (string)
   - 返回：带有 ID 的 User 对象
   ```

3. **包含相关示例**
   - 少样本学习提升性能
   - 同时展示正反示例
   - 变化示例复杂度

4. **实施上下文版本控制**
   - 跟踪上下文随时间的变化
   - A/B 测试不同的上下文策略
   - 监控性能指标

5. **对大型知识库使用检索**
   - RAG 优于暴力包含
   - 实施语义搜索
   - 添加相关性评分

### 不应该做 ✗

1. **不要超载上下文窗口**
   - 更多 ≠ 更好
   - 注意力分散损害性能
   - 使用检索和优先级排序

2. **不要包含矛盾信息**
   - 困惑模型
   - 导致输出不一致
   - 验证上下文一致性

3. **不要使用模糊指令**
   - 明确格式要求
   - 清晰指定约束
   - 提供具体示例

4. **不要忽略 token 成本**
   - 优化效率
   - 适当时使用压缩
   - 缓存常用上下文

5. **不要忘记安全约束**
   - 包含系统级防护措施
   - 清晰定义红线
   - 实施内容过滤

## 工具与框架

### 上下文管理

- **LangChain**：用于上下文管理和链编排的综合框架
- **LlamaIndex**：具有多种检索策略的高级 RAG 实现
- **Haystack**：具有上下文处理功能的 NLP 生产框架
- **DSPy**：具有自动优化功能的声明式提示词编程

### 向量存储与检索

- **ChromaDB**：开源嵌入数据库
- **Pinecone**：托管向量数据库服务
- **Weaviate**：知识图增强的向量搜索
- **FAISS**：Facebook 的高效相似性搜索库

### 评估与监控

- **LangSmith**：调试、测试和监控 LLM 应用
- **Arize Phoenix**：ML 可观察性平台
- **Weights & Biases**：AI 系统的实验跟踪
- **Promptfoo**：自动化提示词和 LLM 评估

## 高级主题

### 1. Skills（应用层能力单元）

**什么是 Skills?**

Skills 是上下文工程的**应用层概念**，指通过提示词、工具、知识等上下文配置手段，使基础模型在特定任务上表现出的可复用能力单元。

**核心特征**：

| 特征 | 说明 |
|------|------|
| **工程实践产物** | Skills 不是模型的内置属性，而是通过精心设计的上下文配置实现的能力 |
| **任务导向** | 每个 Skill 针对特定的任务或领域，具有明确的目标 |
| **可组合性** | 多个 Skills 可以组合使用，处理复杂任务 |
| **可复用性** | Skill 可以在不同场景中重复使用，无需重复构建 |

**Skills 与模型的关系**：

```
┌─────────────────────────────────────────────────────────┐
│              基础模型 (Base Model)                │
│  - 通用语言理解能力                                  │
│  - 通用推理能力                                     │
│  - 基础知识                                       │
└─────────────────────────────────────────────────────────┘
                          ↑
                          │ 通过上下文配置
                          │
┌─────────────────────────────────────────────────────────┐
│              Skills (应用层能力单元)                  │
│                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │ Code       │  │ Data       │  │ Writing    │   │
│  │ Reviewer   │  │ Analyst    │  │ Assistant  │   │
│  └────────────┘  └────────────┘  └────────────┘   │
│                                                       │
│  每个 Skill =                                         │
│  - 特定提示词模板                                    │
│  + 工具调用配置                                     │
│  + 知识来源定义                                     │
│  + 行为约束                                         │
└─────────────────────────────────────────────────────────┘
                          ↑
                          │ 部署为
                          │
┌─────────────────────────────────────────────────────────┐
│              AI 应用 / 服务                            │
│  - 代码助手                                          │
│  - 数据分析平台                                      │
│  - 内容生成工具                                      │
└─────────────────────────────────────────────────────────┘
```

**Skill 的构成要素**：

#### 1. 提示词配置
定义 Skill 的行为模式和输出规范。

```yaml
skill_name: "code_reviewer"

system_prompt: |
  You are a senior code reviewer with expertise in:
  - Security best practices
  - Performance optimization
  - Maintainability patterns
  - Clean code principles

  Review the provided code and provide:
  1. Security concerns (if any)
  2. Performance issues (if any)
  3. Maintainability improvements
  4. Overall assessment

  Be constructive and specific in your feedback.

output_format: "structured_markdown"
tone: "professional, helpful, precise"
```

#### 2. 工具调用配置
定义 Skill 可以使用的工具和调用策略。

```yaml
skill_name: "code_reviewer"

tools:
  - name: "analyze_security"
    description: "Scan code for security vulnerabilities"
    triggers:
      - "security review"
      - "vulnerability scan"

  - name: "check_performance"
    description: "Analyze code performance characteristics"
    triggers:
      - "performance review"
      - "optimization check"

tool_selection_rules:
  default_tool: "manual_analysis"
  conditional_tools:
    - condition: "code_contains_database_queries"
      use_tool: "analyze_security"
    - condition: "code_is_complex_algorithm"
      use_tool: "check_performance"
```

#### 3. 知识来源配置
定义 Skill 依赖的知识库和检索策略。

```yaml
skill_name: "code_reviewer"

knowledge_sources:
  - name: "security_guidelines"
    type: "vector_store"
    retrieval:
      method: "semantic_search"
      top_k: 5
      threshold: 0.8

  - name: "language_docs"
    type: "api_reference"
    uri: "https://docs.python.org/3"
    retrieval:
      method: "keyword_search"
      fallback: "web_search"

  - name: "project_patterns"
    type: "codebase"
    path: "./examples/"
    retrieval:
      method: "similarity_search"
      language: "python"
```

**Skill 设计原则**：

| 原则 | 说明 | 示例 |
|------|------|------|
| **单一职责** | 每个 Skill 专注于一个特定任务 | Code Reviewer ≠ Code Reviewer + Formatter |
| **原子性** | Skill 不能再分解为更小的独立单元 | Code Analysis Skill 可分解 → 需重构 |
| **可测试性** | Skill 的输入/输出可验证 | 输入：代码片段 → 输出：结构化报告 |
| **独立性** | Skill 尽量减少对其他 Skill 的依赖 | Code Reviewer 不依赖 Code Formatter |
| **可组合性** | 多个 Skills 可组合完成复杂任务 | Code Reviewer + Security Scanner = Comprehensive Review |

**Skills 组合模式**：

#### 1. 串行组合
```
Task → Skill A → Skill B → Skill C → Final Result
```

**示例**：代码审查流程
```
Code → Style Checker → Security Scanner → Performance Analyzer → Final Report
```

#### 2. 并行组合
```
            ┌→ Skill A ─┐
Task ─────┼→ Skill B ─┼→ Aggregation → Final Result
            └→ Skill C ─┘
```

**示例**：综合代码分析
```
Code → [Style, Security, Performance] → Aggregated Report
```

#### 3. 条件组合
```
Task → Condition Check
           ↓
      [Path A] → Skill A ─┐
                      │ → Final Result
      [Path B] → Skill B ─┘
```

**示例**：根据文件类型选择处理方式
```
File → Detect Type
           ↓
      [Python] → Python Analyzer ─┐
                              │ → Analysis Result
      [JavaScript] → JS Analyzer ─┘
```

#### 4. 层次组合
```
Task → High-Level Skill
           ↓
      ┌─ Sub-Skill 1 ─┐
      │                │ → Aggregated Output
      └─ Sub-Skill 2 ─┘
```

**示例**：全面代码审查
```
Code Reviewer
  ├── Style Checker
  ├── Security Scanner
  ├── Performance Analyzer
  └── Documentation Validator
```

**Skill 注册与发现**：

```typescript
// Skill Registry
class SkillRegistry {
  private skills: Map<string, Skill> = new Map();

  register(skill: Skill): void {
    if (skill.validate()) {
      this.skills.set(skill.id, skill);
    }
  }

  discover(query: string): Skill[] {
    // Analyze query to find matching skills
    const queryAnalysis = this.analyzeQuery(query);

    // Find skills by capability matching
    return Array.from(this.skills.values())
      .filter(skill => this.matches(queryAnalysis, skill));
  }

  compose(skillIds: string[]): CompositeSkill {
    // Combine multiple skills into composite
    const skills = skillIds.map(id => this.get(id));
    return new CompositeSkill(skills);
  }
}

// Skill Definition
interface Skill {
  id: string;
  name: string;
  description: string;
  capabilities: string[];
  prompt: PromptConfig;
  tools: ToolConfig[];
  knowledge: KnowledgeConfig[];
  validate(): boolean;
  execute(context: Context): Promise<Result>;
}
```

**Skills 与其他概念的关系**：

| 概念 | 关系 | 说明 |
|------|------|------|
| **提示词工程** | Skill 的实现手段 | 提示词是配置 Skill 行为的核心方式 |
| **工具调用** | Skill 的能力扩展 | 工具使 Skill 能执行超出模型能力的操作 |
| **知识检索** | Skill 的知识基础 | RAG 为 Skill 提供特定领域的专业知识 |
| **Spec** | Skill 的契约定义 | Spec 定义 Skill 的输入/输出和行为规范 |
| **MCP** | Skill 的工具接入层 | MCP 提供标准化的工具访问协议 |

**Skill 最佳实践**：

1. **设计阶段**
   - 明确 Skill 的单一职责
   - 定义清晰的输入/输出契约
   - 评估可测试性和可组合性

2. **实现阶段**
   - 提示词：结构化、可维护、版本化
   - 工具：最小化、安全、有回退策略
   - 知识：来源可靠、检索高效、持续更新

3. **测试阶段**
   - 单元测试：验证 Skill 的独立行为
   - 集成测试：验证 Skills 的组合效果
   - 回归测试：确保更新不破坏现有能力

4. **部署阶段**
   - 版本管理：追踪 Skill 的演化
   - 监控：追踪使用情况和性能指标
   - A/B 测试：持续优化效果

**示例：完整的 Skill 定义**

```yaml
skill:
  id: "security_code_reviewer"
  name: "Security Code Reviewer"
  version: "2.1.0"
  description: "Reviews code for security vulnerabilities and compliance"

  # 任务目标
  objective: "Identify security vulnerabilities and provide remediation guidance"

  # 提示词配置
  prompt_config:
    system_message: |
      You are a security expert specializing in code security review.
      Your role is to:
      1. Identify potential security vulnerabilities
      2. Explain the risks associated with each issue
      3. Provide actionable remediation steps
      4. Reference relevant security standards (OWASP, CWE, etc.)

      Be thorough but prioritize issues by severity.
      Use CWE (Common Weakness Enumeration) IDs where applicable.

    output_template: |
      ## Security Review Report

      ### Summary
      {summary}

      ### Critical Issues
      {critical_issues}

      ### Medium Issues
      {medium_issues}

      ### Low Issues
      {low_issues}

      ### Recommendations
      {recommendations}

  # 工具配置
  tool_config:
    tools:
      - name: "cwe_lookup"
        description: "Query CWE database for vulnerability details"

      - name: "dependency_scan"
        description: "Scan dependencies for known vulnerabilities"

    usage_policy:
      default: "analysis_first"
      parallel: true
      timeout: 30_seconds

  # 知识配置
  knowledge_config:
    sources:
      - name: "owasp_top_10"
        type: "vector_db"
        relevance_threshold: 0.85

      - name: "cwe_database"
        type: "api"
        uri: "https://cwe.mitre.org/api"

      - name: "security_patterns"
        type: "knowledge_graph"
        retrieval: "graph_traversal"

  # 行为约束
  constraints:
    must:
      - "prioritize_critical_vulnerabilities"
      - "provide_remediation_steps"
      - "reference_security_standards"

    must_not:
      - "flag_false_positives_as_critical"
      - "suggest_over_engineered_solutions"
      - "ignore_best_practices"

  # 质量指标
  quality_metrics:
    - name: "vulnerability_detection_rate"
      target: "> 0.90"

    - name: "false_positive_rate"
      target: "< 0.10"

    - name: "remediation_helpfulness"
      target: "> 0.85"

  # 兼容性
  compatibility:
    supported_languages: ["python", "javascript", "java", "go", "rust"]
    min_context_window: 8000_tokens
    recommended_models: ["claude-3.5", "gpt-4"]
```

### 2. 动态上下文选择

根据以下因素自适应的上下文：
- 查询复杂度
- 可用的上下文空间
- 历史性能
- 用户反馈

### 3. 多智能体上下文系统

```
Specialist Agents → Context Exchange → Orchestration → Unified Response
```

Each agent operates with optimized context for its domain, then shares relevant insights.

**Skills 在多智能体系统中的应用**：

在多智能体架构中，Skills 作为能力单元被分配给不同的专业化 Agent：

```yaml
multi_agent_system:
  agents:
    - id: "research_agent"
      skills: ["semantic_search", "document_synthesis", "citation_generation"]
      context_source: "knowledge_base"

    - id: "analysis_agent"
      skills: ["data_analysis", "pattern_detection", "insight_generation"]
      context_source: "raw_data"

    - id: "writing_agent"
      skills: ["technical_writing", "documentation_formatting", "audience_adaptation"]
      context_source: "content_outline"
```

**基于 Skills 的任务分发**：

```
User Request → Skills Required → Agent Discovery → Context Packaging → Task Delegation
                          ↓
                    [search, analyze, write]
                          ↓
    ┌──────────────┬──────────────┬──────────────┐
    ↓              ↓              ↓              ↓
Research Agent Analysis Agent Writing Agent Orchestrator
```

### 4. 上下文缓存策略

- **LRU 缓存**：最近最少使用的上下文
- **语义缓存**：相似查询的重用
- **预测性预取**：预期需要的上下文
- **分层缓存**：具有不同粒度的多级缓存

### 5. 流式上下文

- 渐进式上下文加载
- 实时上下文更新
- 动态上下文替换
- 流式 token 优化

## 标准化与协议

### 规范语言（Spec）

**什么是 Spec？**

规范语言（Spec）是一种用于定义 AI 系统中交互边界、行为契约和上下文需求的形式化语言。它是上下文工程的"API 契约"——对系统应如何行为的精确、可测试、可共享的定义。

**为什么 Spec 对上下文工程很重要**

Spec 将上下文工程从一种艺术形式转化为工程学科：

| 没有 Spec | 有 Spec |
|------------|----------|
| 对上下文需求的隐式假设 | 明确、文档化的上下文需求 |
| 交互中的行为不一致 | 通过契约保证一致性 |
| 手动、容易出错的提示词编写 | 自动化、可测试的提示词生成 |
| 团队特定的最佳实践 | 可共享、版本控制的标准 |
| "在我的机器上能运行"的可重现性问题 | 保证跨环境的可重现性 |

**Spec 在上下文工程场景中的应用**

#### 1. 单智能体上下文规范

对于独立的 AI 智能体，Spec 定义：

- **角色与行为**：智能体应该和不应该做什么
- **输入/输出契约**：预期的上下文格式和响应结构
- **上下文需求**：可靠运行所需的信息
- **质量约束**：性能、准确性、安全要求

**示例：代码审查智能体 Spec**
```yaml
spec_version: "1.0"
agent_type: "single_agent"

identity:
  name: "CodeReviewBot"
  role: "资深代码审查员"
  expertise: ["安全性", "性能", "可维护性"]
  tone: "建设性、精确、全面"

context_requirements:
  mandatory:
    - "code_content"
    - "language_identifier"
    - "repository_context"
  optional:
    - "previous_reviews"
    - "team_guidelines"
    - "ci_cd_status"

behavioral_constraints:
  must_not:
    - "suggest deprecated APIs"
    - "break existing patterns without justification"
    - "exceed 5 suggestions per review"
  must:
    - "explain security implications"
    - "reference official documentation"
    - "estimate effort for each suggestion"

output_contract:
  format: "structured_markdown"
  sections:
    - "summary"
    - "security_issues"
    - "performance_opportunities"
    - "maintainability_suggestions"
  max_length: 2000_tokens
```

#### 2. Multi-Agent Boundary Specification

For systems with multiple agents, Specs define:

- **Agent Capabilities**: What each agent can do (Skills definition)
- **Interaction Boundaries**: Which agent handles which types of tasks
- **Context Handoff Protocols**: How context flows between agents
- **Conflict Resolution**: How to handle overlapping responsibilities

**Example: Multi-Agent Team Spec**
```yaml
spec_version: "1.0"
system_type: "multi_agent"

agents:
  - id: "researcher"
    capabilities:
      - "semantic_search"
      - "document_synthesis"
      - "citation_generation"
    context_requirements:
      - "query"
      - "knowledge_base"
      - "retrieval_strategy"
    output_provides: ["research_summary", "sources"]

  - id: "analyst"
    capabilities:
      - "data_analysis"
      - "pattern_detection"
      - "insight_generation"
    context_requirements:
      - "raw_data"
      - "analysis_parameters"
      - "domain_context"
    output_provides: ["analysis_report", "visualizations"]

  - id: "writer"
    capabilities:
      - "technical_writing"
      - "documentation_formatting"
      - "audience_adaptation"
    context_requirements:
      - "content_outline"
      - "style_guide"
      - "target_audience"
    output_provides: ["final_document", "metadata"]

interaction_boundaries:
  - trigger: "information_request"
    primary_agent: "researcher"
    fallback_agents: ["analyst"]

  - trigger: "data_analysis_request"
    primary_agent: "analyst"
    fallback_agents: ["researcher"]

  - trigger: "content_creation"
    primary_agent: "writer"
    context_sources: ["researcher", "analyst"]

context_handoff_protocols:
  handoff_format: "structured_json"
  required_fields:
    - "task_context"
    - "intermediate_results"
    - "state_snapshot"
  validation:
    - "completeness_check"
    - "consistency_check"

conflict_resolution:
  strategy: "primary_agent_decides"
  escalation: "human_in_arbitration"
```

#### 3. Tool Invocation Specification

For function calling and tool use, Specs define:

- **Tool Interfaces**: Input/output schemas
- **Tool Selection Rules**: Which tool to use when
- **Error Handling**: How to handle tool failures
- **Tool Composition**: How tools can be chained

**Example: Tool Invocation Spec**
```yaml
spec_version: "1.0"
type: "tool_suite"

tools:
  - name: "database_query"
    description: "Execute SQL queries on production database"
    input_schema:
      type: "object"
      properties:
        query:
          type: "string"
          pattern: "^(SELECT|INSERT|UPDATE|DELETE).*"
        timeout:
          type: "integer"
          default: 5000
    output_schema:
      type: "array"
      items:
        type: "object"
    constraints:
      max_query_length: 10000
      requires_permission: "db_read_write"

  - name: "file_system_read"
    description: "Read files from local filesystem"
    input_schema:
      type: "object"
      properties:
        path:
          type: "string"
          pattern: "^/workspace/.*"
        encoding:
          type: "string"
          enum: ["utf-8", "base64"]
          default: "utf-8"
    output_schema:
      type: "object"
      properties:
        content:
          type: "string"
        metadata:
          type: "object"
    constraints:
      max_file_size: 1048576  # 1MB
      allowed_extensions: [".md", ".txt", ".json", ".yaml"]

tool_selection_rules:
  - condition: "data_retrieval_from_database"
    preferred_tools: ["database_query"]
    fallback_tools: ["api_call"]

  - condition: "reading_local_files"
    preferred_tools: ["file_system_read"]
    fallback_tools: ["api_call"]

tool_composition:
  chains:
    - name: "report_generation"
      steps:
        - tool: "database_query"
          context_pass: true
        - tool: "file_system_read"
          depends_on: "database_query"
        - tool: "data_analysis"
          depends_on: "file_system_read"

error_handling:
  retry_policy:
    max_retries: 3
    backoff_strategy: "exponential"
  fallback_actions:
    - "use_cached_result"
    - "alert_user"
    - "log_error_and_continue"
```

#### 4. Prompt Engineering Specification

For prompt design and optimization, Specs define:

- **Prompt Templates**: Reusable prompt structures
- **Variable Binding**: How to fill in dynamic values
- **Prompt Versioning**: Track prompt evolution
- **A/B Testing Framework**: Compare prompt variants

**Example: Prompt Engineering Spec**
```yaml
spec_version: "1.0"
type: "prompt_engineering"

prompt_templates:
  - name: "code_explanation"
    description: "Explain code to developers"
    template: |
      You are a {role}.
      Explain the following {language} code:

      ```{language}
      {code}
      ```

      Focus on:
      1. What the code does (high-level)
      2. Key patterns and techniques used
      3. Potential improvements

      Output format: {output_format}
    variables:
      role:
        type: "string"
        default: "Senior Software Engineer"
      language:
        type: "string"
        required: true
      code:
        type: "string"
        required: true
      output_format:
        type: "string"
        enum: ["markdown", "plain_text"]
        default: "markdown"
    constraints:
      max_code_length: 5000
      estimated_output_tokens: 800

  - name: "bug_fix_assistance"
    description: "Help fix bugs in code"
    template: |
      Context: {context}

      Bug Report:
      {bug_report}

      Code:
      ```{language}
      {code}
      ```

      Task: Identify and fix the bug.

      Steps:
      1. Analyze the bug report
      2. Locate the issue in the code
      3. Propose a fix
      4. Explain your reasoning
    variables:
      context:
        type: "string"
        required: true
      bug_report:
        type: "string"
        required: true
      language:
        type: "string"
        required: true
      code:
        type: "string"
        required: true

prompt_versioning:
  current_version: "v2.3"
  changelog:
    - version: "v2.3"
      date: "2025-01-15"
      changes:
        - "Improved code explanation template"
        - "Added output_format variable"
    - version: "v2.2"
      date: "2025-01-10"
      changes:
        - "Fixed bug in bug_fix_assistance template"

ab_testing:
  active_tests:
    - test_id: "prompt_variant_comparison"
      template_name: "code_explanation"
      variants:
        - version: "v2.3"
          weight: 0.7
        - version: "v2.2"
          weight: 0.3
      metrics:
        - "user_satisfaction"
        - "response_time"
        - "helpfulness_rating"
      duration: 14_days
```

**Spec Implementation Pattern**

```typescript
// Spec-driven Context Builder
class SpecDrivenContextBuilder {
  constructor(private spec: ContextSpecification) {}

  async build(contextData: ContextData): Promise<BuiltContext> {
    // 1. Validate inputs against spec
    this.validateInputs(contextData, this.spec.input_contract);

    // 2. Assemble context based on spec requirements
    const context = await this.assembleContext(
      contextData,
      this.spec.context_requirements
    );

    // 3. Apply constraints from spec
    const constrained = this.applyConstraints(
      context,
      this.spec.behavioral_constraints
    );

    // 4. Format output according to spec
    return this.formatOutput(
      constrained,
      this.spec.output_contract
    );
  }
}
```

**Spec Benefits**

| Benefit | Description | Impact |
|---------|-------------|--------|
| **Reproducibility** | Identical behavior across runs | Debugging, testing, deployment |
| **Testability** | Automated verification of specs | CI/CD integration, quality assurance |
| **Version Control** | Track changes over time | Rollback capabilities, audit trails |
| **Team Collaboration** | Shared understanding of requirements | Reduced miscommunication |
| **Scalability** | Compose specs for complex systems | Handle growth gracefully |
| **Compliance** | Documented behavior for audits | Regulatory requirements |

### 模型上下文协议（MCP）

**What is MCP?**

The Model Context Protocol (MCP) is an open standard introduced by Anthropic in November 2024 that standardizes how AI systems connect to external tools and data sources. It's the "USB-C for AI" — a universal interface that eliminates the need for custom integrations.

**Why MCP Matters for Context Engineering**

MCP is a concrete implementation of Spec principles for context exchange:

| Before MCP | With MCP |
|------------|----------|
| Custom integration for each data source | Standardized interface across all sources |
| Manual context assembly | Automatic discovery and context retrieval |
| Fragile, brittle connections | Robust, maintainable integrations |
| Team-specific patterns | Industry-wide best practices |

**Core MCP Components**

#### 1. Tools (Actions)

Tools are executable functions that AI agents can invoke to perform actions.

**Context Engineering Perspective**:
- Tool descriptions become part of the context
- Tools enrich context with real-time data
- Tool selection is a context optimization problem

**Example Tool Definition**:
```json
{
  "name": "search_database",
  "description": "Query the PostgreSQL database for customer records",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "SQL query to execute"
      }
    },
    "required": ["query"]
  }
}
```

**Context Impact**: Tool definitions are included in the system context, enabling the AI to understand available actions and invoke them appropriately.

#### 2. Resources (Data)

Resources provide read-only access to data and information.

**Context Engineering Perspective**:
- Resources are the primary source of dynamic context
- Resource selection determines context relevance
- Resource structure affects how context is consumed

**Example Resource Definition**:
```json
{
  "uri": "file:///workspace/config/app.yaml",
  "name": "application_config",
  "description": "Main application configuration file",
  "mimeType": "application/yaml"
}
```

**Context Impact**: Resources are retrieved and injected into context based on relevance to the task, providing just-in-time information.

#### 3. Prompts (Templates)

Prompts are reusable, parameterized instruction templates.

**Context Engineering Perspective**:
- Prompt templates standardize context patterns
- Prompts ensure consistent context structure
- Prompts encapsulate best practices

**Example Prompt Template**:
```json
{
  "name": "analyze_code",
  "description": "Analyze code for issues and improvements",
  "arguments": {
    "language": "The programming language",
    "focus_areas": "Areas to focus on (security, performance, style)"
  }
}
```

**Context Impact**: Prompt templates provide structured context that can be dynamically populated, ensuring consistent quality across interactions.

**MCP Architecture for Context Engineering**

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (AI Application)             │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Context      │  │ Context      │  │ Context      │      │
│  │ Engine       │  │ Router       │  │ Compressor   │      │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼────────────────┼──────────────────┼──────────────┘
          │                │                  │
          ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Context Protocol                   │
│  (Standardized Context Exchange Layer - Spec-Driven)       │
└─────────────────────────────────────────────────────────────┘
          │                │                  │
          ▼                ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ MCP Server:  │  │ MCP Server:  │  │ MCP Server:  │
│ Database     │  │ File System  │  │ API Gateway  │
└──────────────┘  └──────────────┘  └──────────────┘
```

**MCP Integration Best Practices**

1. **Context Discovery Phase**
   ```typescript
   // Discover available capabilities
   const capabilities = await mcpClient.discover();

   // Analyze to determine context needs
   const contextStrategy = planContextStrategy(
     userQuery,
     capabilities.tools,
     capabilities.resources
   );
   ```

2. **Dynamic Context Assembly**
   ```typescript
   // Select relevant resources
   const relevantResources = selectResources(
     capabilities.resources,
     contextStrategy.query,
     contextBudget
   );

   // Include appropriate tool descriptions
   const toolContext = filterTools(
     capabilities.tools,
     contextStrategy.requiredCapabilities
   );
   ```

3. **Context Optimization**
   ```typescript
   // Compress retrieved resources
   const compressedResources = await compressContext(
     relevantResources,
     targetTokenCount
   );

   // Prioritize by relevance score
   const prioritizedContext = prioritizeByRelevance(
     [toolContext, compressedResources],
     contextStrategy.weights
   );
   ```

**MCP + Context Engineering Pattern**

```
User Request
    ↓
┌─────────────────────────────────────┐
│  1. Parse & Understand Intent        │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  2. Load Context Spec               │
│     (Define what context is needed)  │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  3. Query MCP Capabilities           │
│     (Discover available data/tools)  │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  4. Assemble Dynamic Context         │
│     (Select & retrieve from MCP)     │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  5. Optimize Context                │
│     (Compress, prioritize, format)   │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  6. Send to Model                    │
│     (With optimized context)         │
└─────────────┬───────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  7. Process & Iterate                │
│     (Use MCP tools if needed)        │
└─────────────────────────────────────┘
```

**MCP Servers for Context Engineering**

| Server Type | Context Contribution | Use Case |
|-------------|---------------------|----------|
| **Filesystem** | Source code, documentation, configs | Code assistants, documentation bots |
| **Database** | Structured data, records | Data analysis, reporting agents |
| **API Gateway** | Real-time data, external services | Information gathering agents |
| **Vector Store** | Semantic search results | RAG-based systems |
| **Git** | Version history, diffs | Code review, changelog generation |

**Getting Started with MCP**

1. **Install MCP Client SDK**
   ```bash
   npm install @modelcontextprotocol/sdk
   ```

2. **Connect to MCP Servers**
   ```typescript
   import { Client } from '@modelcontextprotocol/sdk/client/index.js';

   const client = new Client({
     name: "my-context-app",
     version: "1.0.0"
   });

   await client.connect({
     command: "node",
     args: ["path/to/mcp-server"]
   });
   ```

3. **Discover Capabilities**
   ```typescript
   const tools = await client.listTools();
   const resources = await client.listResources();
   const prompts = await client.listPrompts();
   ```

4. **Integrate into Context Builder**
   ```typescript
   class MCPContextBuilder {
     constructor(private mcpClient: Client) {}

     async buildForTask(query: string): Promise<Context> {
       const strategy = await this.planContext(query);
       const resources = await this.fetchResources(strategy);
       const tools = await this.selectTools(strategy);

       return new Context()
         .addInstructions(strategy.instructions)
         .addResources(resources)
         .addTools(tools)
         .compress(strategy.budget);
     }
   }
   ```

**Resources**

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)
- [MCP Server Registry](https://registry.modelcontextprotocol.io/)

## 常见陷阱

### 陷阱 1："厨房水槽"方法
**问题**：在上下文中包含所有可用内容
**解决方案**：严格的相关性过滤和优先级排序

### 陷阱 2：忽视模型能力
**问题**：要求超出模型能力的推理
**解决方案**：了解模型优势，使用适当的技术

### 陷阱 3：静态上下文
**问题**：无论查询如何都使用相同的上下文
**解决方案**：动态上下文选择和组装

### 陷阱 4：没有评估
**问题**：假设上下文是最优的而不进行测试
**解决方案**：持续 A/B 测试和指标跟踪

### 陷阱 5：忽视 Token 成本
**问题**：没有优化的昂贵上下文使用
**解决方案**：实施压缩、缓存、智能检索

## 案例研究

### 案例 1：代码助手

**挑战**：为多种语言和框架提供准确的代码帮助

**解决方案**：
- 分层上下文：系统规则 → 语言文档 → 代码库片段 → 用户查询
- 对大型代码库使用 RAG
- 基于检测到的语言/框架的动态上下文

**结果**：错误建议减少 40%，响应时间加快 60%

### 案例 2：研究助手

**挑战**：综合来自跨领域学术论文的信息

**解决方案**：
- 多阶段检索（关键词 → 语义 → 基于引用）
- 相关论文的上下文摘要
- 引用链接的上下文结构

**结果**：提高了综合质量，保持了源的可追溯性

### 案例 3：客户支持

**挑战**：为数千种产品提供一致、准确的支持

**解决方案**：
- 特定产品的知识库
- 分层上下文（常见问题 → 详细文档 → 工程笔记）
- 基于用户历史的上下文个性化

**结果**：升级率降低 50%，提高了客户满意度

## 资源

### 学术论文
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al., 2022)
- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)

### 博客与文章
- [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/claude/prompt-engineering)
- [OpenAI's Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain's Context Window Management](https://python.langchain.com/docs/modules/memory/context_window_management)

### 社区
- [Context Engineering Discord](https://discord.gg/example)
- [r/ContextEngineering on Reddit](https://reddit.com/r/ContextEngineering)
- [Context Engineering Slack](https://contexteng.slack.com)

## 贡献

欢迎贡献！感兴趣领域：
- 新的上下文优化技术
- 案例研究和基准测试
- 工具集成和框架
- 评估方法

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解指南。

## 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

*系统性阐述上下文工程以构建更好的 AI 系统*
