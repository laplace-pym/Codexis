# Claude Code 逆向工程 - 实用方案

## 问题分析

原教程中的 monkey patch 方法遇到了问题：
- Claude CLI 是一个 **48 万行**的打包文件
- 简单地在文件末尾添加代码无法有效拦截 API 调用
- 需要更深入的逆向工程技术

## 替代方案

### 方案 1：使用网络代理（推荐）

使用 **mitmproxy** 拦截 HTTPS 请求：

```bash
# 安装 mitmproxy
brew install mitmproxy

# 启动代理
mitmweb --mode reverse:https://api.anthropic.com@8080

# 配置环境变量让 Claude 使用代理
export HTTPS_PROXY=http://localhost:8080
export NODE_TLS_REJECT_UNAUTHORIZED=0

# 运行 Claude
claude
```

所有 API 请求会在浏览器中显示：http://localhost:8081

### 方案 2：使用 Chrome DevTools Protocol

如果 Claude Code 在 IDE 中运行，可以通过 DevTools 查看网络请求。

### 方案 3：分析已有的逆向结果

直接学习原仓库已经分析好的结果：

1. **Prompts 目录**：https://github.com/Yuyz0112/claude-code-reverse/tree/main/prompts
   - `system-workflow.txt` - 核心 Agent 工作流
   - `system-compact.txt` - 上下文压缩
   - `task-instructions.txt` - 任务管理
   
2. **Tools 目录**：https://github.com/Yuyz0112/claude-code-reverse/tree/main/tools
   - 工具定义和使用方式

3. **可视化工具**：https://yuyz0112.github.io/claude-code-reverse/visualize.html
   - 已经有现成的日志可以分析

## 快速学习方案（无需逆向）

### 1. 查看现成的分析结果

```bash
# 克隆仓库
git clone https://github.com/Yuyz0112/claude-code-reverse.git
cd claude-code-reverse

# 查看核心 prompt
cat prompts/system-workflow.txt
cat prompts/system-reminder-start.txt
cat prompts/task-instructions.txt

# 查看工具定义
ls tools/
```

### 2. 分析可视化日志

访问：https://yuyz0112.github.io/claude-code-reverse/visualize.html

示例日志在 `logs/` 目录下。

## 关键学习点

### 1. System Workflow Prompt

定义了 Agent 的核心行为：
- 如何理解任务
- 何时使用工具
- 如何管理上下文
- 错误处理策略

### 2. Tools 设计

Claude Code 使用的工具类型：
- **文件操作**：read_file, write_file, search_files
- **代码执行**：execute_command, run_terminal
- **上下文管理**：list_files, get_file_info
- **任务管理**：TodoWrite (短期记忆)
- **Sub Agent**：启动子任务

### 3. Context Management

- **压缩策略**：当上下文不足时触发
- **短期记忆**：通过 Todo JSON 文件保存
- **Sub Agent**：隔离"脏上下文"

### 4. Multi-Agent 设计

- 主 Agent 处理用户交互
- Sub Agent 处理独立任务（如代码搜索）
- 结果返回时只保留关键信息

## 实践建议

### 方法 A：直接学习已有结果

1. 阅读 `prompts/` 中的所有提示词
2. 理解每个工具的用途
3. 分析示例日志中的对话流程
4. 在自己的项目中应用这些模式

### 方法 B：使用网络代理

```bash
# 终端 1：启动代理
mitmweb

# 终端 2：使用 Claude
export HTTPS_PROXY=http://localhost:8080
export NODE_TLS_REJECT_UNAUTHORIZED=0
claude
```

### 方法 C：参考官方文档

Anthropic 已经公开了很多关于 Agent 设计的最佳实践：
- https://docs.anthropic.com/en/docs/agents
- Tool use 指南
- Prompt engineering 指南

## 核心收获

不需要完整逆向，重点学习：
1. **Prompt 设计模式** - 如何指导 LLM 行为
2. **工具调用策略** - 何时使用哪些工具
3. **上下文管理** - 如何节省 token
4. **Multi-Agent 架构** - 任务隔离和结果传递

## 下一步

建议直接阅读已逆向的结果，然后在自己的项目（Codexis）中应用这些模式！
