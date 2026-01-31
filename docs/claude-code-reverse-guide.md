# Claude Code 逆向工程教程 - 操作指南

## 概述
这个教程通过监控 Claude Code 与 LLM API 的交互来理解其内部工作原理，而不是直接分析复杂的混淆代码。

## 前置准备

### 1. 安装必要工具
```bash
# 安装 js-beautify（用于格式化 JavaScript）
npm install -g js-beautify

# 确认已安装 Claude CLI
which claude
```

### 2. 克隆逆向工程仓库
```bash
cd ~/Desktop
git clone https://github.com/Yuyz0112/claude-code-reverse.git
cd claude-code-reverse
```

## 步骤一：定位 Claude CLI 文件

```bash
# 1. 找到 claude 命令的位置
which claude
# 输出示例：/usr/local/bin/claude

# 2. 查看它链接到哪个文件
ls -l /usr/local/bin/claude
# 输出示例：/usr/local/bin/claude -> /some/path/node_modules/@anthropic/cli/dist/cli.js

# 3. 记录这个真实路径
REAL_PATH="/some/path/node_modules/@anthropic/cli/dist/cli.js"
```

## 步骤二：备份并格式化 cli.js

```bash
# 1. 进入 cli.js 所在目录
cd "$(dirname "$REAL_PATH")"

# 2. 备份原始文件（重要！）
cp cli.js cli.js.backup

# 3. 格式化 cli.js
js-beautify cli.js > cli.js.formatted
mv cli.js.formatted cli.js
```

## 步骤三：应用 Monkey Patch

### 方法 A：使用仓库提供的 patch 文件

```bash
# 1. 下载 patch 文件
# 从 https://github.com/Yuyz0112/claude-code-reverse/blob/main/cli.js.patch

# 2. 手动查看 patch 内容，找到需要修改的位置
# patch 的核心是拦截 beta.messages.create 方法

# 3. 在 cli.js 中搜索 beta.messages.create 相关代码
# 添加日志记录逻辑
```

### 方法 B：手动添加日志代码

在 cli.js 中找到 `beta.messages.create` 方法调用的地方，添加如下逻辑：

```javascript
// 在方法调用前添加
const fs = require('fs');
const logFile = `messages-${Date.now()}.log`;

// 拦截请求
const originalCreate = client.beta.messages.create;
client.beta.messages.create = async function(...args) {
    // 记录请求
    fs.appendFileSync(logFile, JSON.stringify({
        type: 'request',
        timestamp: new Date().toISOString(),
        data: args[0]
    }) + '\n');
    
    // 执行原始请求
    const result = await originalCreate.apply(this, args);
    
    // 记录响应
    fs.appendFileSync(logFile, JSON.stringify({
        type: 'response',
        timestamp: new Date().toISOString(),
        data: result
    }) + '\n');
    
    return result;
};
```

## 步骤四：使用 Claude Code 生成日志

```bash
# 1. 运行 Claude Code 执行各种任务
claude

# 2. 尝试不同的任务场景：
# - 简单的代码问题："帮我写一个快速排序"
# - 文件操作："读取 README.md 并总结"
# - 复杂任务："分析这个项目的架构"
# - Sub Agent 触发："帮我在代码库中找到所有的 API 调用"

# 3. 每次使用后会生成 messages.log 文件
# 保存这些日志文件用于分析
```

## 步骤五：解析和可视化日志

```bash
# 1. 进入逆向工程仓库目录
cd ~/Desktop/claude-code-reverse

# 2. 使用 parser.js 解析日志
node parser.js path/to/messages.log > parsed-log.json

# 3. 打开 visualize.html 可视化工具
# 在浏览器中打开：file:///path/to/claude-code-reverse/visualize.html
# 或使用在线版本：https://yuyz0112.github.io/claude-code-reverse/visualize.html

# 4. 在可视化工具中加载你的日志文件
```

## 步骤六：分析结果

### 关注的关键点：

1. **System Prompts（系统提示词）**
   - 查看 Claude Code 如何定义 Agent 的角色和行为
   - 工作流程指导
   - 工具使用规则

2. **Tools（工具定义）**
   - 有哪些工具可用
   - 工具的参数和返回值格式
   - 何时调用什么工具

3. **Context Management（上下文管理）**
   - 如何压缩历史对话
   - Todo 短期记忆管理
   - Sub Agent 的启动和结果返回

4. **Multi-Agent Pattern（多 Agent 模式）**
   - 主 Agent 和 Sub Agent 的交互
   - 如何分解和隔离任务

## 步骤七：恢复原始文件（可选）

```bash
# 如果要恢复到原始状态
cd "$(dirname "$REAL_PATH")"
mv cli.js.backup cli.js
```

## 注意事项

⚠️ **重要警告**：
- 修改 Claude CLI 可能违反服务条款
- 仅用于学习和研究目的
- 不要分享或公开敏感的 API 交互数据
- 定期检查是否有 CLI 更新，更新会覆盖你的修改

## 学习目标

通过这个过程，你将理解：
1. Claude Code 的核心工作流程
2. 如何设计 System Prompts 来指导 LLM
3. 工具调用模式和最佳实践
4. 上下文管理策略
5. Multi-Agent 系统设计

## 进阶探索

- 尝试在不同场景下使用 Claude Code
- 对比不同任务的 prompt 差异
- 分析 Sub Agent 的触发条件
- 研究上下文压缩的时机和方法
- 探索 IDE 集成的特殊工具

## 参考资源

- 原仓库：https://github.com/Yuyz0112/claude-code-reverse
- 可视化工具：https://yuyz0112.github.io/claude-code-reverse/visualize.html
- Anthropic TS SDK：https://github.com/anthropics/anthropic-sdk-typescript
