# 🔍 Claude Code 日志捕获 - 使用指南

## 快速开始（3 步）

### 第 1 步：运行设置脚本

```bash
cd /Users/bytedance/Desktop/Codexis
./setup-claude-logging.sh
```

这会：
- ✅ 安装 mitmproxy（网络代理工具）
- ✅ 创建日志捕获脚本
- ✅ 设置好所有必要的文件

### 第 2 步：启动日志捕获（终端 1）

**打开第一个终端窗口**，运行：

```bash
~/start-claude-logging.sh
```

你会看到：
```
🚀 启动 Claude Code 日志捕获
================================
📍 日志保存位置: ~/.claude-reverse-logs
⚠️  保持这个窗口运行
```

**不要关闭这个窗口！** 它会实时显示拦截的请求。

### 第 3 步：使用 Claude Code（终端 2）

**打开第二个终端窗口**，运行：

```bash
~/claude-logged
```

或者手动设置环境变量：

```bash
export HTTPS_PROXY=http://127.0.0.1:8080
export NODE_TLS_REJECT_UNAUTHORIZED=0
claude
```

## 测试示例

在 Claude Code 中尝试：

```
❯ 帮我写一个计算斐波那契数列的函数
```

此时：
- **终端 1**：会显示拦截到的 API 请求和响应
- **日志文件**：会自动保存在 `~/.claude-reverse-logs/`

## 查看日志

```bash
# 列出所有日志文件
ls -lh ~/.claude-reverse-logs/

# 查看最新的日志
ls -t ~/.claude-reverse-logs/*.log | head -1 | xargs cat

# 或用 jq 格式化查看
ls -t ~/.claude-reverse-logs/*.log | head -1 | xargs cat | jq .
```

## 分析日志

### 方法 1：在线可视化工具

1. 访问：https://yuyz0112.github.io/claude-code-reverse/visualize.html
2. 上传你的日志文件
3. 查看完整的对话流程、prompts、工具调用

### 方法 2：手动分析

打开日志文件，重点关注：

1. **System Prompts**（系统提示词）
   - 定义 Agent 的角色和行为
   - 工作流程指导

2. **Tools**（工具定义）
   - 有哪些工具可用
   - 工具的参数和使用方式

3. **Messages**（对话消息）
   - 用户输入
   - Agent 的思考过程
   - 工具调用和结果

4. **Context Management**
   - 如何压缩历史对话
   - Todo 记忆管理

## 停止日志捕获

在终端 1 中按 `Ctrl+C` 停止 mitmproxy。

## 常见问题

### Q: 看到 "证书错误" 怎么办？
A: 设置了 `NODE_TLS_REJECT_UNAUTHORIZED=0` 就会忽略证书验证

### Q: Claude Code 无法连接？
A: 确保：
1. 终端 1 的 mitmproxy 正在运行
2. 环境变量 `HTTPS_PROXY` 设置正确
3. 端口 8080 没有被占用

### Q: 日志文件为空？
A: 可能是：
1. Claude Code 没有真正发送 API 请求
2. 代理设置不正确
3. 检查终端 1 是否有错误信息

### Q: 如何查看实时日志？
A: 在新终端运行：
```bash
tail -f ~/.claude-reverse-logs/*.log
```

## 学习建议

1. **先做简单任务**："写一个 Hello World"
   - 查看基础的 API 交互
   
2. **再做复杂任务**："分析这个项目并重构"
   - 观察 Sub Agent 的启动
   - 理解上下文管理

3. **对比不同场景**：
   - 代码生成 vs 文件操作
   - 简单问答 vs 复杂任务
   - 发现模式的差异

## 下一步

学习到 Claude Code 的设计后，可以：
1. 应用到自己的 Codexis 项目
2. 改进 prompt 设计
3. 优化工具调用策略
4. 实现更好的上下文管理

## 恢复正常使用

如果要正常使用 Claude（不记录日志）：

```bash
# 直接运行（不设置代理）
claude

# 或者清除环境变量
unset HTTPS_PROXY
unset HTTP_PROXY
unset NODE_TLS_REJECT_UNAUTHORIZED
```
