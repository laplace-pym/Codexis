#!/bin/bash

# Claude Code 逆向工程设置脚本
# 基于 https://github.com/Yuyz0112/claude-code-reverse

set -e

echo "🚀 开始 Claude Code 逆向工程设置..."
echo ""

# 定义路径
CLAUDE_BIN="/Users/bytedance/.nvm/versions/node/v24.13.0/bin/claude"
CLAUDE_CLI="/Users/bytedance/.nvm/versions/node/v24.13.0/lib/node_modules/@anthropic-ai/claude-code/cli.js"
CLAUDE_DIR="$(dirname "$CLAUDE_CLI")"
BACKUP_FILE="${CLAUDE_CLI}.backup"
LOG_DIR="$HOME/.claude-reverse-logs"

echo "📍 Claude CLI 位置："
echo "   二进制：$CLAUDE_BIN"
echo "   实际文件：$CLAUDE_CLI"
echo ""

# 步骤 1: 检查 js-beautify
echo "📦 步骤 1/6: 检查 js-beautify..."
if ! command -v js-beautify &> /dev/null; then
    echo "   ⚠️  js-beautify 未安装，正在安装..."
    npm install -g js-beautify
else
    echo "   ✅ js-beautify 已安装"
fi
echo ""

# 步骤 2: 克隆逆向工程仓库
echo "📥 步骤 2/6: 克隆逆向工程仓库..."
REVERSE_REPO="$HOME/claude-code-reverse"
if [ ! -d "$REVERSE_REPO" ]; then
    git clone https://github.com/Yuyz0112/claude-code-reverse.git "$REVERSE_REPO"
    echo "   ✅ 仓库已克隆到 $REVERSE_REPO"
else
    echo "   ✅ 仓库已存在：$REVERSE_REPO"
fi
echo ""

# 步骤 3: 备份原始文件
echo "💾 步骤 3/6: 备份原始 cli.js..."
if [ ! -f "$BACKUP_FILE" ]; then
    cp "$CLAUDE_CLI" "$BACKUP_FILE"
    echo "   ✅ 已备份到 ${BACKUP_FILE}"
else
    echo "   ✅ 备份文件已存在"
fi
echo ""

# 步骤 4: 格式化 cli.js
echo "🎨 步骤 4/6: 格式化 cli.js（这可能需要一些时间）..."
if ! grep -q "// REVERSE_PATCHED" "$CLAUDE_CLI"; then
    js-beautify "$BACKUP_FILE" > "${CLAUDE_CLI}.formatted"
    mv "${CLAUDE_CLI}.formatted" "$CLAUDE_CLI"
    echo "   ✅ 格式化完成"
else
    echo "   ⚠️  文件已被修改过，跳过格式化"
fi
echo ""

# 步骤 5: 创建日志目录
echo "📁 步骤 5/6: 创建日志目录..."
mkdir -p "$LOG_DIR"
echo "   ✅ 日志将保存到 $LOG_DIR"
echo ""

# 步骤 6: 添加 Monkey Patch
echo "🔧 步骤 6/6: 应用 Monkey Patch..."
if ! grep -q "// REVERSE_PATCHED" "$CLAUDE_CLI"; then
    cat >> "$CLAUDE_CLI" << 'PATCH_EOF'

// REVERSE_PATCHED - 逆向工程日志记录
(function() {
    const fs = require('fs');
    const path = require('path');
    const logDir = path.join(process.env.HOME, '.claude-reverse-logs');
    const logFile = path.join(logDir, `messages-${Date.now()}.log`);
    
    // 确保日志目录存在
    if (!fs.existsSync(logDir)) {
        fs.mkdirSync(logDir, { recursive: true });
    }
    
    console.log(`\n🔍 逆向工程日志已启用`);
    console.log(`📝 日志文件: ${logFile}\n`);
    
    // 找到并 patch Anthropic client
    // 注意：这是简化版本，实际的 patch 需要根据具体的代码结构调整
    const originalRequire = require('module').prototype.require;
    require('module').prototype.require = function(id) {
        const module = originalRequire.apply(this, arguments);
        
        if (id === '@anthropic-ai/sdk' || id.includes('anthropic')) {
            try {
                if (module.Anthropic && module.Anthropic.prototype) {
                    const originalMessages = module.Anthropic.prototype.messages;
                    if (originalMessages && !originalMessages.__patched) {
                        const originalCreate = originalMessages.create;
                        originalMessages.create = async function(...args) {
                            // 记录请求
                            fs.appendFileSync(logFile, JSON.stringify({
                                type: 'request',
                                timestamp: new Date().toISOString(),
                                data: args[0]
                            }, null, 2) + '\n---\n');
                            
                            try {
                                const result = await originalCreate.apply(this, args);
                                
                                // 记录响应
                                fs.appendFileSync(logFile, JSON.stringify({
                                    type: 'response',
                                    timestamp: new Date().toISOString(),
                                    data: result
                                }, null, 2) + '\n---\n');
                                
                                return result;
                            } catch (error) {
                                // 记录错误
                                fs.appendFileSync(logFile, JSON.stringify({
                                    type: 'error',
                                    timestamp: new Date().toISOString(),
                                    error: error.message
                                }, null, 2) + '\n---\n');
                                throw error;
                            }
                        };
                        originalMessages.create.__patched = true;
                    }
                }
            } catch (e) {
                // 静默失败
            }
        }
        
        return module;
    };
})();

PATCH_EOF
    echo "   ✅ Monkey Patch 已应用"
else
    echo "   ⚠️  Patch 已存在，跳过"
fi
echo ""

# 完成
echo "✨ 设置完成！"
echo ""
echo "📚 下一步操作："
echo "   1. 运行 'claude' 命令使用 Claude Code"
echo "   2. 执行各种任务，日志会自动记录到 $LOG_DIR"
echo "   3. 使用可视化工具分析日志："
echo "      在线版本: https://yuyz0112.github.io/claude-code-reverse/visualize.html"
echo "      本地版本: open $REVERSE_REPO/visualize.html"
echo ""
echo "🔧 恢复原始版本："
echo "   cp $BACKUP_FILE $CLAUDE_CLI"
echo ""
echo "📖 查看完整指南："
echo "   cat /Users/bytedance/Desktop/Codexis/docs/claude-code-reverse-guide.md"
echo ""
