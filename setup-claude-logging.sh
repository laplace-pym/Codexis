#!/bin/bash

# Claude Code 日志捕获脚本 - 使用 mitmproxy
# 这是目前最可靠的方法来查看 Claude Code 的 API 交互

set -e

echo "🔍 Claude Code 日志捕获工具"
echo "================================"
echo ""

# 步骤 1: 安装 mitmproxy
echo "📦 步骤 1: 检查并安装 mitmproxy..."
if ! command -v mitmproxy &> /dev/null; then
    echo "   正在安装 mitmproxy（可能需要几分钟）..."
    brew install mitmproxy
    echo "   ✅ mitmproxy 安装完成"
else
    echo "   ✅ mitmproxy 已安装"
fi
echo ""

# 步骤 2: 创建 mitmproxy 脚本来保存日志
LOG_DIR="$HOME/.claude-reverse-logs"
mkdir -p "$LOG_DIR"

MITM_SCRIPT="$LOG_DIR/mitm_logger.py"

cat > "$MITM_SCRIPT" << 'PYTHON_SCRIPT'
import json
import time
from mitmproxy import http

class AnthropicLogger:
    def __init__(self):
        self.log_file = f"{time.time():.0f}-claude-api.log"
        self.log_path = f"$HOME/.claude-reverse-logs/{self.log_file}"
        print(f"📝 日志文件: {self.log_path}")
    
    def request(self, flow: http.HTTPFlow) -> None:
        # 只记录 Anthropic API 请求
        if "anthropic.com" in flow.request.pretty_host or "claude" in flow.request.pretty_host:
            try:
                request_data = {
                    "type": "request",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": flow.request.method,
                    "url": flow.request.pretty_url,
                    "headers": dict(flow.request.headers),
                }
                
                # 尝试解析 JSON body
                if flow.request.content:
                    try:
                        request_data["body"] = json.loads(flow.request.content.decode('utf-8'))
                    except:
                        request_data["body"] = flow.request.content.decode('utf-8', errors='ignore')
                
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(request_data, indent=2, ensure_ascii=False))
                    f.write("\n" + "="*80 + "\n\n")
                    
                print(f"📤 请求: {flow.request.method} {flow.request.path}")
            except Exception as e:
                print(f"❌ 记录请求错误: {e}")
    
    def response(self, flow: http.HTTPFlow) -> None:
        # 只记录 Anthropic API 响应
        if "anthropic.com" in flow.request.pretty_host or "claude" in flow.request.pretty_host:
            try:
                response_data = {
                    "type": "response",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": flow.response.status_code,
                    "headers": dict(flow.response.headers),
                }
                
                # 尝试解析 JSON body
                if flow.response.content:
                    try:
                        response_data["body"] = json.loads(flow.response.content.decode('utf-8'))
                    except:
                        response_data["body"] = flow.response.content.decode('utf-8', errors='ignore')
                
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(response_data, indent=2, ensure_ascii=False))
                    f.write("\n" + "="*80 + "\n\n")
                    
                print(f"📥 响应: {flow.response.status_code}")
            except Exception as e:
                print(f"❌ 记录响应错误: {e}")

addons = [AnthropicLogger()]
PYTHON_SCRIPT

# 替换 $HOME 变量
sed -i '' "s|\$HOME|$HOME|g" "$MITM_SCRIPT"

echo "✅ mitmproxy 日志脚本已创建: $MITM_SCRIPT"
echo ""

# 步骤 3: 创建启动脚本
LAUNCH_SCRIPT="$HOME/start-claude-logging.sh"

cat > "$LAUNCH_SCRIPT" << 'EOF'
#!/bin/bash

LOG_DIR="$HOME/.claude-reverse-logs"
MITM_SCRIPT="$LOG_DIR/mitm_logger.py"

echo "🚀 启动 Claude Code 日志捕获"
echo "================================"
echo ""
echo "📍 日志保存位置: $LOG_DIR"
echo ""
echo "⚠️  重要提示："
echo "   1. 这个窗口会显示所有拦截的请求"
echo "   2. 保持这个窗口运行"
echo "   3. 在另一个终端运行 Claude Code"
echo ""
echo "按 Ctrl+C 停止捕获"
echo ""

# 启动 mitmproxy
mitmproxy --listen-host 127.0.0.1 --listen-port 8080 -s "$MITM_SCRIPT" --ssl-insecure
EOF

chmod +x "$LAUNCH_SCRIPT"

echo "✅ 启动脚本已创建: $LAUNCH_SCRIPT"
echo ""

# 步骤 4: 创建 Claude 包装脚本
CLAUDE_WRAPPER="$HOME/claude-logged"

cat > "$CLAUDE_WRAPPER" << 'EOF'
#!/bin/bash

# 配置代理
export HTTP_PROXY=http://127.0.0.1:8080
export HTTPS_PROXY=http://127.0.0.1:8080
export NODE_TLS_REJECT_UNAUTHORIZED=0

echo "🔍 Claude Code (带日志记录)"
echo "代理: http://127.0.0.1:8080"
echo ""

# 运行 Claude
claude "$@"
EOF

chmod +x "$CLAUDE_WRAPPER"

echo "✅ Claude 包装脚本已创建: $CLAUDE_WRAPPER"
echo ""

# 完成
echo "================================"
echo "✨ 设置完成！"
echo ""
echo "📚 使用方法："
echo ""
echo "1️⃣  在终端 1 中启动日志捕获："
echo "   $LAUNCH_SCRIPT"
echo ""
echo "2️⃣  在终端 2 中运行 Claude Code："
echo "   $CLAUDE_WRAPPER"
echo "   或者手动设置环境变量："
echo "   export HTTPS_PROXY=http://127.0.0.1:8080"
echo "   export NODE_TLS_REJECT_UNAUTHORIZED=0"
echo "   claude"
echo ""
echo "3️⃣  查看生成的日志："
echo "   ls -lh $LOG_DIR"
echo ""
echo "4️⃣  分析日志（使用在线工具）："
echo "   https://yuyz0112.github.io/claude-code-reverse/visualize.html"
echo ""

# 创建快速启动别名
echo ""
echo "💡 提示：可以添加以下别名到 ~/.zshrc 或 ~/.bashrc："
echo ""
echo "   alias claude-start-log='$LAUNCH_SCRIPT'"
echo "   alias claude-with-log='$CLAUDE_WRAPPER'"
echo ""
