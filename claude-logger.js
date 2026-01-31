#!/usr/bin/env node

/**
 * Claude Code API 日志记录代理
 * 使用方法：将此文件放在 PATH 中 claude 命令之前
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// 日志配置
const LOG_DIR = path.join(process.env.HOME, '.claude-reverse-logs');
const LOG_FILE = path.join(LOG_DIR, `messages-${Date.now()}.log`);

// 确保日志目录存在
if (!fs.existsSync(LOG_DIR)) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
}

console.log(`\n🔍 Claude 逆向日志记录器已启动`);
console.log(`📝 日志文件: ${LOG_FILE}\n`);

// 写入日志
function writeLog(data) {
    fs.appendFileSync(LOG_FILE, JSON.stringify(data, null, 2) + '\n---\n\n');
}

// Hook fetch 和 HTTP 请求
const Module = require('module');
const originalRequire = Module.prototype.require;

Module.prototype.require = function(id) {
    const module = originalRequire.apply(this, arguments);
    
    // 拦截 https 模块
    if (id === 'https' || id === 'http') {
        const originalRequest = module.request;
        
        module.request = function(...args) {
            const req = originalRequest.apply(this, args);
            const originalWrite = req.write;
            const originalEnd = req.end;
            
            let body = '';
            
            req.write = function(chunk, ...writeArgs) {
                if (chunk) body += chunk.toString();
                return originalWrite.call(this, chunk, ...writeArgs);
            };
            
            req.end = function(chunk, ...endArgs) {
                if (chunk) body += chunk.toString();
                
                // 记录请求
                try {
                    if (body && body.includes('anthropic')) {
                        const data = JSON.parse(body);
                        writeLog({
                            type: 'request',
                            timestamp: new Date().toISOString(),
                            data: data
                        });
                    }
                } catch (e) {
                    // 忽略解析错误
                }
                
                return originalEnd.call(this, chunk, ...endArgs);
            };
            
            // 拦截响应
            req.on('response', (res) => {
                let responseBody = '';
                res.on('data', (chunk) => {
                    responseBody += chunk.toString();
                });
                res.on('end', () => {
                    try {
                        if (responseBody) {
                            const data = JSON.parse(responseBody);
                            writeLog({
                                type: 'response',
                                timestamp: new Date().toISOString(),
                                data: data
                            });
                        }
                    } catch (e) {
                        // 忽略解析错误
                    }
                });
            });
            
            return req;
        };
    }
    
    // 拦截 fetch（Node 18+）
    if (id === 'node-fetch' || id === 'undici') {
        const originalFetch = module.fetch || module;
        
        if (typeof originalFetch === 'function') {
            module.fetch = async function(...args) {
                const [url, options] = args;
                
                // 记录请求
                if (options && options.body) {
                    try {
                        const body = typeof options.body === 'string' 
                            ? JSON.parse(options.body) 
                            : options.body;
                        
                        writeLog({
                            type: 'request',
                            timestamp: new Date().toISOString(),
                            url: url,
                            data: body
                        });
                    } catch (e) {}
                }
                
                const response = await originalFetch.apply(this, args);
                
                // 拦截响应
                const originalJson = response.json;
                response.json = async function() {
                    const data = await originalJson.call(this);
                    writeLog({
                        type: 'response',
                        timestamp: new Date().toISOString(),
                        data: data
                    });
                    return data;
                };
                
                return response;
            };
        }
    }
    
    return module;
};

// 运行原始的 Claude CLI
const CLAUDE_PATH = '/Users/bytedance/.nvm/versions/node/v24.13.0/lib/node_modules/@anthropic-ai/claude-code/cli.js';

require(CLAUDE_PATH);
