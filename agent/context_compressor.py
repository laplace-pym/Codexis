"""
Context Compressor - Structured auto-compression for long context governance.

Implements the AU2 algorithm: 92% trigger threshold + 8-segment structured summary.
When conversation token usage exceeds the configured threshold (default 92%),
the compressor intelligently compresses the conversation history into 8 structured
dimensions, preserving critical context while dramatically reducing token count.

Three-layer memory architecture:
  - Short-term: messages[] (current conversation)
  - Mid-term: compressed summaries (AU2 structured compression)
  - Long-term: persistent project memory (CLAUDE.md / config)
"""

import json
from typing import Optional
from dataclasses import dataclass, field

from llm.base import BaseLLM, Message, LLMResponse
from utils.logger import get_logger


# ──────────────────────────────────────────────────────────────────
# Default context window sizes per provider/model family
# ──────────────────────────────────────────────────────────────────
DEFAULT_MAX_CONTEXT_TOKENS = {
    "deepseek": 64_000,
    "openai": 128_000,
    "anthropic": 200_000,
}

# Default fallback
DEFAULT_CONTEXT_WINDOW = 64_000

# Default trigger threshold (92%)
DEFAULT_COMPRESSION_THRESHOLD = 0.92

# Approximate characters per token (for estimation when usage data unavailable)
CHARS_PER_TOKEN = 4

# Number of recent messages to preserve uncompressed
PRESERVE_RECENT_MESSAGES = 4


# ──────────────────────────────────────────────────────────────────
# AU2 8-Segment Compression Prompt
# ──────────────────────────────────────────────────────────────────
COMPRESSION_PROMPT = """你是一个对话压缩专家。请按照以下 8 个结构化维度，对给定的对话历史进行高保真压缩摘要。
要求：保留所有关键技术细节、代码路径、决策理由，去除冗余寒暄和重复内容。

## 1. 背景上下文 (Background Context)
- 项目类型和技术栈
- 当前工作目标和环境
- 用户的总体目标

## 2. 关键决策 (Key Decisions)
- 重要的技术选择和原因
- 架构决策和设计考虑
- 问题解决方案的选择

## 3. 工具使用记录 (Tool Usage Log)
- 主要使用的工具类型和调用情况
- 文件操作历史（创建/修改/读取的文件路径）
- 命令执行结果摘要

## 4. 用户意图演进 (User Intent Evolution)
- 需求的变化过程
- 优先级调整
- 新增功能需求

## 5. 执行结果汇总 (Execution Results)
- 成功完成的任务
- 生成的代码和文件清单
- 验证和测试结果

## 6. 错误与解决 (Errors and Solutions)
- 遇到的问题类型
- 错误解决方法
- 经验教训

## 7. 未解决问题 (Open Issues)
- 当前待解决的挑战
- 已知但未处理的问题
- 需要后续关注的事项

## 8. 后续计划 (Future Plans)
- 下一步行动计划
- 长期目标和规划
- 用户表达的期望

请将以上信息压缩到 {max_tokens} 个 Token 以内，保持技术准确性和上下文连续性。
直接输出压缩后的结构化摘要，不要包含任何前缀说明。

---

以下是需要压缩的对话历史：

{conversation}"""


@dataclass
class CompressionMetrics:
    """Metrics for a compression operation."""
    original_messages: int = 0
    compressed_messages: int = 0
    estimated_original_tokens: int = 0
    estimated_compressed_tokens: int = 0
    compression_ratio: float = 0.0
    compressions_performed: int = 0

    def to_dict(self) -> dict:
        return {
            "original_messages": self.original_messages,
            "compressed_messages": self.compressed_messages,
            "estimated_original_tokens": self.estimated_original_tokens,
            "estimated_compressed_tokens": self.estimated_compressed_tokens,
            "compression_ratio": self.compression_ratio,
            "compressions_performed": self.compressions_performed,
        }


class ContextCompressor:
    """
    Structured auto-compression for long context governance.

    Monitors token usage during agent execution and triggers AU2 8-segment
    compression when usage exceeds the configurable threshold (default 92%).

    Usage:
        compressor = ContextCompressor(llm=my_llm, provider="deepseek")

        # In the execution loop, before each LLM call:
        messages = compressor.maybe_compress(messages)
    """

    def __init__(
        self,
        llm: BaseLLM,
        provider: Optional[str] = None,
        max_context_tokens: Optional[int] = None,
        threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        preserve_recent: int = PRESERVE_RECENT_MESSAGES,
    ):
        """
        Args:
            llm: The LLM instance used for generating compression summaries.
            provider: Provider name (deepseek/openai/anthropic) for default window size.
            max_context_tokens: Max context window size in tokens. Auto-detected if None.
            threshold: Trigger threshold as ratio (0-1). Default 0.92 (92%).
            preserve_recent: Number of recent messages to keep uncompressed.
        """
        self.llm = llm
        self.threshold = threshold
        self.preserve_recent = preserve_recent
        self.logger = get_logger()

        # Determine max context window
        if max_context_tokens:
            self.max_context_tokens = max_context_tokens
        elif provider and provider in DEFAULT_MAX_CONTEXT_TOKENS:
            self.max_context_tokens = DEFAULT_MAX_CONTEXT_TOKENS[provider]
        else:
            self.max_context_tokens = DEFAULT_CONTEXT_WINDOW

        self.trigger_tokens = int(self.max_context_tokens * self.threshold)

        # Compression target: aim for ~50% of max after compression
        self.target_tokens = int(self.max_context_tokens * 0.5)

        # Metrics tracking
        self.metrics = CompressionMetrics()

        # Track last known token usage from LLM response
        self._last_usage: Optional[dict] = None

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def update_usage(self, usage: Optional[dict]) -> None:
        """Update token usage from the latest LLM response."""
        if usage:
            self._last_usage = usage

    def should_compress(self, messages: list[Message]) -> bool:
        """
        Check if compression should be triggered.

        Uses real token usage from LLM if available, falls back to estimation.
        """
        token_count = self._get_token_count(messages)
        return token_count >= self.trigger_tokens

    def maybe_compress(self, messages: list[Message]) -> list[Message]:
        """
        Compress messages if the threshold is exceeded.

        This is the main entry point — call this before each LLM invocation
        in the execution loop.

        Returns:
            The (possibly compressed) message list.
        """
        if not self.should_compress(messages):
            return messages

        self.logger.info(
            f"🗜️ Context compression triggered "
            f"(~{self._get_token_count(messages):,} tokens, "
            f"threshold: {self.trigger_tokens:,})"
        )

        try:
            compressed = self._compress(messages)
            return compressed
        except Exception as e:
            self.logger.warning(f"Compression failed, keeping original messages: {e}")
            return messages

    # ──────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────

    def _get_token_count(self, messages: list[Message]) -> int:
        """
        Get the current token count.

        Prefers real usage data from the LLM response (prompt_tokens).
        Falls back to character-based estimation.
        """
        # Try real usage data first
        if self._last_usage:
            prompt_tokens = self._last_usage.get("prompt_tokens")
            if prompt_tokens:
                return int(prompt_tokens)
            # Some providers use total_tokens
            total_tokens = self._last_usage.get("total_tokens")
            if total_tokens:
                return int(total_tokens)

        # Fallback: estimate from message content
        return self._estimate_tokens(messages)

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count from message content length."""
        total_chars = sum(len(msg.content or "") for msg in messages)
        return total_chars // CHARS_PER_TOKEN

    def _compress(self, messages: list[Message]) -> list[Message]:
        """
        Execute AU2 8-segment structured compression.

        Strategy:
        1. Keep system message (index 0) intact
        2. Compress middle messages into a structured summary
        3. Keep the N most recent messages intact for continuity

        Returns:
            New message list: [system, compressed_summary, ...recent_messages]
        """
        if len(messages) <= self.preserve_recent + 2:
            # Too few messages to compress meaningfully
            return messages

        # Split messages
        system_msg = messages[0] if messages[0].role.value == "system" else None
        start_idx = 1 if system_msg else 0

        # Messages to compress (middle section)
        compress_end = len(messages) - self.preserve_recent
        messages_to_compress = messages[start_idx:compress_end]

        # Recent messages to preserve
        recent_messages = messages[compress_end:]

        if not messages_to_compress:
            return messages

        # Build conversation text for compression
        conversation_text = self._format_messages_for_compression(messages_to_compress)

        # Generate compression prompt
        prompt = COMPRESSION_PROMPT.format(
            max_tokens=self.target_tokens,
            conversation=conversation_text,
        )

        # Call LLM to compress
        compression_messages = [
            Message.system("你是一个专业的对话压缩助手，擅长结构化信息提取和摘要。"),
            Message.user(prompt),
        ]

        response = self.llm.chat_sync(
            messages=compression_messages,
            temperature=0.3,  # Lower temperature for more focused compression
            max_tokens=self.target_tokens,
        )

        if not response.content:
            self.logger.warning("Compression returned empty result, keeping originals")
            return messages

        # Build compressed message list
        compressed = []

        if system_msg:
            compressed.append(system_msg)

        # Insert compressed summary as a system-injected context message
        compressed.append(Message.user(
            f"[Context Compression — 以下是之前对话的结构化摘要]\n\n{response.content}"
        ))
        compressed.append(Message.assistant(
            "理解，我已掌握之前的对话上下文。请继续。"
        ))

        # Append recent messages
        compressed.extend(recent_messages)

        # Update metrics
        original_tokens = self._estimate_tokens(messages)
        compressed_tokens = self._estimate_tokens(compressed)

        self.metrics.original_messages = len(messages)
        self.metrics.compressed_messages = len(compressed)
        self.metrics.estimated_original_tokens = original_tokens
        self.metrics.estimated_compressed_tokens = compressed_tokens
        self.metrics.compression_ratio = (
            compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        )
        self.metrics.compressions_performed += 1

        # Reset usage tracking after compression
        self._last_usage = None

        self.logger.info(
            f"✅ Compression complete: "
            f"{len(messages)} → {len(compressed)} messages, "
            f"~{original_tokens:,} → ~{compressed_tokens:,} tokens "
            f"(ratio: {self.metrics.compression_ratio:.1%})"
        )

        return compressed

    def _format_messages_for_compression(self, messages: list[Message]) -> str:
        """Format messages into a readable text for the compression LLM."""
        parts = []
        for msg in messages:
            role = msg.role.value.upper()
            content = msg.content or ""

            # Truncate very long tool results to save compression input tokens
            if msg.role.value == "tool" and len(content) > 500:
                content = content[:500] + "\n... [truncated]"

            parts.append(f"[{role}]: {content}")

        return "\n\n".join(parts)
