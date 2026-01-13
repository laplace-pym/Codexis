"""
Anthropic LLM Adapter - Implementation for Anthropic Claude API.
"""

import json
from typing import Optional
import httpx

from .base import (
    BaseLLM,
    Message,
    MessageRole,
    ToolCall,
    LLMResponse,
    ToolDefinition,
)


class AnthropicLLM(BaseLLM):
    """
    LLM adapter for Anthropic Claude API.
    
    Note: Anthropic API has a different format than OpenAI.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url, model)
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
    
    def _build_request_body(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build the request body for Anthropic API."""
        # Extract system message
        system_content = ""
        api_messages = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            elif msg.role == MessageRole.USER:
                api_messages.append({
                    "role": "user",
                    "content": msg.content,
                })
            elif msg.role == MessageRole.ASSISTANT:
                content = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                api_messages.append({
                    "role": "assistant",
                    "content": content if content else msg.content,
                })
            elif msg.role == MessageRole.TOOL:
                api_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content,
                    }],
                })
        
        body = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system_content:
            body["system"] = system_content
        
        if tools:
            body["tools"] = [tool.to_anthropic_format() for tool in tools]
        
        return body
    
    def _parse_response(self, response_data: dict) -> LLMResponse:
        """Parse the Anthropic API response into LLMResponse."""
        content_blocks = response_data.get("content", [])
        
        text_content = ""
        tool_calls = []
        
        for block in content_blocks:
            if block["type"] == "text":
                text_content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["input"],
                ))
        
        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=response_data.get("stop_reason", "end_turn"),
            usage=response_data.get("usage"),
            raw_response=response_data,
        )
    
    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send async chat completion request to Anthropic."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        body = self._build_request_body(messages, tools, temperature, max_tokens)
        
        response = await self._client.post(
            url,
            headers=self._get_headers(),
            json=body,
        )
        response.raise_for_status()
        
        return self._parse_response(response.json())
    
    def chat_sync(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Send synchronous chat completion request to Anthropic."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=120.0)
        
        url = f"{self.base_url.rstrip('/')}/v1/messages"
        body = self._build_request_body(messages, tools, temperature, max_tokens)
        
        response = self._sync_client.post(
            url,
            headers=self._get_headers(),
            json=body,
        )
        response.raise_for_status()
        
        return self._parse_response(response.json())
    
    async def close(self):
        """Close the HTTP clients."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
