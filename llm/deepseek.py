"""
DeepSeek LLM Adapter - Implementation for DeepSeek API.
"""

import json
from typing import Optional, Any
import httpx

from .base import (
    BaseLLM,
    Message,
    ToolCall,
    LLMResponse,
    ToolDefinition,
)


class DeepSeekLLM(BaseLLM):
    """
    LLM adapter for DeepSeek API.
    
    DeepSeek uses an OpenAI-compatible API format.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url, model)
        self._client: Optional[httpx.AsyncClient] = None
        self._sync_client: Optional[httpx.Client] = None
    
    def _get_headers(self) -> dict:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _build_request_body(
        self,
        messages: list[Message],
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """Build the request body for the API call."""
        # Convert messages to API format
        api_messages = []
        for msg in messages:
            msg_dict = {
                "role": msg.role.value,
                "content": msg.content,
            }
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name
            if msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    }
                    for tc in msg.tool_calls
                ]
            api_messages.append(msg_dict)
        
        body = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            body["max_tokens"] = max_tokens
        
        if tools:
            body["tools"] = [tool.to_openai_format() for tool in tools]
            body["tool_choice"] = "auto"
        
        return body
    
    def _parse_tool_arguments(self, arg_str: str) -> dict:
        """
        Parse tool arguments with robust error handling.
        
        Handles common JSON parsing issues:
        - Unescaped newlines in strings
        - Double-encoded JSON
        - Malformed JSON with recovery attempts
        """
        # First try: standard JSON parsing
        try:
            return json.loads(arg_str)
        except json.JSONDecodeError:
            pass
        
        # Second try: handle double-encoded JSON (string containing JSON string)
        try:
            if arg_str.startswith('"') and arg_str.endswith('"'):
                unescaped = json.loads(arg_str)  # Remove outer quotes
                if isinstance(unescaped, str):
                    return json.loads(unescaped)  # Parse inner JSON
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Third try: fix common issues (unescaped newlines, etc.)
        try:
            # Replace unescaped newlines with escaped ones in string values
            import re
            # Pattern: "key": "value with \n newline"
            fixed = re.sub(
                r'("(?:[^"\\]|\\.)*")\s*:\s*"([^"]*\\n[^"]*)"',
                lambda m: f'{m.group(1)}: {json.dumps(m.group(2))}',
                arg_str
            )
            # More aggressive: replace all unescaped newlines in string values
            fixed = re.sub(
                r'(?<!\\)\n(?!")',
                '\\n',
                fixed
            )
            return json.loads(fixed)
        except (json.JSONDecodeError, Exception):
            pass
        
        # Fourth try: extract key parameters using regex (fallback)
        try:
            import re
            arguments = {}
            
            # Extract path parameter (simple string value)
            path_match = re.search(r'"path"\s*:\s*"([^"]+)"', arg_str)
            if path_match:
                arguments["path"] = path_match.group(1)
            
            # Extract content parameter - handle both single-line and multi-line
            # Strategy: Find "content": and extract everything until the matching closing quote
            # This handles very long content and escaped characters
            content_pattern = r'"content"\s*:\s*"'
            content_start = re.search(content_pattern, arg_str, re.DOTALL)
            if content_start:
                start_pos = content_start.end()
                # Find the matching closing quote (accounting for escaped quotes)
                pos = start_pos
                content_value = ""
                max_length = len(arg_str) - start_pos  # Safety limit
                chars_read = 0
                
                while pos < len(arg_str) and chars_read < max_length:
                    if arg_str[pos] == '\\':
                        # Escaped character
                        if pos + 1 < len(arg_str):
                            content_value += arg_str[pos:pos+2]
                            pos += 2
                            chars_read += 2
                        else:
                            content_value += arg_str[pos]
                            pos += 1
                            chars_read += 1
                    elif arg_str[pos] == '"':
                        # Found closing quote
                        break
                    else:
                        content_value += arg_str[pos]
                        pos += 1
                        chars_read += 1
                
                # Only add if we found a closing quote or extracted something meaningful
                if content_value or pos < len(arg_str):
                    # Unescape the content properly
                    content = (content_value
                              .replace('\\n', '\n')
                              .replace('\\r', '\r')
                              .replace('\\t', '\t')
                              .replace('\\"', '"')
                              .replace('\\\\', '\\'))
                    arguments["content"] = content
            
            # Extract other common parameters
            for param in ["encoding", "append", "create_dirs", "recursive", "pattern"]:
                # Try string value first
                str_match = re.search(f'"{param}"\\s*:\\s*"([^"]+)"', arg_str)
                if str_match:
                    arguments[param] = str_match.group(1)
                    continue
                
                # Try boolean or number
                value_match = re.search(f'"{param}"\\s*:\\s*([^,}}]+)', arg_str)
                if value_match:
                    value = value_match.group(1).strip()
                    # Try to convert to appropriate type
                    if value.lower() in ("true", "false"):
                        arguments[param] = value.lower() == "true"
                    elif value.isdigit():
                        arguments[param] = int(value)
                    else:
                        arguments[param] = value.strip('"')
            
            if arguments:
                return arguments
        except Exception:
            pass
        
        # Last resort: return error info for debugging
        return {
            "_parse_error": "Failed to parse JSON arguments",
            "_raw_preview": arg_str[:200] if len(arg_str) > 200 else arg_str
        }
    
    def _parse_response(self, response_data: dict) -> LLMResponse:
        """Parse the API response into LLMResponse."""
        choice = response_data["choices"][0]
        message = choice["message"]
        
        content = message.get("content", "") or ""
        tool_calls = []
        
        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                arguments = self._parse_tool_arguments(tc["function"]["arguments"])
                
                # Log warning if parsing failed or if critical arguments are missing
                if "_parse_error" in arguments:
                    import warnings
                    warnings.warn(
                        f"Failed to parse tool arguments for {tc['function']['name']}: "
                        f"{arguments.get('_raw_preview', 'N/A')}"
                    )
                elif tc['function']['name'] == 'write_file':
                    # Debug: log if content is missing for write_file
                    if 'content' not in arguments and 'path' in arguments:
                        import warnings
                        raw_args = tc["function"]["arguments"]
                        warnings.warn(
                            f"write_file tool: extracted path but missing content. "
                            f"Raw arguments length: {len(raw_args)}, "
                            f"Preview: {raw_args[:500]}"
                        )
                
                tool_calls.append(ToolCall(
                    id=tc["id"],
                    name=tc["function"]["name"],
                    arguments=arguments,
                ))
        
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
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
        """Send async chat completion request to DeepSeek."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=300.0)
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
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
        """Send synchronous chat completion request to DeepSeek."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=300.0)
        
        url = f"{self.base_url.rstrip('/')}/chat/completions"
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
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._sync_client:
            self._sync_client.close()
