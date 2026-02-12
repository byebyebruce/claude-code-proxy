import json
import logging
import os
import sys
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import litellm
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Get default OpenAI base URL from environment
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:",
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]

        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

app = FastAPI()

# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    method = request.method
    path = request.url.path
    logger.debug(f"Request: {method} {path}")
    response = await call_next(request)
    return response

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest, model_override: str = None) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""

    messages = []

    # Add system message if present
    if anthropic_request.system:
        if isinstance(anthropic_request.system, str):
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"

            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})

    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                text_content = ""

                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""

                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"

                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"

                messages.append({"role": "user", "content": text_content.strip()})
            else:
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }

                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    processed_content_block["content"] = [{"type": "text", "text": block.content}]
                                elif isinstance(block.content, list):
                                    processed_content_block["content"] = block.content
                                else:
                                    processed_content_block["content"] = [{"type": "text", "text": str(block.content)}]
                            else:
                                processed_content_block["content"] = [{"type": "text", "text": ""}]

                            processed_content.append(processed_content_block)

                messages.append({"role": msg.role, "content": processed_content})

    # Cap max_tokens to 16384 for OpenAI models
    max_tokens = min(anthropic_request.max_tokens, 16384)
    logger.debug(f"Capping max_tokens to 16384 (original value: {anthropic_request.max_tokens})")

    # Use model_override if provided, otherwise use the original model
    model = model_override if model_override else anthropic_request.model

    # Create LiteLLM request dict
    litellm_request = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences

    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p

    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k

    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []

        for tool in anthropic_request.tools:
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                    logger.error(f"Could not convert tool to dict: {tool}")
                    continue

            input_schema = tool_dict.get("input_schema", {})

            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools

    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice

        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            litellm_request["tool_choice"] = "auto"

    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any],
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""

    try:
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }

            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")

        content = []

        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})

        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"tool_{uuid.uuid4()}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"tool_{uuid.uuid4()}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}

                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")

                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })

        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)

        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"

        if not content:
            content.append({"type": "text", "text": ""})

        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )

        return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""
        text_sent = False
        text_block_closed = False
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0

        async for chunk in response_generator:
            try:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens

                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]

                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        delta = getattr(choice, 'message', {})

                    finish_reason = getattr(choice, 'finish_reason', None)

                    delta_content = None

                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']

                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content

                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"

                    delta_tool_calls = None

                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']

                    if delta_tool_calls:
                        if tool_index is None:
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif accumulated_text and not text_sent and not text_block_closed:
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]

                        for tool_call in delta_tool_calls:
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0

                            if tool_index is None or current_index != tool_index:
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index

                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")

                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""

                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''

                            if arguments:
                                try:
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    args_json = arguments

                                tool_content += args_json if isinstance(args_json, str) else ""

                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"

                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True

                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"

                        usage = {"output_tokens": output_tokens}

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                logger.error(f"Error processing chunk: {str(e)}")
                continue

        if not has_sent_stop_reason:
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            usage = {"output_tokens": output_tokens}

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"

        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        yield "data: [DONE]\n\n"

@app.api_route("/{full_path:path}", methods=["POST"])
async def proxy_request(full_path: str, request: Request):
    """Handle all POST requests, extracting base URL from path.

    Example: /https://api.openai.com/v1/messages -> base_url=https://api.openai.com, endpoint=/v1/messages
    """
    base_url = None

    # Check for /v1/messages/count_tokens endpoint
    if full_path.endswith("/v1/messages/count_tokens"):
        base_url = full_path[:-len("/v1/messages/count_tokens")]
        if base_url:
            return await handle_token_count_request(request, base_url)
        else:
            return await handle_token_count_request(request, None)

    # Check for /v1/messages endpoint
    elif full_path.endswith("/v1/messages"):
        base_url = full_path[:-len("/v1/messages")]
        if base_url:
            return await handle_messages_request(request, base_url)
        else:
            return await handle_messages_request(request, None)

    else:
        raise HTTPException(status_code=404, detail=f"Unknown endpoint: {full_path}")

async def handle_messages_request(raw_request: Request, base_url: Optional[str] = None):
    """Handle messages request with optional base URL override."""
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))

        # Parse the request
        request = MessagesRequest(**body_json)
        original_model = body_json.get("model", "unknown")

        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        logger.debug(f"PROCESSING REQUEST: Model={request.model}, Stream={request.stream}, BaseURL={base_url}")

        # Get API key from request header (supports both x-api-key and Authorization: Bearer)
        api_key = raw_request.headers.get("x-api-key")
        if not api_key:
            auth_header = raw_request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
        if not api_key:
            raise HTTPException(status_code=401, detail="Missing API key (x-api-key or Authorization: Bearer)")

        # Use openai/ prefix for the model
        litellm_model = f"openai/{request.model}"

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request, model_override=litellm_model)

        # Set API key from request header
        litellm_request["api_key"] = api_key

        # Set base URL (from path or environment variable)
        effective_base_url = base_url or OPENAI_BASE_URL
        if effective_base_url:
            litellm_request["api_base"] = effective_base_url
            logger.debug(f"Using base URL: {effective_base_url}")

        # Process OpenAI model request format
        if "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")

            for i, msg in enumerate(litellm_request["messages"]):
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break

                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])

                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"

                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        continue

                if "content" in msg:
                    if isinstance(msg["content"], list):
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"

                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"

                        if not text_content.strip():
                            text_content = "..."

                        litellm_request["messages"][i]["content"] = text_content.strip()
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..."

                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]

            for i, msg in enumerate(litellm_request["messages"]):
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")

                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..."

        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        if request.stream:
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )
            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")

            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

            return anthropic_response

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()

        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }

        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)

        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)

        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")

        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"

        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

async def handle_token_count_request(raw_request: Request, base_url: Optional[str] = None):
    """Handle token count request with optional base URL override."""
    try:
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))

        request = TokenCountRequest(**body_json)
        original_model = request.model

        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        litellm_model = f"openai/{request.model}"

        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            ),
            model_override=litellm_model
        )

        try:
            from litellm import token_counter

            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200
            )

            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }

            # Set base URL (from path or environment variable)
            effective_base_url = base_url or OPENAI_BASE_URL
            if effective_base_url:
                token_counter_args["api_base"] = effective_base_url

            token_count = token_counter(**token_counter_args)

            return TokenCountResponse(input_tokens=token_count)

        except ImportError:
            logger.error("Could not import token_counter from litellm")
            return TokenCountResponse(input_tokens=1000)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "OpenAI Proxy - Use /{base_url}/v1/messages, e.g. /https://api.openai.com/v1/messages"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful format."""
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"

    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"

    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"

    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        print("\nUsage:")
        print("  POST /{base_url}/v1/messages - Send messages to the specified OpenAI-compatible API")
        print("  Example: POST /https://api.openai.com/v1/messages")
        sys.exit(0)

    print(f"{Colors.GREEN}Server starting on http://0.0.0.0:8082{Colors.RESET}")
    print(f"{Colors.DIM}Usage: POST /https://api.openai.com/v1/messages{Colors.RESET}")
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
