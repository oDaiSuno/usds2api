import json
import logging
import os
import random
import re
import secrets
import string
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError


class UpstreamService(BaseModel):
    name: str
    base_url: str
    api_key: Optional[str] = None
    timeout: Optional[float] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)


class FeatureConfig(BaseModel):
    enable_function_calling: bool = True
    convert_developer_to_system: bool = True
    log_level: str = "INFO"
    prompt_template: Optional[str] = None
    model_passthrough: bool = False
    passthrough_service: Optional[str] = None
    trigger_signal: Optional[str] = None
    random_trigger: bool = True
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600
    cache_cleanup_interval: int = 300


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8021
    timeout: float = 180.0


class AppConfig(BaseModel):
    upstream_services: List[UpstreamService]
    client_keys: List[str] = Field(default_factory=list)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _env_truthy(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_extra_headers(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _build_config_from_env() -> Optional[AppConfig]:
    upstream_url = os.getenv("FORMAT_PROXY_UPSTREAM_URL")
    if not upstream_url:
        return None

    upstream = UpstreamService(
        name=os.getenv("FORMAT_PROXY_UPSTREAM_NAME", "ustcds"),
        base_url=upstream_url,
        api_key=os.getenv("FORMAT_PROXY_UPSTREAM_API_KEY") or None,
        timeout=_env_optional_float("FORMAT_PROXY_UPSTREAM_TIMEOUT"),
        extra_headers=_parse_extra_headers(os.getenv("FORMAT_PROXY_UPSTREAM_EXTRA_HEADERS_JSON")),
    )

    server_defaults = ServerConfig()
    server = ServerConfig(
        host=os.getenv("FORMAT_PROXY_SERVER_HOST", server_defaults.host),
        port=_env_int("FORMAT_PROXY_SERVER_PORT", server_defaults.port),
        timeout=_env_float("FORMAT_PROXY_SERVER_TIMEOUT", server_defaults.timeout),
    )

    features = FeatureConfig()
    features.enable_function_calling = _env_truthy(
        "FORMAT_PROXY_FEATURE_ENABLE_FUNCTION_CALLING", features.enable_function_calling
    )
    features.convert_developer_to_system = _env_truthy(
        "FORMAT_PROXY_FEATURE_CONVERT_DEVELOPER_TO_SYSTEM", features.convert_developer_to_system
    )
    features.log_level = os.getenv("FORMAT_PROXY_FEATURE_LOG_LEVEL", features.log_level)
    features.prompt_template = os.getenv("FORMAT_PROXY_FEATURE_PROMPT_TEMPLATE", features.prompt_template)
    features.model_passthrough = _env_truthy(
        "FORMAT_PROXY_FEATURE_MODEL_PASSTHROUGH", features.model_passthrough
    )
    features.passthrough_service = os.getenv(
        "FORMAT_PROXY_FEATURE_PASSTHROUGH_SERVICE", features.passthrough_service
    )
    features.trigger_signal = os.getenv("FORMAT_PROXY_FEATURE_TRIGGER_SIGNAL", features.trigger_signal)
    features.random_trigger = _env_truthy(
        "FORMAT_PROXY_FEATURE_RANDOM_TRIGGER", features.random_trigger
    )
    features.cache_max_size = _env_int(
        "FORMAT_PROXY_FEATURE_CACHE_MAX_SIZE", features.cache_max_size
    )
    features.cache_ttl_seconds = _env_int(
        "FORMAT_PROXY_FEATURE_CACHE_TTL_SECONDS", features.cache_ttl_seconds
    )
    features.cache_cleanup_interval = _env_int(
        "FORMAT_PROXY_FEATURE_CACHE_CLEANUP_INTERVAL", features.cache_cleanup_interval
    )

    client_keys_raw = os.getenv("FORMAT_PROXY_CLIENT_KEYS", "")
    client_keys = [key.strip() for key in client_keys_raw.split(",") if key.strip()]

    return AppConfig(
        upstream_services=[upstream],
        client_keys=client_keys,
        features=features,
        server=server,
    )


def load_or_bootstrap_config() -> AppConfig:
    env_config = _build_config_from_env()
    if env_config:
        return env_config
    if not os.path.exists(CONFIG_PATH):
        skeleton = {
            "upstream_services": [
                {
                    "name": "ustcds",
                    "base_url": "http://localhost:8010/v1",
                    "api_key": "REPLACE_WITH_UPSTREAM_KEY"
                }
            ],
            "client_keys": [],
            "features": {"enable_function_calling": True}
        }
        with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
            json.dump(skeleton, fh, indent=4)
    raw = _load_json_file(CONFIG_PATH)
    return AppConfig(**raw)


def generate_random_trigger_signal() -> str:
    charset = string.ascii_letters + string.digits
    token = "".join(secrets.choice(charset) for _ in range(4))
    return f"<Function_{token}_Start/>"


class ToolCallMappingManager:
    def __init__(self, max_size: int, ttl_seconds: int, cleanup_interval: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        def _worker() -> None:
            while True:
                time.sleep(self.cleanup_interval)
                self.cleanup_expired()
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def store(self, tool_call_id: str, name: str, args: dict, description: str = "") -> None:
        with self._lock:
            now = time.time()
            if tool_call_id in self._data:
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
            while len(self._data) >= self.max_size:
                oldest_key = next(iter(self._data))
                del self._data[oldest_key]
                del self._timestamps[oldest_key]
            self._data[tool_call_id] = {
                "name": name,
                "args": args,
                "description": description,
                "created_at": now
            }
            self._timestamps[tool_call_id] = now

    def get(self, tool_call_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            if tool_call_id not in self._data:
                return None
            if now - self._timestamps[tool_call_id] > self.ttl_seconds:
                del self._data[tool_call_id]
                del self._timestamps[tool_call_id]
                return None
            self._data.move_to_end(tool_call_id)
            return self._data[tool_call_id]

    def cleanup_expired(self) -> None:
        with self._lock:
            now = time.time()
            expired = [key for key, ts in self._timestamps.items() if now - ts > self.ttl_seconds]
            for key in expired:
                del self._data[key]
                del self._timestamps[key]


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: Literal["function"]
    function: ToolFunction


class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ToolChoice(BaseModel):
    type: Literal["function"]
    function: Dict[str, str]


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None

    class Config:
        extra = "allow"


def _ensure_bearer(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return token
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


app_config = load_or_bootstrap_config()

log_level = app_config.features.log_level.upper()
if log_level == "DISABLED":
    logging.basicConfig(level=logging.CRITICAL + 1)
else:
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger("format_proxy")

GLOBAL_TRIGGER_SIGNAL = (
    app_config.features.trigger_signal
    if app_config.features.trigger_signal
    else generate_random_trigger_signal() if app_config.features.random_trigger else "<Function_Call_Start/>"
)

TOOL_CALL_MANAGER = ToolCallMappingManager(
    max_size=app_config.features.cache_max_size,
    ttl_seconds=app_config.features.cache_ttl_seconds,
    cleanup_interval=app_config.features.cache_cleanup_interval,
)

http_client: Optional[httpx.AsyncClient] = None
app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    global http_client
    timeout = httpx.Timeout(app_config.server.timeout)
    http_client = httpx.AsyncClient(timeout=timeout)
    logger.info("format_proxy_enhanced ready")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None


def store_tool_call_mapping(tool_call_id: str, name: str, args: dict, description: str = "") -> None:
    TOOL_CALL_MANAGER.store(tool_call_id, name, args, description)


def get_tool_call_mapping(tool_call_id: str) -> Optional[Dict[str, Any]]:
    return TOOL_CALL_MANAGER.get(tool_call_id)


def format_tool_result_for_ai(tool_call_id: str, result_content: str) -> str:
    info = get_tool_call_mapping(tool_call_id)
    if not info:
        return f"Tool execution result:\n<tool_result>\n{result_content}\n</tool_result>"
    return (
        "Tool execution result:\n"
        f"- Tool name: {info['name']}\n"
        "- Execution result:\n"
        "<tool_result>\n"
        f"{result_content}\n"
        "</tool_result>"
    )


def format_assistant_tool_calls_for_ai(tool_calls: List[Dict[str, Any]], trigger_signal: str) -> str:
    xml_blocks: List[str] = []
    for call in tool_calls:
        func = call.get("function", {})
        name = func.get("name", "")
        try:
            args_dict = json.loads(func.get("arguments", "{}"))
        except Exception:
            args_dict = {"raw_arguments": func.get("arguments")}
        arg_lines = []
        for key, value in args_dict.items():
            arg_lines.append(f"<{key}>{json.dumps(value, ensure_ascii=False)}</{key}>")
        args_joined = "\n".join(arg_lines)
        block = (
            "<function_call>\n"
            f"<tool>{name}</tool>\n"
            "<args>\n"
            f"{args_joined}\n"
            "</args>\n"
            "</function_call>"
        )
        xml_blocks.append(block)
    wrapped = "\n".join(xml_blocks)
    return f"{trigger_signal}\n<function_calls>\n{wrapped}\n</function_calls>"




def build_tool_call_records(parsed_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for item in parsed_tools:
        call_id = f"call_{uuid.uuid4().hex}"
        store_tool_call_mapping(call_id, item["name"], item["args"], f"Calling {item['name']}")
        records.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": item["name"], "arguments": json.dumps(item["args"])}
            }
        )
    return records


def build_streaming_text_chunk(content: str, model: str, payload_json: Optional[Dict[str, Any]] = None) -> str:
    base_payload = payload_json or {}
    chunk = {
        "id": base_payload.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        "object": "chat.completion.chunk",
        "created": base_payload.get("created", int(time.time())),
        "model": base_payload.get("model", model),
        "choices": [{"index": 0, "delta": {"content": content}}]
    }
    return f"data: {json.dumps(chunk)}\n\n"

def get_function_call_prompt_template(trigger_signal: str) -> str:
    custom = app_config.features.prompt_template
    if custom:
        return custom.format(trigger_signal=trigger_signal, tools_list="{tools_list}")
    return (
        "You have access to the following tools:\n\n{tools_list}\n\n"
        "IMPORTANT: Emit tool calls only in XML.\n"
        f"Trigger with \n{trigger_signal}\n"
        "The XML body must contain <function_calls> with one or more <function_call> entries."
    )


def generate_function_prompt(tools: List[Tool], trigger_signal: str) -> Tuple[str, str]:
    parts: List[str] = []
    for idx, tool in enumerate(tools, 1):
        func = tool.function
        schema = func.parameters or {}
        props: Dict[str, Any] = schema.get("properties", {}) or {}
        required: List[str] = schema.get("required", []) or []
        prop_lines: List[str] = []
        for key, meta in props.items():
            meta = meta or {}
            p_type = meta.get("type", "any")
            desc = meta.get("description")
            required_flag = "Yes" if key in required else "No"
            line = [f"- {key}: {p_type}"]
            line.append(f"  required: {required_flag}")
            if desc:
                line.append(f"  description: {desc}")
            enums = meta.get("enum")
            if enums is not None:
                line.append(f"  enum: {json.dumps(enums, ensure_ascii=False)}")
            prop_lines.append("\n".join(line))
        body = "\n".join(prop_lines) or "(no parameters)"
        desc = func.description or "None"
        parts.append(
            f"{idx}. {func.name}\nDescription: {desc}\nRequired: {', '.join(required) if required else 'None'}\n"
            f"Parameters:\n{body}"
        )
    template = get_function_call_prompt_template(trigger_signal)
    return template.replace("{tools_list}", "\n\n".join(parts)), trigger_signal


def remove_think_blocks(text: str) -> str:
    work = text
    while "<think>" in work and "</think>" in work:
        start = work.find("<think>")
        if start < 0:
            break
        depth = 1
        pos = start + 7
        while pos < len(work) and depth > 0:
            if work.startswith("<think>", pos):
                depth += 1
                pos += 7
            elif work.startswith("</think>", pos):
                depth -= 1
                pos += 8
            else:
                pos += 1
        if depth == 0:
            work = work[:start] + work[pos:]
        else:
            break
    return work


class StreamingFunctionCallDetector:
    def __init__(self, trigger_signal: str):
        self.trigger_signal = trigger_signal
        self.reset()

    def reset(self) -> None:
        self.content_buffer = ""
        self.state = "detecting"
        self.in_think_block = False
        self.think_depth = 0
        self.signal = self.trigger_signal
        self.signal_len = len(self.signal)
        self.fc_tag = "<function_calls"
        self.fc_tag_len = len(self.fc_tag)
        # 动态trigger signal匹配模式
        self.dynamic_trigger_pattern = re.compile(r"<Function_[a-zA-Z0-9]{4}_Start/>", re.IGNORECASE)

    def process_chunk(self, delta_content: str) -> Tuple[bool, str]:
        if not delta_content:
            return False, ""

        # 如果已经在解析工具调用状态，继续积累内容
        if self.state == "tool_parsing":
            self.content_buffer += delta_content
            return False, ""

        self.content_buffer += delta_content
        yielded = ""
        idx = 0
        while idx < len(self.content_buffer):
            consumed = self._update_think_state(idx)
            if consumed:
                yielded += self.content_buffer[idx:idx+consumed]
                idx += consumed
                continue
            if not self.in_think_block:
                # 检测固定的trigger signal
                if self.signal_len and self._can_detect(idx) and self.content_buffer[idx:idx + self.signal_len] == self.signal:
                    self.state = "tool_parsing"
                    # 跳过trigger signal，从trigger signal之后开始
                    trigger_end = idx + self.signal_len
                    self.content_buffer = self.content_buffer[trigger_end:]
                    return True, yielded
                # 检测动态trigger signal格式
                dynamic_match = self._detect_dynamic_trigger(idx)
                if dynamic_match:
                    logger.debug(f"detected dynamic trigger signal: {dynamic_match}")
                    self.state = "tool_parsing"
                    # 跳过trigger signal，从trigger signal之后开始
                    trigger_end = idx + len(dynamic_match)
                    self.content_buffer = self.content_buffer[trigger_end:]
                    return True, yielded
                # 检测<function_calls>标签
                if self._can_detect_function_calls(idx):
                    logger.debug("detected <function_calls> without trigger; entering tool parsing")
                    self.state = "tool_parsing"
                    self.content_buffer = self.content_buffer[idx:]
                    return True, yielded
            segment = self.content_buffer[idx:]
            # 检查固定trigger signal的partial match
            if self.signal_len:
                partial_signal = segment[:self.signal_len]
                if self.signal.startswith(partial_signal):
                    break
            # 检查动态trigger signal的partial match
            if self._could_be_dynamic_trigger_start(segment):
                break
            lower_segment = segment.lower()
            candidate_fc = lower_segment[:self.fc_tag_len]
            if self.fc_tag.startswith(candidate_fc):
                break
            if lower_segment.startswith('<think') and len(segment) < len('<think>'):
                break
            if lower_segment.startswith('</think') and len(segment) < len('</think>'):
                break
            yielded += self.content_buffer[idx]
            idx += 1
        self.content_buffer = self.content_buffer[idx:]
        return False, yielded

    def _update_think_state(self, pos: int) -> int:
        remain = self.content_buffer[pos:]
        if remain.startswith("<think>"):
            self.think_depth += 1
            self.in_think_block = True
            return 7
        if remain.endswith("</think>"):
            self.think_depth = max(0, self.think_depth - 1)
            # 重要修复：当遇到</think>时，无论深度如何都退出think block状态
            # 这处理了streaming中可能出现的不平衡标签情况
            self.in_think_block = False
            # 返回整个remain的长度，因为我们要消耗到</think>结束的所有内容
            return len(remain)
        return 0

    def _can_detect(self, pos: int) -> bool:
        return pos + self.signal_len <= len(self.content_buffer)

    def _can_detect_function_calls(self, pos: int) -> bool:
        if pos + self.fc_tag_len > len(self.content_buffer):
            return False
        return self.content_buffer[pos:pos + self.fc_tag_len].lower() == self.fc_tag.lower()

    def _detect_dynamic_trigger(self, pos: int) -> Optional[str]:
        remaining = self.content_buffer[pos:]
        match = self.dynamic_trigger_pattern.search(remaining)
        if match and match.start() == 0:  # 确保匹配从当前位置开始
            return match.group(0)
        return None

    def _could_be_dynamic_trigger_start(self, segment: str) -> bool:
        # 检查是否可能是动态trigger signal的开始部分
        # 检查可能的前缀：<, <F, <Fu, <Fun, <Func, <Functi, <Functio, <Function, <Function_
        possible_prefixes = ["<", "<F", "<Fu", "<Fun", "<Func", "<Functi", "<Functio", "<Function", "<Function_"]
        for prefix in possible_prefixes:
            if prefix.startswith(segment.lower()) or segment.lower().startswith(prefix.lower()):
                return True
        return False

    def finalize(self) -> Optional[List[Dict[str, Any]]]:
        if self.state == "tool_parsing":
            return parse_function_calls_xml(self.content_buffer, self.trigger_signal)
        return None




def parse_function_calls_xml(xml_string: str, trigger_signal: str) -> Optional[List[Dict[str, Any]]]:
    if not xml_string:
        return None
    cleaned = remove_think_blocks(xml_string)
    matches = list(re.finditer(r"<function_calls>([\s\S]*?)</function_calls>", cleaned, flags=re.IGNORECASE))
    if not matches:
        return None
    body = matches[-1].group(1)
    calls = re.findall(r"<function_call>([\s\S]*?)</function_call>", body, flags=re.IGNORECASE)
    results: List[Dict[str, Any]] = []
    for block in calls:
        name: Optional[str] = None
        container_body: Optional[str] = None
        container_match = re.match(r"\s*<([^\s>/]+)>([\s\S]*)</\1>\s*", block, flags=re.IGNORECASE)
        if container_match:
            container_candidate = container_match.group(1).strip()
            container_body = container_match.group(2)
        for name_tag in ("tool", "tool_name", "function", "function_name"):
            name_match = re.search(fr"<{name_tag}>(.*?)</{name_tag}>", block, flags=re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                break
        if not name and container_match:
            container_candidate = container_match.group(1).strip()
            if container_candidate.lower() not in {"args", "parameters", "arguments"}:
                name = container_candidate
        if not name:
            logger.debug("skipping function_call block without name tag: %s", block)
            continue
        args: Dict[str, Any] = {}
        args_content: Optional[str] = None
        for args_tag in ("args", "parameters", "arguments"):
            args_match = re.search(fr"<{args_tag}>([\s\S]*?)</{args_tag}>", block, flags=re.IGNORECASE)
            if args_match:
                args_content = args_match.group(1)
                break
        if not args_content and container_body:
            args_content = container_body
        if args_content:
            arg_pairs = re.findall(r"<([^\s>/]+)>([\s\S]*?)</\1>", args_content, flags=re.IGNORECASE)
            for key, raw in arg_pairs:
                try:
                    args[key] = json.loads(raw)
                except Exception:
                    args[key] = raw
        results.append({"name": name, "args": args})

    return results or None



def safe_process_tool_choice(tool_choice: Union[str, ToolChoice, None]) -> str:
    if tool_choice is None:
        return ""
    if isinstance(tool_choice, str):
        if tool_choice == "none":
            return (
                "\n\nIMPORTANT: You must not trigger any tools in this turn. Provide a direct answer."
            )
        return ""
    if isinstance(tool_choice, ToolChoice):
        name = tool_choice.function.get("name")
        if name:
            return f"\n\nIMPORTANT: Only call the tool named `{name}`."
        return ""


def preprocess_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            processed.append(item)
            continue
        role = item.get("role")
        if role == "tool":
            tool_call_id = item.get("tool_call_id")
            content = item.get("content")
            if tool_call_id and content:
                formatted = format_tool_result_for_ai(tool_call_id, content)
                processed.append({"role": "user", "content": formatted})
            continue
        if role == "assistant" and item.get("tool_calls"):
            formatted_tc = format_assistant_tool_calls_for_ai(item.get("tool_calls", []), GLOBAL_TRIGGER_SIGNAL)
            merged_content = item.get("content") or ""
            content = f"{merged_content}\n{formatted_tc}".strip()
            clone = {k: v for k, v in item.items() if k not in {"tool_calls", "content"}}
            clone["role"] = "assistant"
            clone["content"] = content
            processed.append(clone)
            continue
        if role == "developer" and app_config.features.convert_developer_to_system:
            clone = dict(item)
            clone["role"] = "system"
            processed.append(clone)
            continue
        processed.append(item)
    return processed


def validate_message_structure(messages: List[Dict[str, Any]]) -> bool:
    valid_roles = {"system", "user", "assistant", "tool"}
    if not app_config.features.convert_developer_to_system:
        valid_roles.add("developer")
    for idx, message in enumerate(messages):
        role = message.get("role")
        if role not in valid_roles:
            logger.error("invalid role %s at index %s", role, idx)
            return False
        if role == "tool" and not message.get("tool_call_id"):
            logger.error("tool message missing tool_call_id at index %s", idx)
            return False
    return True


def find_upstream(model_name: str) -> Tuple[UpstreamService, str]:
    if app_config.features.model_passthrough and app_config.features.passthrough_service:
        service = next(
            (svc for svc in app_config.upstream_services if svc.name == app_config.features.passthrough_service),
            None,
        )
        if not service:
            raise HTTPException(status_code=500, detail="Passthrough service not configured")
        return service, model_name
    service = get_primary_service()
    return service, model_name


def get_primary_service() -> UpstreamService:
    if not app_config.upstream_services:
        raise HTTPException(status_code=503, detail="No upstream services configured")
    return app_config.upstream_services[0]


async def verify_api_key(authorization: str = Header(...)) -> str:
    header_value = authorization.strip()
    if not header_value:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return header_value


async def forward_non_stream_request(url: str, payload: dict, headers: dict, has_fc: bool) -> JSONResponse:
    assert http_client is not None
    response = await http_client.post(url, json=payload, headers=headers)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("upstream error %s: %s", exc.response.status_code, exc.response.text)
        error_map = {
            400: ("Invalid request parameters", "invalid_request_error"),
            401: ("Authentication failed", "authentication_error"),
            403: ("Access forbidden", "permission_error"),
            429: ("Rate limit exceeded", "rate_limit_error"),
        }
        message, code = error_map.get(exc.response.status_code, ("Request processing failed", "upstream_error"))
        return JSONResponse(status_code=exc.response.status_code, content={"error": {"message": message, "type": code}})
    model_name = payload.get("model", "")
    try:
        data = response.json()
    except (json.JSONDecodeError, ValueError):
        raw_text = response.text or ""
        if has_fc:
            parsed = parse_function_calls_xml(raw_text, GLOBAL_TRIGGER_SIGNAL)
            if parsed:
                tool_calls = build_tool_call_records(parsed)
                completion = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                            "finish_reason": "tool_calls",
                        }
                    ],
                }
                return JSONResponse(content=completion)
        logger.warning("non-JSON upstream response, returning as plain text")
        fallback = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": raw_text},
                    "finish_reason": "stop",
                }
            ],
        }
        return JSONResponse(content=fallback)
    if has_fc:
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        parsed = parse_function_calls_xml(content or "", GLOBAL_TRIGGER_SIGNAL)
        if parsed:
            tool_calls = build_tool_call_records(parsed)
            if not data.get("choices"):
                data["choices"] = [{"index": 0}]
            primary = data["choices"][0]
            primary["message"] = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            primary["finish_reason"] = "tool_calls"
    return JSONResponse(content=data)



async def stream_proxy_with_fc_transform(url: str, payload: dict, headers: dict, model: str, has_fc: bool, trigger_signal: str):
    assert http_client is not None
    if not has_fc:
        async with http_client.stream("POST", url, json=payload, headers=headers) as response:
            async for chunk in response.aiter_bytes():
                yield chunk
        return
    detector = StreamingFunctionCallDetector(trigger_signal)

    def build_sse_chunks(parsed_tools: List[Dict[str, Any]]) -> List[str]:
        base_calls = build_tool_call_records(parsed_tools)
        tool_calls = []
        for idx, record in enumerate(base_calls):
            tool_calls.append({"index": idx, **record})
        created_ts = int(time.time())
        head = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "tool_calls": tool_calls}, "finish_reason": None}],
        }
        tail = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
        return [f"data: {json.dumps(head)}\n\n", f"data: {json.dumps(tail)}\n\n", "data: [DONE]\n\n"]

    async with http_client.stream("POST", url, json=payload, headers=headers) as response:
        if response.status_code != 200:
            detail = await response.aread()
            logger.error("upstream stream error %s: %s", response.status_code, detail.decode("utf-8", errors="ignore"))
            error_body = {"error": {"message": "Upstream error", "type": "upstream_error"}}
            yield f"data: {json.dumps(error_body)}\n\n"
            yield "data: [DONE]\n\n"
            return
        async for line in response.aiter_lines():
            if detector.state == "tool_parsing":
                if line.startswith("data:"):
                    chunk = line[len("data:"):].strip()
                    if chunk and chunk != "[DONE]":
                        try:
                            payload_json = json.loads(chunk)
                        except Exception:
                            detector.content_buffer += chunk
                            continue
                        delta = payload_json.get("choices", [{}])[0].get("delta", {}).get("content", "") or ""
                        detector.content_buffer += delta
                continue
            if not line.startswith("data:"):
                continue
            chunk = line[len("data:"):].strip()
            if not chunk or chunk == "[DONE]":
                continue
            try:
                payload_json = json.loads(chunk)
            except Exception:
                detected, forward_content = detector.process_chunk(chunk)
                if forward_content:
                    yield build_streaming_text_chunk(forward_content, model)
                if detected:
                    continue
                continue
            delta = payload_json.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "") or ""
            detected, forward_content = detector.process_chunk(content)
            if forward_content:
                forward_chunk = {
                    "id": payload_json.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
                    "object": "chat.completion.chunk",
                    "created": payload_json.get("created", int(time.time())),
                    "model": payload_json.get("model", model),
                    "choices": [{"index": 0, "delta": {"content": forward_content}}],
                }
                yield f"data: {json.dumps(forward_chunk)}\n\n"
            if detected:
                continue
    if detector.state == "tool_parsing":
        parsed = detector.finalize()
        if parsed:
            for chunk in build_sse_chunks(parsed):
                yield chunk
            return
        logger.warning("tool call trigger detected but parsing failed; returning raw content")
        fallback = detector.content_buffer or "Function call parse failed"
        fallback_chunk = {
            "id": f"chatcmpl-fallback-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": fallback}}],
        }
        yield f"data: {json.dumps(fallback_chunk)}\n\n"
    elif detector.state == "detecting" and detector.content_buffer:
        flush_chunk = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": detector.content_buffer}}],
        }
        yield f"data: {json.dumps(flush_chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def proxy_models_request(client_token: Optional[str]) -> Dict[str, Any]:
    service = get_primary_service()
    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client unavailable")
    url = service.base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    auth_token: Optional[str] = None
    if client_token:
        headers["Authorization"] = client_token
    elif service.api_key:
        headers["Authorization"] = _ensure_bearer(service.api_key)
    if service.extra_headers:
        headers.update(service.extra_headers)
    try:
        response = await http_client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        logger.error("upstream models error %s: %s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=exc.response.status_code, detail="Upstream models endpoint error")
    except httpx.RequestError as exc:
        logger.error("failed to reach upstream models endpoint: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach upstream models endpoint")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest, api_key: str = Depends(verify_api_key)):
    try:
        service, upstream_model = find_upstream(body.model)
        upstream_url = service.base_url.rstrip("/") + "/chat/completions"
        processed_messages = preprocess_messages(body.messages)
        if not validate_message_structure(processed_messages):
            logger.warning("message validation failed but continuing")
        payload = body.model_dump(exclude_unset=True)
        payload["model"] = upstream_model
        payload["messages"] = processed_messages
        is_fc_enabled = app_config.features.enable_function_calling
        has_tools = bool(body.tools)
        has_function_call = is_fc_enabled and has_tools
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("preprocess failed: %s", exc)
        return JSONResponse(status_code=422, content={"error": {"message": "Invalid request", "type": "invalid_request_error"}})
    if has_function_call:
        prompt, _ = generate_function_prompt(body.tools or [], GLOBAL_TRIGGER_SIGNAL)
        prompt += safe_process_tool_choice(body.tool_choice)
        payload["messages"].insert(0, {"role": "system", "content": prompt})
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    elif has_tools and not is_fc_enabled:
        payload.pop("tools", None)
        payload.pop("tool_choice", None)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = api_key
    elif service.api_key:
        headers["Authorization"] = _ensure_bearer(service.api_key)
    headers.update(service.extra_headers or {})
    if body.stream:
        headers["Accept"] = "text/event-stream"
        return StreamingResponse(
            stream_proxy_with_fc_transform(upstream_url, payload, headers, body.model, has_function_call, GLOBAL_TRIGGER_SIGNAL),
            media_type="text/event-stream",
        )
    headers["Accept"] = "application/json"
    return await forward_non_stream_request(upstream_url, payload, headers, has_function_call)


@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    return await proxy_models_request(api_key)


@app.get("/models")
async def list_models_public():
    return await proxy_models_request(None)


@app.get("/")
async def root():
    return {
        "status": "format proxy ready",
        "trigger_signal": GLOBAL_TRIGGER_SIGNAL,
        "upstreams": [svc.name for svc in app_config.upstream_services],
        "features": app_config.features.model_dump(),
    }


@app.exception_handler(ValidationError)
async def validation_exception_handler(_: Request, exc: ValidationError):
    logger.error("validation error: %s", exc)
    return JSONResponse(status_code=422, content={"error": {"message": "Invalid request format", "type": "invalid_request_error"}})


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception):
    logger.error("unexpected error: %s", exc)
    return JSONResponse(status_code=500, content={"error": {"message": "Internal server error", "type": "server_error"}})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=app_config.server.host,
        port=app_config.server.port,
        log_level=("critical" if log_level == "DISABLED" else log_level.lower()),
    )
