import json
import os
import time
import uuid
import threading
import asyncio
import httpx
from typing import Any, Dict, List, Optional, TypedDict, Union
from datetime import datetime, timedelta

import requests
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


# Global variables
VALID_CLIENT_KEYS: set = set()
USTC_AUTHS: List[str] = []
auth_round_robin_index: int = 0
auth_rotation_lock = threading.Lock()
MAX_RETRIES = 3


# Pydantic Models
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    reasoning_content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int] = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


class StreamChoice(BaseModel):
    delta: Dict[str, Any] = Field(default_factory=dict)
    index: int = 0
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# FastAPI App
app = FastAPI(title="USTC OpenAI API Adapter")
security = HTTPBearer(auto_error=False)


def load_client_api_keys():
    """Load client API keys from client_api_keys.json"""
    global VALID_CLIENT_KEYS
    try:
        with open("client_api_keys.json", "r", encoding="utf-8") as f:
            keys = json.load(f)
            VALID_CLIENT_KEYS = set(keys) if isinstance(keys, list) else set()
            print(f"Successfully loaded {len(VALID_CLIENT_KEYS)} client API keys.")
    except FileNotFoundError:
        print("Error: client_api_keys.json not found. Client authentication will fail.")
        VALID_CLIENT_KEYS = set()
    except Exception as e:
        print(f"Error loading client_api_keys.json: {e}")
        VALID_CLIENT_KEYS = set()


def load_ustc_auths():
    global USTC_AUTHS
    try:
        with open("ustcds.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            USTC_AUTHS = [item["auth"] for item in data]
            print(f"Loaded {len(USTC_AUTHS)} USTC auth tokens.")
    except Exception as e:
        print(f"Error loading ustcds.json: {e}")
        USTC_AUTHS = []


async def authenticate_client(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Authenticate client based on API key in Authorization header"""
    if not VALID_CLIENT_KEYS:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Client API keys not configured on server.",
        )

    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=401,
            detail="API key required in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if auth.credentials not in VALID_CLIENT_KEYS:
        raise HTTPException(status_code=403, detail="Invalid client API key.")


@app.on_event("startup")
async def startup():
    print("Starting USTC OpenAI API Adapter server...")
    load_client_api_keys()
    load_ustc_auths()
    print("Server initialization completed.")


def get_next_ustc_auth() -> Optional[str]:
    global auth_round_robin_index, USTC_AUTHS
    with auth_rotation_lock:
        if not USTC_AUTHS:
            return None
        auth = USTC_AUTHS[auth_round_robin_index % len(USTC_AUTHS)]
        auth_round_robin_index = (auth_round_robin_index + 1) % len(USTC_AUTHS)
        return auth


async def get_queue_code(auth: str, client: httpx.AsyncClient) -> Optional[str]:
    code = str(uuid.uuid4())
    url = f"https://chat.ustc.edu.cn/ms-api/mei-wei-bu-yong-deng?queue_code={code}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
        "Accept": "application/json, text/plain, */*",
        "authorization": auth,
    }
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return code
    except Exception:
        return None


def get_models_list_response() -> ModelList:
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            model_mapping = json.load(f)
            if not isinstance(model_mapping, dict):
                return ModelList(data=[])
        model_infos = [
            ModelInfo(id=model_id, created=int(time.time()), owned_by="USTC")
            for model_id in model_mapping.keys()
        ]
        return ModelList(data=model_infos)
    except Exception as e:
        print(f"Error loading models.json: {e}")
        return ModelList(data=[])


@app.get("/v1/models", response_model=ModelList)
async def list_v1_models(_: None = Depends(authenticate_client)):
    """List available models - authenticated"""
    return get_models_list_response()


@app.get("/models", response_model=ModelList)
async def list_models_no_auth():
    """List available models without authentication - for client compatibility"""
    return get_models_list_response()


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: None = Depends(authenticate_client)
):
    try:
        with open("models.json", "r", encoding="utf-8") as f:
            model_mapping = json.load(f)
    except Exception:
        model_mapping = {}
    
    if request.model not in model_mapping:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available models: {', '.join(model_mapping.keys())}",
        )

    if not request.messages:
        raise HTTPException(
            status_code=400, detail="No messages provided in the request."
        )

    for attempt in range(MAX_RETRIES):
        auth = get_next_ustc_auth()
        if not auth:
            break

        async with httpx.AsyncClient(timeout=1800, verify=False) as client:
            queue_code = await get_queue_code(auth, client)
            if not queue_code:
                continue

            messages = []
            for msg in request.messages:
                content = msg.content
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if "text" in item:
                                text_parts.append(item["text"])
                            elif "type" in item and item["type"] == "text" and "text" in item:
                                text_parts.append(item["text"])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = " ".join(text_parts)
                messages.append({"role": msg.role, "content": content})

            ustc_model = model_mapping[request.model]
            payload = {
                "messages": messages,
                "queue_code": queue_code,
                "model": ustc_model,
                "stream": request.stream,
                "with_search": False,
            }
            if request.temperature is not None:
                payload["temperature"] = request.temperature

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
                "Accept": "text/event-stream" ,
                "Content-Type": "application/json",
                "authorization": auth,
            }

            try:
                if request.stream:
                    return StreamingResponse(
                        stream_ustc_response(payload, headers),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
                else:
                    return await build_ustc_non_stream_response(payload, headers, request.model)
            except Exception as e:
                print(f"USTC API error: {e}")
                continue

    raise HTTPException(status_code=503, detail="All attempts to contact USTC API failed.")


async def stream_ustc_response(payload: dict, headers: dict):
    url = "https://chat.ustc.edu.cn/ms-api/chat-messages"
    async with httpx.AsyncClient(timeout=1800, verify=False) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line and line.startswith("data: "):
                    yield line + "\n\n"


async def build_ustc_non_stream_response(payload: dict, headers: dict, model: str) -> ChatCompletionResponse:
    payload_copy = payload.copy()
    payload_copy["stream"] = True
    
    headers_copy = headers.copy()
    headers_copy["Accept"] = "text/event-stream"
    
    accumulated_content = ""
    stream_id = ""
    created_time = int(time.time())
    
    try:
        async for line in stream_ustc_response(payload_copy, headers_copy):
            line = line.strip()
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "id" in data and not stream_id:
                        stream_id = data["id"]
                    if "created" in data:
                        created_time = data["created"]
                    if "choices" in data and data["choices"]:
                        choice = data["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            accumulated_content += choice["delta"]["content"]
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error accumulating stream: {e}")  # 添加捕获，防无声失败
    
    return ChatCompletionResponse(
        id=stream_id or f"chatcmpl-{uuid.uuid4().hex}",
        created=created_time,
        model=model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=accumulated_content)
            )
        ]
    )




if __name__ == "__main__":
    import uvicorn

    if not os.path.exists("client_api_keys.json"):
        print("Warning: client_api_keys.json not found. Creating a dummy file.")
        dummy_key = f"sk-ustc-{uuid.uuid4().hex}"
        with open("client_api_keys.json", "w", encoding="utf-8") as f:
            json.dump([dummy_key], f, indent=2)
        print(f"Created dummy client_api_keys.json with key: {dummy_key}")

    if not os.path.exists("models.json"):
        print("Warning: models.json not found. Creating default file.")
        with open("models.json", "w", encoding="utf-8") as f:
            json.dump({"deepseek-v3": "deepseek-v3", "deepseek-r1": "deepseek"}, f, indent=4)
        print("Created models.json with USTC models.")

    if not os.path.exists("ustcds.json"):
        print("Warning: ustcds.json not found. Creating default file.")
        with open("ustcds.json", "w", encoding="utf-8") as f:
            json.dump([{"auth": "Bearer YOUR_AUTH_TOKEN_HERE"}], f, indent=4)
        print("Created ustcds.json template. Please update with your auth tokens.")

    print("\n--- USTC OpenAI API Adapter ---")
    print("Endpoints:")
    print("  GET  /v1/models (Client API Key Auth)")
    print("  GET  /models (No Auth)")
    print("  POST /v1/chat/completions (Client API Key Auth)")
    print("------------------------------------")

    uvicorn.run(app, host="0.0.0.0", port=8000)