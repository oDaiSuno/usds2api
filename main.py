import base64
import json
import os
import time
import uuid
import threading
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import httpx
from bs4 import BeautifulSoup
from Crypto.Cipher import AES
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field


# Global variables
VALID_CLIENT_KEYS: set = set()
USTC_ACCOUNTS: List["USTCAccount"] = []
USTC_ACCOUNT_CONFIGS: List[Dict[str, Any]] = []
auth_round_robin_index: int = 0
auth_rotation_lock = threading.Lock()
MAX_RETRIES = 3
account_refresh_lock: Optional[asyncio.Lock] = None
token_refresh_task: Optional[asyncio.Task] = None

TOKEN_REFRESH_INTERVAL_SECONDS = int(
    os.getenv("USTC_TOKEN_REFRESH_INTERVAL_SECONDS", str(8 * 3600))
)
QUEUE_MAX_ATTEMPTS = int(os.getenv("USTC_QUEUE_MAX_ATTEMPTS", "20"))
QUEUE_WAIT_SECONDS = float(os.getenv("USTC_QUEUE_WAIT_SECONDS", "3"))
TOKEN_EXPIRED_KEYWORD = "登录超时"

LOGIN_URL = (
    "https://id.ustc.edu.cn/cas/login?"
    "service=https%3A%2F%2Fchat.ustc.edu.cn%2Fustchat%2F"
)

LOGIN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,"
        "application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://chat.ustc.edu.cn/",
}

QUEUE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0",
    "Accept": "application/json, text/plain, */*",
}


class TokenExpiredError(Exception):
    pass


@dataclass
class USTCAccount:
    token: str = ""
    username: Optional[str] = None
    password: Optional[str] = None
    last_refresh_ts: int = 0
    config_index: int = -1

    def can_refresh(self) -> bool:
        return bool(self.username and self.password)


def ensure_bearer(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return ""
    return token if token.lower().startswith("bearer ") else f"Bearer {token}"


def apply_account_token(account: USTCAccount, token: str) -> bool:
    new_token = ensure_bearer(token)
    changed = account.token != new_token
    account.token = new_token
    account.last_refresh_ts = int(time.time())
    if 0 <= account.config_index < len(USTC_ACCOUNT_CONFIGS):
        USTC_ACCOUNT_CONFIGS[account.config_index]["auth"] = new_token
    return changed


def persist_ustc_accounts() -> None:
    if not USTC_ACCOUNT_CONFIGS:
        return
    try:
        with open("ustcds.json", "w", encoding="utf-8") as f:
            json.dump(USTC_ACCOUNT_CONFIGS, f, indent=4)
    except Exception as exc:
        print(f"写入 ustcds.json 失败: {exc}")


def token_expired_from_text(text: str) -> bool:
    if not text:
        return False
    if TOKEN_EXPIRED_KEYWORD in text:
        return True
    try:
        data = json.loads(text)
    except Exception:
        return False
    detail = data.get("detail")
    return isinstance(detail, str) and TOKEN_EXPIRED_KEYWORD in detail


def normalize_messages(messages: List["ChatMessage"]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for msg in messages:
        content: Union[str, List[Dict[str, Any]]] = msg.content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        text_parts.append(str(item["text"]))
                    elif item.get("type") == "text" and "text" in item:
                        text_parts.append(str(item["text"]))
                elif isinstance(item, str):
                    text_parts.append(item)
            joined = " ".join(part for part in text_parts if part)
            normalized.append({"role": msg.role, "content": joined})
        else:
            normalized.append({"role": msg.role, "content": str(content)})
    return normalized


def pkcs7_pad(data: bytes, block_size: int = 16) -> bytes:
    pad = block_size - len(data) % block_size
    return data + bytes([pad]) * pad


def encrypt_password(key_base64: str, plaintext: str) -> str:
    key = base64.b64decode(key_base64)
    cipher = AES.new(key, AES.MODE_ECB)
    padded = pkcs7_pad(plaintext.encode("utf-8"))
    encrypted = cipher.encrypt(padded)
    return base64.b64encode(encrypted).decode("utf-8")


async def fetch_login_page(client: httpx.AsyncClient) -> tuple[str, str]:
    resp = await client.get(LOGIN_URL, timeout=20.0)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    execution_value: Optional[str] = None
    execution_input = soup.find("input", {"name": "execution"})
    if execution_input and execution_input.get("value"):
        execution_value = execution_input["value"].strip()
    else:
        exec_holder = soup.find(id="login-page-flowkey")
        if exec_holder:
            execution_value = exec_holder.text.strip()

    if not execution_value:
        raise RuntimeError("未解析到 execution")

    crypto_elem = soup.find(id="login-croypto")
    if not crypto_elem or not crypto_elem.text.strip():
        raise RuntimeError("未解析到 login-croypto")

    crypto_key = crypto_elem.text.strip()
    return execution_value, crypto_key


async def get_login_ticket(
    client: httpx.AsyncClient,
    execution: str,
    crypto_key: str,
    username: str,
    encrypted_password: str,
) -> str:
    url = "https://id.ustc.edu.cn/cas/login"
    payload = {
        "username": username,
        "type": "UsernamePassword",
        "_eventId": "submit",
        "geolocation": "",
        "execution": execution,
        "croypto": crypto_key,
        "password": encrypted_password,
        "targetSystem": "sso",
        "siteId": "sourceId",
    }

    headers = LOGIN_HEADERS.copy()
    headers.update(
        {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Cache-Control": "max-age=0",
            "Origin": "https://id.ustc.edu.cn",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Referer": LOGIN_URL,
        }
    )

    response = await client.post(url, data=payload, headers=headers)
    response.raise_for_status()
    ticket_url = str(response.url)
    if "ticket=" not in ticket_url:
        raise RuntimeError("登录未返回 ticket")
    return ticket_url.split("ticket=")[1]


async def get_token(client: httpx.AsyncClient, ticket: str) -> str:
    url = "https://chat.ustc.edu.cn/ms-api/cas"
    payload = {"ticket": ticket}

    headers = {
        "User-Agent": LOGIN_HEADERS["User-Agent"],
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "origin": "https://chat.ustc.edu.cn",
        "referer": "https://chat.ustc.edu.cn/ustchat/?ticket=" + ticket,
        "accept-language": "zh-CN,zh;q=0.9",
    }

    response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    token_data = response.json().get("data", {})
    if "token" not in token_data:
        raise RuntimeError("未解析到 token")
    return token_data["token"]


async def login_and_fetch_token(
    client: httpx.AsyncClient, username: str, password: str
) -> str:
    execution, crypto_key = await fetch_login_page(client)
    encrypted_password = encrypt_password(crypto_key, password)
    ticket = await get_login_ticket(client, execution, crypto_key, username, encrypted_password)
    return await get_token(client, ticket)


async def refresh_account_token(
    account: USTCAccount, *, force: bool = False
) -> None:
    global account_refresh_lock
    if not account.can_refresh():
        return

    if account_refresh_lock is None:
        account_refresh_lock = asyncio.Lock()

    if not force:
        elapsed = int(time.time()) - account.last_refresh_ts
        if elapsed < TOKEN_REFRESH_INTERVAL_SECONDS:
            return

    async with account_refresh_lock:
        if not force:
            elapsed = int(time.time()) - account.last_refresh_ts
            if elapsed < TOKEN_REFRESH_INTERVAL_SECONDS:
                return

        async with httpx.AsyncClient(headers=LOGIN_HEADERS, follow_redirects=True) as client:
            try:
                token = await login_and_fetch_token(
                    client, account.username or "", account.password or ""
                )
                if apply_account_token(account, token):
                    persist_ustc_accounts()
            except Exception as exc:
                print(f"刷新账号 {account.username or 'N/A'} 失败: {exc}")
                if force:
                    raise


async def periodic_token_refresh_loop() -> None:
    while True:
        await asyncio.sleep(TOKEN_REFRESH_INTERVAL_SECONDS)
        for account in list(USTC_ACCOUNTS):
            try:
                await refresh_account_token(account, force=True)
            except Exception:
                continue


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


async def load_ustc_accounts():
    global USTC_ACCOUNTS, USTC_ACCOUNT_CONFIGS, account_refresh_lock, token_refresh_task
    account_refresh_lock = asyncio.Lock()

    try:
        with open("ustcds.json", "r", encoding="utf-8") as f:
            raw_accounts = json.load(f)
    except FileNotFoundError:
        print("Error: ustcds.json not found. No USTC账号可用。")
        USTC_ACCOUNTS = []
        return
    except Exception as exc:
        print(f"Error loading ustcds.json: {exc}")
        USTC_ACCOUNTS = []
        return

    if not isinstance(raw_accounts, list):
        print("ustcds.json 格式错误，需为列表。")
        USTC_ACCOUNTS = []
        return

    USTC_ACCOUNT_CONFIGS = raw_accounts
    accounts: List[USTCAccount] = []
    config_dirty = False

    async with httpx.AsyncClient(headers=LOGIN_HEADERS, follow_redirects=True) as client:
        for idx, item in enumerate(USTC_ACCOUNT_CONFIGS):
            if not isinstance(item, dict):
                continue

            username = item.get("username")
            password = item.get("password")
            token = item.get("auth") or item.get("token") or ""

            if username and password:
                try:
                    token = await login_and_fetch_token(client, username, password)
                except Exception as exc:
                    print(f"登录账号 {username} 失败: {exc}")
                    continue

            if not token:
                print("跳过无 token 的账号配置。")
                continue

            account = USTCAccount(
                username=username,
                password=password,
                config_index=idx,
            )
            if apply_account_token(account, token):
                config_dirty = True
            accounts.append(account)

    USTC_ACCOUNTS = accounts
    print(f"已加载 {len(USTC_ACCOUNTS)} 个 USTC 账号。")

    if config_dirty:
        persist_ustc_accounts()

    if token_refresh_task is None and USTC_ACCOUNTS:
        token_refresh_task = asyncio.create_task(periodic_token_refresh_loop())


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
    await load_ustc_accounts()
    print("Server initialization completed.")


def get_next_account() -> Optional[USTCAccount]:
    global auth_round_robin_index, USTC_ACCOUNTS
    with auth_rotation_lock:
        if not USTC_ACCOUNTS:
            return None
        account = USTC_ACCOUNTS[auth_round_robin_index % len(USTC_ACCOUNTS)]
        auth_round_robin_index = (auth_round_robin_index + 1) % len(USTC_ACCOUNTS)
        return account


async def get_queue_code() -> str:
    code = str(uuid.uuid4())
    url = f"https://chat.ustc.edu.cn/ms-api/mei-wei-bu-yong-deng?queue_code={code}"
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(QUEUE_MAX_ATTEMPTS):
            response = await client.get(url, headers=QUEUE_HEADERS)
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "")
                except Exception:
                    detail = ""
                if TOKEN_EXPIRED_KEYWORD in detail:
                    raise TokenExpiredError(detail)
                response.raise_for_status()

            try:
                wait_value = response.json()["data"]["wait"]
            except Exception:
                wait_value = None

            if wait_value == 0:
                return code

            await asyncio.sleep(QUEUE_WAIT_SECONDS)

    raise RuntimeError("排队获取 code 超时")


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
            detail=(
                f"Model '{request.model}' not found. Available models: "
                + ", ".join(model_mapping.keys())
            ),
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    if not USTC_ACCOUNTS:
        raise HTTPException(status_code=503, detail="No USTC accounts configured.")

    ustc_model = model_mapping[request.model]
    normalized_messages = normalize_messages(request.messages)

    for _ in range(MAX_RETRIES):
        account = get_next_account()
        if not account:
            break

        try:
            await refresh_account_token(account)
        except Exception as exc:
            print(f"刷新账号失败: {exc}")
            continue

        try:
            queue_code = await get_queue_code()
        except TokenExpiredError:
            try:
                await refresh_account_token(account, force=True)
            except Exception:
                pass
            continue
        except Exception as exc:
            print(f"获取排队码失败: {exc}")
            continue

        payload = {
            "messages": normalized_messages,
            "queue_code": queue_code,
            "model": ustc_model,
            "stream": request.stream,
            "with_search": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
            ),
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "authorization": account.token,
        }

        try:
            if request.stream:
                stream_gen = stream_ustc_response(payload, headers)
                try:
                    first_chunk = await stream_gen.__anext__()
                except StopAsyncIteration:
                    async def empty_stream():
                        yield "data: [DONE]\n\n"

                    return StreamingResponse(
                        empty_stream(),
                        media_type="text/event-stream",
                        headers={
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                            "X-Accel-Buffering": "no",
                        },
                    )
                except TokenExpiredError:
                    try:
                        await refresh_account_token(account, force=True)
                    except Exception:
                        pass
                    continue

                async def chained_stream(first: str, generator):
                    yield first
                    async for chunk in generator:
                        yield chunk

                return StreamingResponse(
                    chained_stream(first_chunk, stream_gen),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            return await build_ustc_non_stream_response(payload, headers, request.model)
        except TokenExpiredError:
            try:
                await refresh_account_token(account, force=True)
            except Exception:
                pass
            continue
        except Exception as exc:
            print(f"USTC API error: {exc}")
            continue

    raise HTTPException(status_code=503, detail="All attempts to contact USTC API failed.")


async def stream_ustc_response(payload: dict, headers: dict):
    url = "https://chat.ustc.edu.cn/ms-api/chat-messages"
    async with httpx.AsyncClient(timeout=1800) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="ignore")
                if token_expired_from_text(body):
                    raise TokenExpiredError(body)
                response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue
                if token_expired_from_text(line):
                    raise TokenExpiredError(line)
                if line.startswith("data: "):
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
    except TokenExpiredError:
        raise
    except Exception as e:
        print(f"Error accumulating stream: {e}")
    
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

    uvicorn.run(app, host="0.0.0.0", port=8010)
