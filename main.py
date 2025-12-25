import os
import httpx
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# --- 1. 使用 lifespan 管理全局 HTTP 客户端 ---
# 这能有效解决“第一次连接慢/失败”的问题，并复用连接池
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时：创建一个全局的 client
    print("正在初始化全局 HTTP 客户端...")
    app.state.http_client = httpx.AsyncClient(timeout=60.0) # 设置更长的 60秒 超时
    yield
    # 关闭时：清理资源
    print("正在关闭 HTTP 客户端...")
    await app.state.http_client.aclose()

# 初始化 FastAPI (带上 lifespan)
app = FastAPI(lifespan=lifespan)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 获取 API Key
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    print("警告: 未检测到 DASHSCOPE_API_KEY 环境变量，请确保已设置。")

# 阿里云 Wan API 配置
BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
MODEL_NAME = "wan2.6-t2v"

class VideoGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    size: str = "1280*720"
    duration: int = 5
    audio: bool = True
    prompt_extend: bool = True

# --- 核心功能函数 ---

async def call_wan_api_create(client: httpx.AsyncClient, data: VideoGenerationRequest):
    url = f"{BASE_URL}/services/aigc/video-generation/video-synthesis"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Async": "enable"
    }

    payload = {
        "model": MODEL_NAME,
        "input": {
            "prompt": data.prompt,
            "negative_prompt": data.negative_prompt
        },
        "parameters": {
            "size": data.size,
            "duration": data.duration,
            "prompt_extend": data.prompt_extend,
            "audio": data.audio
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 使用传入的 client，而不是新建
            response = await client.post(url, headers=headers, json=payload)
            
            # 打印原始响应，方便调试
            if response.status_code != 200:
                print(f"API Error Response: {response.text}") # 在终端打印详细错误
                raise HTTPException(status_code=response.status_code, detail=response.json())
                
            response_json = response.json()
            if "output" not in response_json or "task_id" not in response_json["output"]:
                 raise HTTPException(status_code=500, detail="API response missing task_id")
                 
            return response_json["output"]["task_id"]

        except (httpx.ConnectError, httpx.TimeoutException) as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt == max_retries - 1:
                print(f"Network Error Occurred: {type(e).__name__}: {e}")
                raise HTTPException(status_code=500, detail=f"Network Error: {str(e)}")
            await asyncio.sleep(1)
        except httpx.RequestError as e:
            print(f"Network Error Occurred: {type(e).__name__}: {e}")
            raise HTTPException(status_code=500, detail=f"Network Error: {str(e)}")

    # Should not reach here if logic is correct
    raise HTTPException(status_code=500, detail="Unknown error during API call")

async def call_wan_api_status(client: httpx.AsyncClient, task_id: str):
    url = f"{BASE_URL}/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    try:
        response = await client.get(url, headers=headers)
        return response.json()
    except httpx.RequestError as e:
        print(f"Status Check Network Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- API 路由 ---

@app.post("/api/generate")
async def generate_video(request: VideoGenerationRequest, req: Request):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API Key not configured on server.")
    
    # 从 app.state 获取全局 client
    client = req.app.state.http_client
    task_id = await call_wan_api_create(client, request)
    return {"task_id": task_id, "message": "Task created successfully"}

@app.get("/api/status/{task_id}")
async def check_status(task_id: str, req: Request):
    client = req.app.state.http_client
    result = await call_wan_api_status(client, task_id)
    
    output = result.get("output", {})
    status = output.get("task_status", "UNKNOWN")
    
    response_data = {
        "status": status,
        "progress_msg": "Processing...",
        "video_url": None,
        "usage": result.get("usage", {})
    }

    if status == "SUCCEEDED":
        response_data["video_url"] = output.get("video_url")
        response_data["progress_msg"] = "Completed"
    elif status == "FAILED":
        response_data["progress_msg"] = f"Failed: {output.get('message', 'Unknown error')}"
    
    return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)