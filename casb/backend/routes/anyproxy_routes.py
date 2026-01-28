from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from ..utils.anyproxy_manager import anyproxy_manager

router = APIRouter(prefix="/anyproxy", tags=["anyproxy"])

class StartRequest(BaseModel):
    filename: str = "box-traffic-logs.json"

@router.post("/start")
async def start_proxy(request: StartRequest):
    try:
        await anyproxy_manager.start(request.filename)
        return {"status": "success", "message": f"Proxy started, logging to {request.filename}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"{type(e).__name__}: {str(e)}"}

@router.post("/stop")
async def stop_proxy():
    await anyproxy_manager.stop()
    return {"status": "success", "message": "Proxy stopped"}

@router.get("/status")
async def get_status():
    return {
        "status": "success",
        "isRunning": anyproxy_manager.is_running,
        "currentLogFile": anyproxy_manager.current_log_file
    }

@router.get("/logs/stream")
async def stream_logs():
    if not anyproxy_manager.is_running:
        raise HTTPException(status_code=400, detail="Proxy is not running")
    
    return StreamingResponse(
        anyproxy_manager.stream_logs(),
        media_type="text/event-stream"
    )
