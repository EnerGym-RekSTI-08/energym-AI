from __future__ import annotations
import asyncio
import threading
import uuid
from typing import Optional
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

from .core.pose_detector import PoseDetector
from .exercises.bicep_curl import BicepCurlAnalyzer
from .output.esp32_alert import create_alert_sender
from .output.supabase_sync import SupabaseSync
from .utils.config import load_config

app = FastAPI(title="EnerGym AI Server", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config()
active_sessions: dict[str, dict] = {}

class StartSessionRequest(BaseModel):
    user_id: str
    station_id: str
    exercise_id: str      
    exercise_name: str 
    workout_id: Optional[str] = None

class StopSessionRequest(BaseModel):
    session_id: str


# ─── Helper: jalankan pipeline AI di background thread ───────────────────────
def _run_pipeline(session_id: str, request: StartSessionRequest) -> None:
    session = active_sessions.get(session_id)
    if not session:
        return

    ex_cfg = cfg["exercises"].get("bicep_curl")  # default ke bicep_curl untuk prototipe
    detector = PoseDetector(**cfg["mediapipe"])
    analyzer = BicepCurlAnalyzer(ex_cfg)
    alerter = create_alert_sender(cfg)

    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])

    logger.info(f"[{session_id}] Pipeline started: user={request.user_id}")
    session["running"] = True

    while session.get("running"):
        ok, frame = cap.read()
        if not ok:
            break

        pose = detector.process(frame)
        analysis = analyzer.analyze(pose)

        # Trigger buzzer ESP32 jika bad form
        if analysis.is_bad_form:
            alerter.send_alert("warn", analysis.form_issues[0])

        # Kirim data ke semua WebSocket yang terhubung
        ws_message = {
            "type": "frame_update",
            "session_id": session_id,
            "rep_count": analysis.rep_count,
            "state": analysis.state.value,
            "elbow_angle": round(analysis.elbow_angle, 1) if analysis.elbow_angle else None,
            "is_bad_form": analysis.is_bad_form,
            "form_issues": analysis.form_issues,
        }
        session["latest_data"] = ws_message

        # Broadcast ke semua subscriber async (thread-safe)
        loop = session.get("event_loop")
        if loop and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                _broadcast(session_id, ws_message), loop
            )

    # Sesi selesai → simpan ke Supabase
    summary = analyzer.session_summary()
    session["summary"] = summary
    logger.info(f"[{session_id}] Pipeline stopped. Summary: {summary}")

    cap.release()
    detector.close()
    alerter.close()

    # Simpan ke Supabase
    syncer = SupabaseSync(cfg)
    syncer.push(
        user_id=request.user_id,
        station_id=request.station_id,
        exercise_id=request.exercise_id,
        workout_id=request.workout_id,
        summary=summary,
    )

    # Notif WebSocket bahwa sesi selesai
    done_msg = {"type": "session_ended", "session_id": session_id, **summary}
    loop = session.get("event_loop")
    if loop and not loop.is_closed():
        asyncio.run_coroutine_threadsafe(_broadcast(session_id, done_msg), loop)


async def _broadcast(session_id: str, message: dict) -> None:
    session = active_sessions.get(session_id)
    if not session:
        return
    dead = []
    for ws in session.get("subscribers", []):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        session["subscribers"].remove(ws)


# ─── HTTP Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(active_sessions)}


@app.post("/session/start")
async def start_session(request: StartSessionRequest):
    session_id = str(uuid.uuid4())[:8]
    loop = asyncio.new_event_loop()

    active_sessions[session_id] = {
        "request": request,
        "running": False,
        "subscribers": [],
        "latest_data": None,
        "summary": None,
        "event_loop": asyncio.get_event_loop(),
    }

    # Jalankan pipeline di background thread
    thread = threading.Thread(
        target=_run_pipeline,
        args=(session_id, request),
        daemon=True,
    )
    thread.start()
    active_sessions[session_id]["thread"] = thread

    logger.info(f"Session {session_id} started for user {request.user_id}")
    return {"session_id": session_id, "status": "started"}


@app.post("/session/stop")
async def stop_session(request: StopSessionRequest):
    session = active_sessions.get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")

    session["running"] = False
    logger.info(f"Session {request.session_id} stop requested.")
    return {"status": "stopping", "session_id": request.session_id}


@app.get("/session/{session_id}/status")
async def get_status(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    return session.get("latest_data") or {"type": "waiting"}


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session tidak ditemukan"})
        await websocket.close()
        return

    session["subscribers"].append(websocket)
    session["event_loop"] = asyncio.get_event_loop()
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            # Keep-alive: terima ping dari client
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        session["subscribers"].remove(websocket)
        logger.info(f"WebSocket disconnected: {session_id}")


def run_server():
    import uvicorn
    uvicorn.run(
        "energym_ai.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
