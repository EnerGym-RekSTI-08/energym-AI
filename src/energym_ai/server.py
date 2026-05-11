from __future__ import annotations
import asyncio
import os
import threading
import time
import uuid
from typing import Optional
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from loguru import logger

from .core.pose_detector import PoseDetector
from .exercises.bicep_curl import BicepCurlAnalyzer
from .exercises.alternating_curl import AlternatingCurlAnalyzer
from .exercises.hammer_curl import HammerCurlAnalyzer
from .output.esp32_alert import create_alert_sender
from .output.supabase_sync import SupabaseSync
from .output.station_status import StationStatusManager
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
latest_frame: bytes | None = None
latest_frame_ts: float | None = None
main_event_loop: asyncio.AbstractEventLoop | None = None

_station_code = os.getenv("ENERGYM_STATION_ID", "STATION_01")
station_status = StationStatusManager(_station_code)

# Mapping exercise_name → (config_key, AnalyzerClass)
_EXERCISE_MAP: dict[str, tuple[str, type]] = {
    "Bicep Curl": ("bicep_curl", BicepCurlAnalyzer),
    "bicep_curl": ("bicep_curl", BicepCurlAnalyzer),
    "Alternating Dumbbell Curl": ("alternating_curl", AlternatingCurlAnalyzer),
    "alternating_curl": ("alternating_curl", AlternatingCurlAnalyzer),
    "Hammer Curl": ("hammer_curl", HammerCurlAnalyzer),
    "hammer_curl": ("hammer_curl", HammerCurlAnalyzer),
}

# ── Shared camera & detector (warmup system) ──────────────────────────
_shared_lock = threading.Lock()
_shared_cap: cv2.VideoCapture | None = None
_shared_detector: PoseDetector | None = None
_shared_warmup_ts: float | None = None      # waktu terakhir warmup / dipakai
_IDLE_TIMEOUT_SEC = 300                      # auto-release setelah 5 menit idle


def _warmup_camera() -> tuple[cv2.VideoCapture, PoseDetector]:
    global _shared_cap, _shared_detector, _shared_warmup_ts
    with _shared_lock:
        # Detector
        if _shared_detector is None:
            logger.info("[warmup] Initializing PoseDetector...")
            _shared_detector = PoseDetector(**cfg["mediapipe"])
            logger.info("[warmup] PoseDetector ready.")

        # Camera
        if _shared_cap is None or not _shared_cap.isOpened():
            logger.info("[warmup] Opening camera...")
            _shared_cap = cv2.VideoCapture(cfg["camera"]["source"])
            _shared_cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
            _shared_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
            # Baca 1 frame dummy untuk flush buffer internal driver
            _shared_cap.read()
            logger.info("[warmup] Camera ready.")

        _shared_warmup_ts = time.time()
        return _shared_cap, _shared_detector


def _release_shared() -> None:
    global _shared_cap, _shared_detector, _shared_warmup_ts
    with _shared_lock:
        if _shared_cap is not None:
            _shared_cap.release()
            _shared_cap = None
            logger.info("[warmup] Camera released.")
        if _shared_detector is not None:
            _shared_detector.close()
            _shared_detector = None
            logger.info("[warmup] Detector released.")
        _shared_warmup_ts = None


def _idle_watcher() -> None:
    while True:
        time.sleep(30)
        with _shared_lock:
            if (
                _shared_warmup_ts
                and _shared_cap is not None
                and not active_sessions
                and (time.time() - _shared_warmup_ts) > _IDLE_TIMEOUT_SEC
            ):
                pass  # will release below
            else:
                continue
        # Release di luar lock untuk hindari deadlock
        _release_shared()
        logger.info("[warmup] Idle timeout → resources released.")


threading.Thread(target=_idle_watcher, daemon=True).start()


@app.on_event("startup")
async def _on_startup() -> None:
    global main_event_loop
    main_event_loop = asyncio.get_running_loop()
    threading.Thread(target=_warmup_camera, daemon=True).start()
    threading.Thread(target=station_status.on_server_start, daemon=True).start()
    logger.info("[startup] Warmup triggered in background.")


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    threading.Thread(target=station_status.on_server_stop, daemon=False).start()


def _safe_broadcast(session_id: str, message: dict) -> None:
    session = active_sessions.get(session_id)
    if not session:
        return
    loop = session.get("event_loop") or main_event_loop
    if not loop or loop.is_closed():
        logger.warning(f"[{session_id}] Cannot broadcast: event loop not available")
        return
    try:
        asyncio.run_coroutine_threadsafe(_broadcast(session_id, message), loop)
    except RuntimeError as exc:
        logger.exception(f"[{session_id}] Broadcast failed: {exc}")


class StartSessionRequest(BaseModel):
    user_id: str
    station_id: str
    exercise_id: str
    exercise_name: str
    workout_id: Optional[str] = None


class StopSessionRequest(BaseModel):
    session_id: str


def _run_pipeline(session_id: str, request: StartSessionRequest) -> None:
    global latest_frame, latest_frame_ts, _shared_warmup_ts
    session = active_sessions.get(session_id)
    if not session:
        return

    # Resolve exercise analyzer dari exercise_name
    mapping = _EXERCISE_MAP.get(request.exercise_name)
    if mapping is None:
        mapping = _EXERCISE_MAP.get("bicep_curl")
        logger.warning(
            f"[{session_id}] exercise_name '{request.exercise_name}' tidak dikenal, "
            f"fallback ke bicep_curl"
        )

    config_key, AnalyzerClass = mapping
    ex_cfg = cfg["exercises"].get(config_key)
    if ex_cfg is None:
        logger.error(f"[{session_id}] Config '{config_key}' tidak ditemukan")
        session["running"] = False
        session["error"] = "config_not_found"
        _safe_broadcast(session_id, {"type": "error", "message": f"Config '{config_key}' tidak ditemukan"})
        return

    # Pakai shared camera + detector (sudah pre-warmed)
    cap, detector = _warmup_camera()
    analyzer = AnalyzerClass(ex_cfg)
    alerter = create_alert_sender(cfg)
    logger.info(f"[{session_id}] Using analyzer: {AnalyzerClass.__name__} (config: {config_key})")

    if not cap.isOpened():
        logger.error(f"[{session_id}] Kamera tidak bisa dibuka")
        session["running"] = False
        session["error"] = "camera_not_opened"
        _safe_broadcast(session_id, {"type": "error", "message": "Kamera tidak bisa dibuka"})
        return

    logger.info(f"[{session_id}] Pipeline started: user={request.user_id}")
    session["running"] = True
    frame_interval = 0.0
    target_fps = cfg.get("camera", {}).get("target_fps", 0)
    if target_fps and target_fps > 0:
        frame_interval = 1.0 / float(target_fps)
    read_failures = 0
    processing_errors = 0

    while session.get("running"):
        loop_started = time.time()
        ok, frame = cap.read()
        if not ok:
            read_failures += 1
            logger.warning(f"[{session_id}] Gagal baca frame ({read_failures})")
            if read_failures >= 10:
                session["error"] = "camera_read_failed"
                _safe_broadcast(session_id, {"type": "error", "message": "Gagal membaca frame kamera"})
                break
            time.sleep(0.05)
            continue

        read_failures = 0
        try:
            pose = detector.process(frame)

            draw_frame = frame.copy()
            detector.draw(draw_frame, pose)

            # Encode ke JPEG dan simpan ke global untuk /stream/snapshot
            _, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            latest_frame = buffer.tobytes()
            latest_frame_ts = time.time()

            analysis = analyzer.analyze(pose)

            if analysis.is_bad_form and analysis.form_issues:
                alerter.send_alert("warn", analysis.form_issues[0])

            ws_message = {
                "type": "frame_update",
                "session_id": session_id,
                "rep_count": analysis.rep_count,
                "bad_rep_count": getattr(analysis, "bad_rep_count", None),
                "state": analysis.state.value,
                "elbow_angle": round(analysis.elbow_angle, 1) if analysis.elbow_angle else None,
                "is_bad_form": analysis.is_bad_form,
                "form_issues": analysis.form_issues,
            }
            if hasattr(analysis, "active_arm"):
                ws_message["active_arm"] = analysis.active_arm
            session["latest_data"] = ws_message

            _safe_broadcast(session_id, ws_message)
            processing_errors = 0
        except Exception as exc:
            processing_errors += 1
            logger.exception(f"[{session_id}] Error pipeline: {exc}")
            if processing_errors >= 5:
                session["error"] = "pipeline_error"
                _safe_broadcast(session_id, {"type": "error", "message": "Pipeline error"})
                break

        if frame_interval > 0:
            elapsed = time.time() - loop_started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    summary = analyzer.session_summary()
    session["summary"] = summary
    logger.info(f"[{session_id}] Pipeline stopped. Summary: {summary}")

    alerter.close()
    latest_frame = None
    latest_frame_ts = None
    _shared_warmup_ts = time.time()  # reset idle timer

    station_status.on_session_stop()

    syncer = SupabaseSync(cfg)
    syncer.push(
        user_id=request.user_id,
        station_id=request.station_id,
        exercise_id=request.exercise_id,
        workout_id=request.workout_id,
        summary=summary,
    )

    done_msg = {"type": "session_ended", "session_id": session_id, **summary}
    _safe_broadcast(session_id, done_msg)


async def _broadcast(session_id: str, message: dict) -> None:
    session = active_sessions.get(session_id)
    if not session:
        return
    dead = []
    for ws in list(session.get("subscribers", [])):
        try:
            await ws.send_json(message)
        except Exception as exc:
            logger.warning(f"[{session_id}] WebSocket send failed: {exc}")
            dead.append(ws)
    for ws in dead:
        if ws in session["subscribers"]:
            session["subscribers"].remove(ws)


@app.get("/health")
def health():
    cam_ready = _shared_cap is not None and _shared_cap.isOpened()
    det_ready = _shared_detector is not None
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "camera_ready": cam_ready,
        "detector_ready": det_ready,
    }


@app.post("/camera/warmup")
async def warmup_camera():
    threading.Thread(target=_warmup_camera, daemon=True).start()
    return {"status": "warming_up"}


@app.post("/session/start")
async def start_session(request: StartSessionRequest):
    session_id = str(uuid.uuid4())[:8]
    active_sessions[session_id] = {
        "request": request,
        "running": False,
        "subscribers": [],
        "latest_data": None,
        "summary": None,
        "event_loop": asyncio.get_running_loop(),
        "error": None,
    }
    thread = threading.Thread(target=_run_pipeline, args=(session_id, request), daemon=True)
    thread.start()
    active_sessions[session_id]["thread"] = thread
    logger.info(f"Session {session_id} started for user {request.user_id}")
    station_status.on_session_start(workout_id=request.workout_id)
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


@app.get("/stream/snapshot")
async def snapshot():
    global latest_frame
    now = time.time()
    if latest_frame is None:
        # Placeholder frame saat webcam belum aktif
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Webcam belum aktif", (40, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        frame_data = buf.tobytes()
    elif latest_frame_ts and (now - latest_frame_ts) > 2.0:
        placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Frame timeout", (70, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        _, buf = cv2.imencode('.jpg', placeholder)
        frame_data = buf.tobytes()
    else:
        frame_data = latest_frame

    return Response(
        content=frame_data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
        }
    )


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = active_sessions.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": "Session tidak ditemukan"})
        await websocket.close()
        return

    session["subscribers"].append(websocket)
    session["event_loop"] = asyncio.get_running_loop()
    logger.info(f"WebSocket connected for session {session_id}")

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        if websocket in session.get("subscribers", []):
            session["subscribers"].remove(websocket)
        logger.info(f"WebSocket disconnected: {session_id}")


def run_server():
    import uvicorn
    uvicorn.run("energym_ai.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")


if __name__ == "__main__":
    run_server()