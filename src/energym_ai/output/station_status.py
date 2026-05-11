from __future__ import annotations
import os
import threading
import time
from datetime import datetime, timezone
from loguru import logger

try:
    from supabase import create_client, Client
    from dotenv import load_dotenv
    load_dotenv()
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class StationStatusManager:
    HEARTBEAT_INTERVAL = 15  # seconds

    def __init__(self, station_code: str) -> None:
        self.station_code = station_code
        self._client: Client | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        if not _AVAILABLE:
            logger.warning("[StationStatus] supabase-py not installed — station status updates disabled.")
            return

        url = os.getenv("SUPABASE_URL", "")
        key = os.getenv("SUPABASE_SERVICE_KEY", "")
        if not url or not key:
            logger.warning("[StationStatus] SUPABASE_URL/SUPABASE_SERVICE_KEY missing — updates disabled.")
            return

        try:
            self._client = create_client(url, key)
            logger.info(f"[StationStatus] Initialized for station '{station_code}'.")
        except Exception as e:
            logger.warning(f"[StationStatus] Supabase init failed: {e}")

    # ── Public lifecycle methods ──────────────────────────────────────────────

    def on_server_start(self) -> None:
        """Call once when the FastAPI server starts. Marks station online."""
        self._update({"status": "online", "current_workout_id": None})
        self._start_heartbeat()
        logger.info(f"[StationStatus] Station '{self.station_code}' marked online.")

    def on_server_stop(self) -> None:
        """Call once when the FastAPI server shuts down. Marks station offline."""
        self._stop_event.set()
        self._update({"status": "offline", "current_workout_id": None})
        logger.info(f"[StationStatus] Station '{self.station_code}' marked offline.")

    def on_session_start(self, workout_id: str | None = None) -> None:
        """Call when an AI session begins. Marks station busy."""
        self._update({"status": "busy", "current_workout_id": workout_id})

    def on_session_stop(self) -> None:
        """Call when an AI session ends. Marks station online (idle)."""
        self._update({"status": "online", "current_workout_id": None})

    def update_hardware_status(
        self,
        webcam: str | None = None,
        esp32: str | None = None,
        actuator: str | None = None,
    ) -> None:
        """Update hardware component statuses."""
        payload: dict = {}
        if webcam is not None:
            payload["webcam_status"] = webcam
        if esp32 is not None:
            payload["esp32_status"] = esp32
        if actuator is not None:
            payload["actuator_status"] = actuator
        if payload:
            self._update(payload)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _update(self, fields: dict) -> None:
        if self._client is None:
            return
        fields["last_sync"] = datetime.now(timezone.utc).isoformat()
        try:
            self._client.table("stations") \
                .update(fields) \
                .eq("station_code", self.station_code) \
                .execute()
        except Exception as e:
            logger.warning(f"[StationStatus] Update failed: {e}")

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.HEARTBEAT_INTERVAL):
            self._update({})  # only updates last_sync
            logger.debug(f"[StationStatus] Heartbeat sent for '{self.station_code}'.")
