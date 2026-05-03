from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


class SupabaseSync:
    def __init__(self, cfg: dict) -> None:
        cloud_cfg = cfg["output"]["cloud"]
        self.enabled: bool = cloud_cfg.get("enabled", True)

        url = os.getenv("SUPABASE_URL", cloud_cfg.get("supabase_url", ""))
        key = os.getenv("SUPABASE_SERVICE_KEY", "")

        self.fallback_dir = Path(cloud_cfg.get("fallback_local_dir", "./data/offline_queue"))
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

        self._client: Client | None = None
        if self.enabled and SUPABASE_AVAILABLE and url and key:
            try:
                self._client = create_client(url, key)
                logger.info("Supabase client initialized.")
            except Exception as e:
                logger.warning(f"Supabase init gagal: {e}")

    def push(
        self,
        user_id: str,
        station_id: str,
        exercise_id: str,
        summary: dict,
        workout_id: str | None = None,
    ) -> bool:
        if not self.enabled:
            logger.info(f"[SupabaseSync OFF] {summary}")
            return True

        payload = {
            "user_id": user_id,
            "workout_id": workout_id,
            "exercise_id": exercise_id,
            "station_id": station_id,
            "exercise_name": summary.get("exercise"),
            "duration_seconds": summary.get("duration_seconds"),
            "valid_reps": summary.get("valid_reps"),
            "bad_reps": summary.get("bad_reps"),
            "accuracy": summary.get("accuracy"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._client:
            try:
                self._client.table("workout_sessions").insert(payload).execute()
                logger.success(f"Supabase sync OK: user={user_id}, reps={summary.get('valid_reps')}")
                return True
            except Exception as e:
                logger.error(f"Supabase sync gagal: {e}")
                self._save_offline(payload)
                return False
        else:
            logger.warning("Supabase client tidak tersedia, simpan offline.")
            self._save_offline(payload)
            return False

    def _save_offline(self, payload: dict) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        fpath = self.fallback_dir / f"queue_{ts}.json"
        with fpath.open("w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Disimpan offline: {fpath.name}")
