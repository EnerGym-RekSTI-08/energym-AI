from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import requests
from loguru import logger


class CloudSync:
    def __init__(self, cfg: dict) -> None:
        cloud_cfg = cfg["output"]["cloud"]
        self.enabled: bool = cloud_cfg.get("enabled", True)
        self.endpoint: str = cloud_cfg["endpoint"]
        self.api_key: str = os.getenv(cloud_cfg.get("api_key_env", ""), "")
        self.fallback_dir = Path(cloud_cfg.get("fallback_local_dir", "./data/offline_queue"))
        self.fallback_dir.mkdir(parents=True, exist_ok=True)

    def _build_payload(
        self,
        user_id: str,
        station_id: str,
        summary: dict,
    ) -> dict:
        return {
            "session_id": f"{station_id}_{int(datetime.now(timezone.utc).timestamp())}",
            "user_id": user_id,
            "station_id": station_id,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }

    def push(self, user_id: str, station_id: str, summary: dict) -> bool:
        if not self.enabled:
            logger.info(f"[CloudSync OFF] summary={summary}")
            return True

        payload = self._build_payload(user_id, station_id, summary)
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                timeout=5.0,
            )
            resp.raise_for_status()
            logger.success(f"Cloud sync OK: session={payload['session_id']}")
            return True
        except Exception as e:
            logger.error(f"Cloud sync gagal, simpan offline: {e}")
            self._save_offline(payload)
            return False

    def _save_offline(self, payload: dict) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fpath = self.fallback_dir / f"queue_{ts}_{payload['session_id']}.json"
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def retry_offline(self) -> int:
        if not self.enabled:
            return 0
        success = 0
        for fpath in sorted(self.fallback_dir.glob("queue_*.json")):
            try:
                with fpath.open() as f:
                    payload = json.load(f)
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                    timeout=5.0,
                )
                resp.raise_for_status()
                fpath.unlink()
                success += 1
            except Exception as e:
                logger.warning(f"Retry {fpath.name} masih gagal: {e}")
                break
        return success
