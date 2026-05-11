from __future__ import annotations
from abc import ABC, abstractmethod
from time import time
import json
from loguru import logger
import requests

class AlertSender(ABC):
    @abstractmethod
    def send_alert(self, level: str, code: str) -> None: ...
    @abstractmethod
    def send_rep(self, count: int) -> None: ...
    @abstractmethod
    def close(self) -> None: ...

class SupabaseAlertSender(AlertSender):
    def __init__(
        self, 
        project_url: str, 
        api_key: str, 
        station_id: str,
        cooldown_sec: float = 2.0
    ) -> None:
        # URL target ke tabel 'stations' dengan filter station_id
        self._url = f"{project_url}/rest/v1/stations?station_code=eq.{station_id}"
        self._headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal"
        }
        self._station_id = station_id
        self._cooldown = cooldown_sec
        self._last_alert: float = 0.0
        
        logger.info(f"SupabaseAlertSender initialized for Station: {station_id}")

    def send_alert(self, level: str, code: str) -> None:
        """Mengirim sinyal ke tabel stations untuk memicu aktuator ESP32"""
        now = time()
        if now - self._last_alert < self._cooldown:
            return

        # Kita update kolom actuator_status menjadi 'RINGING'
        # Kamu bisa ganti 'RINGING' jadi angka jika di ESP32 pakai logika bad_count
        payload = {"actuator_status": "RINGING"}

        try:
            response = requests.patch(self._url, json=payload, headers=self._headers)
            
            if response.status_code in [200, 201, 204]:
                self._last_alert = now
                logger.warning(f"IOT SIGNAL SENT → Station {self._station_id}: {code}")
            else:
                logger.error(f"Supabase Error {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.error(f"Gagal koneksi ke Supabase: {e}")

    def send_rep(self, count: int) -> None:
        # Kamu bisa tambahkan logic untuk update jumlah rep ke kolom lain jika ada
        pass

    def close(self) -> None:
        # Pastikan saat aplikasi ditutup, status aktuator kembali ke READY
        try:
            requests.patch(self._url, json={"actuator_status": "READY"}, headers=self._headers)
        except:
            pass

class SerialAlertSender(AlertSender):
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baudrate: int = 115200,
        cooldown_sec: float = 1.5,
    ) -> None:
        try:
            import serial
        except ImportError as e:
            raise RuntimeError("pyserial belum terinstall. `pip install pyserial`") from e

        self._serial = serial.Serial(port, baudrate, timeout=1.0)
        self._cooldown = cooldown_sec
        self._last_alert: float = 0.0
        logger.info(f"SerialAlertSender connected: {port} @ {baudrate}")

    def _write_json(self, payload: dict) -> None:
        line = (json.dumps(payload) + "\n").encode("utf-8")
        try:
            self._serial.write(line)
            self._serial.flush()
        except Exception as e:
            logger.error(f"Gagal kirim ke ESP32: {e}")

    def send_alert(self, level: str, code: str) -> None:
        now = time()
        if now - self._last_alert < self._cooldown:
            return
        self._last_alert = now
        self._write_json({"cmd": "alert", "level": level, "code": code})
        logger.warning(f"ALERT → ESP32: {level}/{code}")

    def send_rep(self, count: int) -> None:
        self._write_json({"cmd": "rep", "count": count})

    def close(self) -> None:
        try:
            self._write_json({"cmd": "stop"})
            self._serial.close()
        except Exception:
            pass


class NoOpAlertSender(AlertSender):
    def __init__(self) -> None:
        self._last_alert = 0.0

    def send_alert(self, level: str, code: str) -> None:
        if time() - self._last_alert > 1.0:
            logger.warning(f"[MOCK ALERT] {level}/{code}")
            self._last_alert = time()

    def send_rep(self, count: int) -> None:
        logger.debug(f"[MOCK REP] count={count}")

    def close(self) -> None:
        pass


def create_alert_sender(cfg: dict) -> AlertSender:
    esp_cfg = cfg["output"]["esp32"]
    if not esp_cfg.get("enabled", False):
        return NoOpAlertSender()

    mode = esp_cfg.get("mode", "serial")
    
    if mode == "serial":
        # ... (kode serial yang lama) ...
        return SerialAlertSender(...) 
        
    elif mode == "supabase":
        return SupabaseAlertSender(
            project_url=cfg["supabase"]["url"],
            api_key=cfg["supabase"]["key"],
            station_id=cfg.get("station_id", "STATION_01"), # Ambil dari config atau default
            cooldown_sec=esp_cfg.get("alert_cooldown_sec", 2.0)
        )
    
    return NoOpAlertSender()
