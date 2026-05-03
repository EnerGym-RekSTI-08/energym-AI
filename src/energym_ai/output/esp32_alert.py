from __future__ import annotations
from abc import ABC, abstractmethod
from time import time
import json
from loguru import logger


class AlertSender(ABC):
    @abstractmethod
    def send_alert(self, level: str, code: str) -> None: ...
    @abstractmethod
    def send_rep(self, count: int) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


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
        try:
            return SerialAlertSender(
                port=esp_cfg["serial_port"],
                baudrate=esp_cfg["baudrate"],
                cooldown_sec=esp_cfg.get("alert_cooldown_sec", 1.5),
            )
        except Exception as e:
            logger.warning(f"Serial gagal, fallback ke NoOp: {e}")
            return NoOpAlertSender()
    else:
        # TODO: implement HTTP / MQTT bila dibutuhkan
        logger.warning(f"Mode {mode} belum diimplementasi, pakai NoOp.")
        return NoOpAlertSender()
