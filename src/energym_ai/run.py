from __future__ import annotations
import argparse
import sys
import time
import cv2
from loguru import logger

from .core.pose_detector import PoseDetector
from .exercises.bicep_curl import BicepCurlAnalyzer
from .output.esp32_alert import create_alert_sender
from .output.cloud_sync import CloudSync
from .utils.config import load_config


def _draw_overlay(frame, analysis, fps: float) -> None:
    h, w = frame.shape[:2]
    color_ok = (0, 255, 0)
    color_bad = (0, 0, 255)
    color_info = (255, 255, 255)

    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Reps: {analysis.rep_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_ok, 2)
    cv2.putText(frame, f"State: {analysis.state.value}", (180, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_info, 2)
    if analysis.elbow_angle is not None:
        cv2.putText(frame, f"Elbow: {analysis.elbow_angle:.0f}°", (380, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_info, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_info, 2)

    if analysis.is_bad_form:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 0), -1)
        msg = "BAD FORM: " + ", ".join(analysis.form_issues)
        cv2.putText(frame, msg, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bad, 2)


def run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    ex_cfg = cfg["exercises"][args.exercise]

    cap = cv2.VideoCapture(cfg["camera"]["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    if not cap.isOpened():
        logger.error("Webcam tidak bisa dibuka.")
        return 1

    detector = PoseDetector(**cfg["mediapipe"])
    analyzer = BicepCurlAnalyzer(ex_cfg)
    alerter = create_alert_sender(cfg)
    cloud = CloudSync(cfg)

    logger.info(f"Mulai sesi: {ex_cfg['display_name']} | user={args.user_id} | station={args.station_id}")
    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Frame tidak terbaca, stop.")
                break

            pose = detector.process(frame)
            analysis = analyzer.analyze(pose)

            if analysis.is_bad_form:
                alerter.send_alert(level="warn", code=analysis.form_issues[0])

            detector.draw(frame, pose)
            _draw_overlay(frame, analysis, fps)
            cv2.imshow("EnerGym AI - Press 'q' to stop", frame)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_t))
            prev_t = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        summary = analyzer.session_summary()
        logger.info(f"Session summary: {summary}")
        cloud.push(user_id=args.user_id, station_id=args.station_id, summary=summary)

        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        alerter.close()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="EnerGym AI Pipeline Runner")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--exercise", default="bicep_curl",
                        choices=["bicep_curl", "hammer_curl"])
    parser.add_argument("--user-id", required=True, help="ID user dari mobile app")
    parser.add_argument("--station-id", default="STATION_01")
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
