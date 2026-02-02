#!/usr/bin/env python3
# /home/pi/droid/sentinel.py
"""
Sentinel Droid controller: motion tracking + LED eye + servo + optional LBPH face recognition
plus dataset utilities (capture / scan+move-bad / train).

Default run (motion only):
  python3 sentinel.py

Run with face recognition:
  python3 sentinel.py --enable-face --model ./face_model.yml --labels ./face_labels.json

Backward-compatible dataset capture (your command style):
  python3 sentinel.py --capture-face Darren --capture-count 25 --capture-out ./faces

Recommended dataset utilities:
  python3 sentinel.py capture-face --name Darren --count 25 --out ./faces
  python3 sentinel.py scan-faces --dataset ./faces --move-bad
  python3 sentinel.py train-lbph --dataset ./faces --model-out ./face_model.yml --labels-out ./face_labels.json

Simulation (no Pi hardware libs):
  python3 sentinel.py --simulate --duration 60

Notes:
- LBPH requires OpenCV "contrib" (cv2.face.LBPHFaceRecognizer_create).
- Picamera2 + rpi_ws281x + ServoKit are used only in hardware mode.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple

import argparse
import asyncio
import json
import logging
import random
import math
import os
import shutil
import time


# -----------------------
# CONFIG
# -----------------------
@dataclass(frozen=True)
class Config:
    # LEDs
    led_count: int = 8
    led_pin: int = 18
    led_brightness: int = 35
    eye_rgb: Tuple[int, int, int] = (255, 20, 10)

    # Servo (PCA9685)
    servo_ch: int = 0
    servo_min: int = 20
    servo_max: int = 160
    servo_neutral: int = 90
    servo_left: int = 130
    servo_right: int = 50

    move_step_deg: int = 2
    servo_tick_sec: float = 0.03

    # Motion detection
    motion_threshold: int = 6000
    cooldown_sec: float = 1.2
    return_to_idle_sec: float = 2.5
    process_pause_sec: float = 0.45
    camera_tick_sec: float = 0.12

    # LEDs
    led_tick_sec: float = 0.04

    # Face recognition (LBPH)
    face_check_interval_sec: float = 0.8
    face_threshold: float = 65.0  # lower is stricter


CFG = Config()


# -----------------------
# STATE MACHINE
# -----------------------
class State(Enum):
    IDLE = auto()
    PROCESS = auto()
    OBSERVE = auto()
    ACK = auto()
    ALERT = auto()


@dataclass
class Shared:
    state: State = State.IDLE
    last_motion_time: float = 0.0
    desired_observe_angle: int = CFG.servo_neutral

    pending_ack: bool = False
    last_seen_name: str = ""
    last_seen_conf: float = 999.0

    state_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    led_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# -----------------------
# PORTABLE ABSTRACTIONS
# -----------------------
class LedRing(Protocol):
    def set_all(self, rgb: Tuple[int, int, int]) -> None: ...
    def clear(self) -> None: ...


class ServoDriver(Protocol):
    def set_angle(self, angle: int) -> None: ...
    def close(self) -> None: ...


class MotionSensor(Protocol):
    def check_motion_direction(self) -> Tuple[bool, str, int, float]: ...
    def get_last_frame_rgb(self): ...
    def close(self) -> None: ...


# -----------------------
# HELPERS
# -----------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rgb_scaled(rgb: Tuple[int, int, int], intensity: float) -> Tuple[int, int, int]:
    r, g, b = rgb
    r = int(clamp(r * intensity, 0, 255))
    g = int(clamp(g * intensity, 0, 255))
    b = int(clamp(b * intensity, 0, 255))
    return r, g, b


async def set_state(shared: Shared, new_state: State) -> None:
    async with shared.state_lock:
        shared.state = new_state


async def get_state(shared: Shared) -> State:
    async with shared.state_lock:
        return shared.state


def angle_from_x_norm(x_norm: float) -> int:
    x_norm = float(clamp(x_norm, 0.0, 1.0))
    span = CFG.servo_right - CFG.servo_left
    target = CFG.servo_left + span * x_norm
    return int(clamp(target, CFG.servo_min, CFG.servo_max))


def build_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return logging.getLogger("sentinel")


# -----------------------
# LED TASK
# -----------------------
async def led_task(shared: Shared, ring: LedRing) -> None:
    t0 = time.time()
    while True:
        t = time.time() - t0
        s = await get_state(shared)

        if s == State.IDLE:
            period = 7.0
            x = (math.sin(2 * math.pi * (t / period)) + 1) / 2
            intensity = 0.10 + 0.10 * x
            ring.set_all(rgb_scaled(CFG.eye_rgb, intensity))

        elif s == State.PROCESS:
            phase = t % 2.4
            if phase < 0.25:
                ring.set_all(rgb_scaled(CFG.eye_rgb, 0.26))
            elif phase < 0.55:
                ring.set_all(rgb_scaled(CFG.eye_rgb, 0.18))
            elif phase < 0.80:
                ring.set_all(rgb_scaled(CFG.eye_rgb, 0.24))
            else:
                ring.set_all(rgb_scaled(CFG.eye_rgb, 0.16))

        elif s == State.OBSERVE:
            ring.set_all(rgb_scaled(CFG.eye_rgb, 0.22))

        elif s == State.ALERT:
            period = 3.5
            x = (math.sin(2 * math.pi * (t / period)) + 1) / 2
            intensity = 0.28 + 0.08 * x
            ring.set_all(rgb_scaled(CFG.eye_rgb, intensity))

        elif s == State.ACK:
            ring.set_all(rgb_scaled(CFG.eye_rgb, 0.05))
            await asyncio.sleep(0.12)
            ring.set_all(rgb_scaled(CFG.eye_rgb, 0.20))
            await asyncio.sleep(0.12)
            # Return to OBSERVE so the droid stays "watching" the recognized person.
            await set_state(shared, State.OBSERVE)

        await asyncio.sleep(CFG.led_tick_sec)


# -----------------------
# SERVO TASK (continuous tracking + pending ACK)
# -----------------------
def _step_towards(current: int, target: int, step: int) -> int:
    if current == target:
        return current
    if current < target:
        return min(target, current + step)
    return max(target, current - step)


async def servo_task(shared: Shared, servo: ServoDriver) -> None:
    current = int(clamp(CFG.servo_neutral, CFG.servo_min, CFG.servo_max))
    servo.set_angle(current)

    while True:
        s = await get_state(shared)

        if s in (State.OBSERVE, State.ALERT):
            # Always track latest desired angle.
            target = int(clamp(shared.desired_observe_angle, CFG.servo_min, CFG.servo_max))
            nxt = _step_towards(current, target, CFG.move_step_deg)
            if nxt != current:
                servo.set_angle(nxt)
                current = nxt

            if shared.pending_ack:
                shared.pending_ack = False
                await set_state(shared, State.ACK)

            now = time.time()
            if (now - shared.last_motion_time) > CFG.return_to_idle_sec:
                # drift back to neutral then idle
                await set_state(shared, State.IDLE)

        elif s == State.IDLE:
            target = int(clamp(CFG.servo_neutral, CFG.servo_min, CFG.servo_max))
            nxt = _step_towards(current, target, CFG.move_step_deg)
            if nxt != current:
                servo.set_angle(nxt)
                current = nxt

        await asyncio.sleep(CFG.servo_tick_sec)


# -----------------------
# FACE AUTH (LBPH)
# -----------------------
class FaceAuth:
    def __init__(self, model_path: Path, labels_path: Path, threshold: float) -> None:
        import cv2  # type: ignore

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model: {model_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels: {labels_path}")

        if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
            raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python (cv2.face.*).")

        self._cv2 = cv2
        self._thr = float(threshold)

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            raise FileNotFoundError(f"Missing Haar cascade: {cascade_path}")
        self._cascade = cv2.CascadeClassifier(str(cascade_path))

        self._rec = cv2.face.LBPHFaceRecognizer_create()
        self._rec.read(str(model_path))

        self._id_to_name: Dict[int, str] = {int(k): v for k, v in json.loads(labels_path.read_text()).items()}

    def recognize(self, frame_rgb) -> Tuple[Optional[str], float]:
        """Return (name|None, distance)."""
        cv2 = self._cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        faces = self._cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None, 999.0

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (200, 200))

        label_id, dist = self._rec.predict(roi)
        name = self._id_to_name.get(int(label_id))
        if name and float(dist) <= self._thr:
            return name, float(dist)
        return None, float(dist)


# -----------------------
# CAMERA TASK (motion + optional face)
# -----------------------
async def camera_task(
    shared: Shared,
    sensor: MotionSensor,
    face: Optional[FaceAuth],
    *,
    unknown_alert: bool,
    log: logging.Logger,
) -> None:
    last_face_check = 0.0

    while True:
        s = await get_state(shared)

        if s in (State.IDLE, State.OBSERVE, State.ALERT):
            triggered, direction, strength, x_norm = sensor.check_motion_direction()

            # Presence refresh: keeps OBSERVE alive even if cooldown blocks "triggered".
            if strength > CFG.motion_threshold:
                shared.last_motion_time = time.time()

            if triggered:
                shared.desired_observe_angle = angle_from_x_norm(x_norm)

                if s == State.IDLE:
                    await set_state(shared, State.PROCESS)
                    await asyncio.sleep(CFG.process_pause_sec)
                    await set_state(shared, State.OBSERVE)

            # Face checks only when there's meaningful motion/presence
            now = time.time()
            if face and strength > CFG.motion_threshold and (now - last_face_check) >= CFG.face_check_interval_sec:
                last_face_check = now
                frame = sensor.get_last_frame_rgb()
                if frame is not None:
                    name, dist = face.recognize(frame)
                    if name:
                        shared.last_seen_name = name
                        shared.last_seen_conf = dist
                        shared.pending_ack = True
                        log.info("FACE recognized=%s dist=%.2f", name, dist)
                    else:
                        log.info("FACE unknown dist=%.2f", dist)
                        if unknown_alert:
                            await set_state(shared, State.ALERT)

        await asyncio.sleep(CFG.camera_tick_sec)


# -----------------------
# DATASET: capture / scan / train
# -----------------------
def _haar_cascade(cv2_mod):
    cascade_path = Path(cv2_mod.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.exists():
        raise FileNotFoundError(f"Missing Haar cascade: {cascade_path}")
    return cv2_mod.CascadeClassifier(str(cascade_path))


def capture_face_dataset(
    *,
    name: str,
    count: int,
    out_dir: Path,
    face_size: int = 200,
    settle_ms: int = 300,
    min_area: int = 1200,
) -> None:
    """
    Captures `count` cropped face images into: out_dir/name/###.jpg
    Trigger: automatic once you run the command and place your face in view.
    """
    import cv2  # type: ignore
    from picamera2 import Picamera2  # type: ignore

    out_dir = out_dir.expanduser().resolve()
    person_dir = out_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    cascade = _haar_cascade(cv2)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()
    try:
        time.sleep(settle_ms / 1000.0)

        saved = 0
        last_save = 0.0
        print(f"Capturing {count} face images for '{name}' into {person_dir}")
        print("Tip: vary angle + distance slightly. Press Ctrl+C to stop.")

        while saved < count:
            frame = picam2.capture_array()
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                continue

            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            if (w * h) < min_area:
                continue

            now = time.time()
            if now - last_save < 0.18:
                continue

            roi = frame[y : y + h, x : x + w]
            roi = cv2.resize(roi, (face_size, face_size))

            saved += 1
            last_save = now

            p = person_dir / f"{saved:03d}.jpg"
            bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(p), bgr)
            print(f"[{saved:02d}/{count}] saved {p.name}")

        print("Done.")
    finally:
        try:
            picam2.stop()
        except Exception:
            pass


def _scan_one_image(cv2_mod, cascade, img_path: Path) -> Tuple[str, int]:
    """Return (status, face_count): status in {'ok','unreadable','no_face','multi_face'}."""
    img = cv2_mod.imread(str(img_path))
    if img is None:
        return "unreadable", 0
    gray = cv2_mod.cvtColor(img, cv2_mod.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return "no_face", 0
    if len(faces) > 1:
        return "multi_face", len(faces)
    return "ok", 1


def scan_faces_dataset(dataset_dir: Path, *, move_bad: bool) -> int:
    """
    Scans dataset_dir/<person>/* for face detectability.
    If move_bad=True, moves bad images under dataset_dir/_bad/<reason>/<person>/...
    Returns number of bad images found.
    """
    import cv2  # type: ignore

    dataset_dir = dataset_dir.expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    cascade = _haar_cascade(cv2)

    bad = 0
    total = 0
    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "_bad"):
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            total += 1
            status, nfaces = _scan_one_image(cv2, cascade, img_path)
            if status == "ok":
                continue

            bad += 1
            print(f"BAD [{status}] {img_path} (faces={nfaces})")

            if move_bad:
                dst = dataset_dir / "_bad" / status / person_dir.name
                dst.mkdir(parents=True, exist_ok=True)
                shutil.move(str(img_path), str(dst / img_path.name))

    print(f"Scan complete. total={total} bad={bad} move_bad={move_bad}")
    return bad


def train_lbph(dataset_dir: Path, model_out: Path, labels_out: Path) -> None:
    """
    Train LBPH from dataset_dir/<person>/*.jpg|png.
    Writes:
      - model_out (yml)
      - labels_out (json mapping id->name)
    """
    import cv2  # type: ignore

    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError("OpenCV face module missing. Install opencv-contrib-python (cv2.face.*).")

    dataset_dir = dataset_dir.expanduser().resolve()
    model_out = model_out.expanduser().resolve()
    labels_out = labels_out.expanduser().resolve()

    cascade = _haar_cascade(cv2)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    images = []
    labels = []
    id_to_name: Dict[int, str] = {}
    name_to_id: Dict[str, int] = {}

    total = 0
    used = 0

    for person_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name != "_bad"):
        person = person_dir.name
        if person not in name_to_id:
            pid = len(name_to_id)
            name_to_id[person] = pid
            id_to_name[pid] = person

        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            roi = gray[y : y + h, x : x + w]
            roi = cv2.resize(roi, (200, 200))

            images.append(roi)
            labels.append(name_to_id[person])
            used += 1

    if used < 4:
        raise RuntimeError(f"Not enough usable face samples ({used}). Add more clear photos per person.")

    recognizer.train(images, labels)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(model_out))
    labels_out.write_text(json.dumps({str(k): v for k, v in id_to_name.items()}, indent=2))

    print("Trained LBPH.")
    print(f"  total scanned: {total}")
    print(f"  usable:       {used}")
    print(f"  people:       {', '.join(sorted(name_to_id.keys()))}")
    print(f"  model:        {model_out}")
    print(f"  labels:       {labels_out}")


# -----------------------
# HARDWARE / SIM BACKENDS
# -----------------------
class SimLedRing:
    def set_all(self, rgb: Tuple[int, int, int]) -> None:
        return

    def clear(self) -> None:
        return


class SimServo:
    def set_angle(self, angle: int) -> None:
        return

    def close(self) -> None:
        return


class SimSensor:
    def __init__(self) -> None:
        self._phase = 0.0

    def check_motion_direction(self) -> Tuple[bool, str, int, float]:
        self._phase += 0.2
        x = (math.sin(self._phase) + 1) / 2
        strength = 8000 if random.random() < 0.4 else 0  # type: ignore[name-defined]
        triggered = strength > 0
        direction = "left" if x < 0.33 else ("right" if x > 0.66 else "center")
        return triggered, direction, strength, float(x)

    def get_last_frame_rgb(self):
        return None

    def close(self) -> None:
        return


class Ws281xRing:
    def __init__(self) -> None:
        from rpi_ws281x import PixelStrip, Color  # type: ignore

        self._Color = Color
        self._strip = PixelStrip(
            CFG.led_count,
            CFG.led_pin,
            freq_hz=800000,
            dma=10,
            invert=False,
            brightness=CFG.led_brightness,
            channel=0,
        )
        self._strip.begin()

    def set_all(self, rgb: Tuple[int, int, int]) -> None:
        r, g, b = rgb
        c = self._Color(r, g, b)
        for i in range(self._strip.numPixels()):
            self._strip.setPixelColor(i, c)
        self._strip.show()

    def clear(self) -> None:
        self.set_all((0, 0, 0))


class Pca9685Servo:
    def __init__(self) -> None:
        from adafruit_servokit import ServoKit  # type: ignore

        self._kit = ServoKit(channels=16)

    def set_angle(self, angle: int) -> None:
        angle = int(clamp(angle, CFG.servo_min, CFG.servo_max))
        self._kit.servo[CFG.servo_ch].angle = angle

    def close(self) -> None:
        return


class PiCameraMotionSensor:
    def __init__(self) -> None:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from picamera2 import Picamera2  # type: ignore

        self._cv2 = cv2
        self._np = np

        self._last_frame = None
        self._last_trigger = 0.0
        self._prev_gray = None

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
        self.picam2.start()

    def get_last_frame_rgb(self):
        return self._last_frame

    def check_motion_direction(self) -> Tuple[bool, str, int, float]:
        cv2 = self._cv2
        frame = self.picam2.capture_array()
        self._last_frame = frame
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False, "center", 0, 0.5

        delta = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        motion_score = int(self._np.sum(thresh))
        now = time.time()

        if motion_score <= CFG.motion_threshold:
            return False, "center", motion_score, 0.5

        cooldown_active = (now - self._last_trigger) <= CFG.cooldown_sec

        res = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = res[0] if len(res) == 2 else res[1]
        if not contours:
            return (not cooldown_active), "center", motion_score, 0.5

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return (not cooldown_active), "center", motion_score, 0.5

        x, y, ww, hh = cv2.boundingRect(largest)
        cx = x + ww / 2.0

        if cx < w / 3:
            direction = "left"
        elif cx > 2 * w / 3:
            direction = "right"
        else:
            direction = "center"

        x_norm = float(clamp(cx / w, 0.0, 1.0))
        if cooldown_active:
            return False, direction, motion_score, x_norm

        self._last_trigger = now
        return True, direction, motion_score, x_norm

    def close(self) -> None:
        try:
            self.picam2.stop()
        except Exception:
            pass


def build_system(simulate: bool) -> Tuple[LedRing, ServoDriver, MotionSensor, bool]:
    if simulate:
        return SimLedRing(), SimServo(), SimSensor(), True
    return Ws281xRing(), Pca9685Servo(), PiCameraMotionSensor(), False


# -----------------------
# MAIN RUNNER
# -----------------------
async def run_sentinel(*, simulate: bool, duration: Optional[float], enable_face: bool,
                       model: Path, labels: Path, face_threshold: float, unknown_alert: bool,
                       log_level: str) -> None:
    log = build_logger(log_level)
    shared = Shared()

    ring, servo, sensor, sim_active = build_system(simulate)
    if sim_active:
        log.info("Simulation mode active.")

    face = None
    if enable_face and not sim_active:
        face = FaceAuth(model, labels, threshold=face_threshold)
        log.info("Face recognition enabled threshold=%.1f", face_threshold)

    tasks = [
        asyncio.create_task(led_task(shared, ring), name="led_task"),
        asyncio.create_task(servo_task(shared, servo), name="servo_task"),
        asyncio.create_task(camera_task(shared, sensor, face, unknown_alert=unknown_alert, log=log), name="camera_task"),
    ]

    try:
        if duration is None:
            await asyncio.gather(*tasks)
        else:
            await asyncio.sleep(duration)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        try:
            ring.clear()
        except Exception:
            pass

        try:
            servo.set_angle(CFG.servo_neutral)
        except Exception:
            pass

        try:
            servo.close()
        except Exception:
            pass

        try:
            sensor.close()
        except Exception:
            pass


# -----------------------
# CLI
# -----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sentinel Droid controller")
    p.add_argument("--simulate", action="store_true", help="Run without Pi hardware libs.")
    p.add_argument("--duration", type=float, default=None, help="Run for N seconds then exit.")
    p.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")

    p.add_argument("--enable-face", action="store_true", help="Enable LBPH recognition during runtime.")
    p.add_argument("--model", default="./face_model.yml", help="LBPH model file.")
    p.add_argument("--labels", default="./face_labels.json", help="Labels json file.")
    p.add_argument("--face-threshold", type=float, default=CFG.face_threshold, help="LBPH distance threshold (lower=stiffer).")
    p.add_argument("--unknown-alert", action="store_true", help="Set ALERT state when face is unknown.")

    sub = p.add_subparsers(dest="cmd", required=False)

    cap = sub.add_parser("capture-face", help="Capture face dataset for one person (Pi camera)")
    cap.add_argument("--name", required=True)
    cap.add_argument("--count", type=int, default=25)
    cap.add_argument("--out", required=True)

    scan = sub.add_parser("scan-faces", help="Scan dataset and optionally move bad images")
    scan.add_argument("--dataset", required=True)
    scan.add_argument("--move-bad", action="store_true")

    tr = sub.add_parser("train-lbph", help="Train LBPH model from dataset")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--model-out", required=True)
    tr.add_argument("--labels-out", required=True)

    return p


def dispatch_cli() -> None:
    import sys

    raw = sys.argv[1:]

    # Backwards-compatible flags (your command style)
    if "--capture-face" in raw:
        i = raw.index("--capture-face")
        name = raw[i + 1] if i + 1 < len(raw) else ""
        if not name:
            raise SystemExit("Missing value for --capture-face NAME")

        count = 25
        out = "./faces"
        if "--capture-count" in raw:
            j = raw.index("--capture-count")
            count = int(raw[j + 1])
        if "--capture-out" in raw:
            j = raw.index("--capture-out")
            out = raw[j + 1]

        capture_face_dataset(name=name, count=count, out_dir=Path(out))
        return

    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cmd == "capture-face":
        capture_face_dataset(name=args.name, count=args.count, out_dir=Path(args.out))
        return

    if args.cmd == "scan-faces":
        scan_faces_dataset(Path(args.dataset), move_bad=bool(args.move_bad))
        return

    if args.cmd == "train-lbph":
        train_lbph(Path(args.dataset), Path(args.model_out), Path(args.labels_out))
        return

    asyncio.run(
        run_sentinel(
            simulate=bool(args.simulate),
            duration=args.duration,
            enable_face=bool(args.enable_face),
            model=Path(args.model).expanduser(),
            labels=Path(args.labels).expanduser(),
            face_threshold=float(args.face_threshold),
            unknown_alert=bool(args.unknown_alert),
            log_level=str(args.log_level),
        )
    )


if __name__ == "__main__":
    try:
        dispatch_cli()
    except KeyboardInterrupt:
        pass
