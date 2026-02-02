"""Face recognition and dataset utilities for Sentinel."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import json
import shutil
import time


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
