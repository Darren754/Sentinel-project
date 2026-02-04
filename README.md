# Sentinel-project

Mobile robot project with motion tracking, LED "eye," servo scanning, and optional LBPH face recognition. The main entry point is `sentinel.py`.

## Features
- Motion detection with servo tracking and LED feedback.
- Optional LBPH face recognition using OpenCV contrib.
- Dataset utilities: capture, scan, and train helpers.
- Simulation mode to run without Raspberry Pi hardware.

## Requirements
- Python 3.9+ recommended
- Optional Raspberry Pi hardware libraries (see Hardware setup)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python sentinel.py
```

## Usage
### Simulation (no Pi hardware)
```bash
python sentinel.py --simulate --duration 60
```

### Face recognition (requires trained model)
```bash
python sentinel.py --enable-face --model ./face_model.yml --labels ./face_labels.json
```

### Dataset structure
Store face images under `faces/<name>/<image>.jpg` (one folder per person). The capture tool writes this structure automatically.

### Dataset utilities (capture, train, recognition)
```bash
# 1) Capture faces (Pi camera) into faces/<name>/<image>.jpg
python sentinel.py capture-face --name Darren --count 25 --out ./faces

# 2) (Optional) scan for bad images and auto-move them into faces/_bad/...
python sentinel.py scan-faces --dataset ./faces --move-bad

# 3) Train LBPH model + labels from faces/<name>/*.jpg
python sentinel.py train-lbph --dataset ./faces --model-out ./face_model.yml --labels-out ./face_labels.json

# 4) Run recognition using the trained model
python sentinel.py --enable-face --model ./face_model.yml --labels ./face_labels.json
```

Backward-compatible capture flags:
```bash
python sentinel.py --capture-face Darren --capture-count 25 --capture-out ./faces
```

## Hardware setup (Raspberry Pi)
Install hardware-specific packages on the Pi (often via apt/pip):
- `picamera2`
- `rpi_ws281x`
- `adafruit-circuitpython-servokit`

These are imported only in hardware mode, so simulation can run on any machine.

## Project layout
- `sentinel.py`: main entry point.
- `src/`: supporting modules (legacy or experimental).

## Notes
- LBPH requires OpenCV contrib (`opencv-contrib-python`).
