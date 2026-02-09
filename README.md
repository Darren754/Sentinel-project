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
python sentinel.py --enable-face --model ./models/lbph.yml --labels ./models/labels.json
```

### Dataset structure
Store face images under `faces/<name>/<image>.jpg` (one folder per person). The capture tool writes this structure automatically.

### Dataset utilities (capture, train, recognition)
```bash
# 1) Capture faces (Pi camera) into faces/<name>/<image>.jpg
python sentinel.py capture-face --name Darren --count 25 --out ./faces
# (Optional) Skip auto-training:
# python sentinel.py capture-face --name Darren --count 25 --out ./faces --no-train

# 2) (Optional) scan for bad images and auto-move them into faces/_bad/...
python sentinel.py scan-faces --dataset ./faces --move-bad

# 3) Auto-training runs immediately after capture (writes ./models/lbph.yml + ./models/labels.json).
#    You can also train manually:
python sentinel.py train-lbph --dataset ./faces --model-out ./models/lbph.yml --labels-out ./models/labels.json

# 4) Run recognition using the trained model
python sentinel.py --enable-face --model ./models/lbph.yml --labels ./models/labels.json
```

Backward-compatible capture flags:
```bash
python sentinel.py --capture-face Darren --capture-count 25 --capture-out ./faces
# (Optional) skip auto-training:
# python sentinel.py --capture-face Darren --capture-count 25 --capture-out ./faces --no-train
```

## Raspberry Pi setup
### System packages (apt)
```bash
sudo apt update
sudo apt install -y python3-picamera2 libcamera-apps i2c-tools
```

### Enable I2C
```bash
sudo raspi-config
```
Navigate to **Interface Options** → **I2C** and enable it, then reboot if prompted.

### Python dependencies (pip)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pi.txt
```

### Hardware run command
```bash
sudo -E .venv/bin/python sentinel.py
```

### Hardware assumptions
- WS2812 data on GPIO18 (Pin 12)
- PCA9685 at I2C address `0x40`
- Camera uses picamera2/libcamera

These are imported only in hardware mode, so simulation can run on any machine.

## Project layout
- `sentinel.py`: main entry point.
- `src/`: supporting modules (legacy or experimental).

## Notes
- LBPH requires OpenCV contrib (`opencv-contrib-python`).
