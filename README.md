# Pick and Place Robotic Arm - Computer Vision Setup

This project implements computer vision capabilities for a pick and place robotic arm using OpenCV and YOLOv8.

## Installation

The project uses a Python virtual environment to manage dependencies.

### Initial Setup

1. **Activate the virtual environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies** (if not already installed):
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```powershell
   python test_installation.py
   ```

## Project Structure

```
EE175/
├── venv/                      # Virtual environment
├── test_installation.py       # Installation verification script
├── webcam_detection.py        # Real-time webcam object detection
├── image_detection.py         # Static image object detection
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Usage

### Test Installation

Run the test script to verify everything is working:
```powershell
python test_installation.py
```

### Webcam Detection

For real-time object detection using your webcam:
```powershell
python webcam_detection.py
```
- Press 'q' to quit

### Image Detection

For detecting objects in a static image:
```powershell
python image_detection.py <image_path>
```

Example:
```powershell
python image_detection.py test_image.jpg
```

## What's Working

✅ **OpenCV** - Computer vision library for image processing  
✅ **YOLOv8** - State-of-the-art object detection model  
✅ **Real-time detection** - Webcam-based live detection  
✅ **Static image detection** - Process images offline  

## Next Steps

For your pick and place robot:

1. **Train custom YOLOv8 model** on specific objects you want to pick
2. **Calibrate camera** to get accurate object positions
3. **Add depth estimation** for 3D positioning (using stereo vision or depth camera)
4. **Integrate with Arduino** to send coordinates to servo motors
5. **Implement inverse kinematics** for the robotic arm

## Notes

- The YOLOv8n model (nano) is used for speed - faster for real-time detection
- For better accuracy, consider using yolov8m.pt or yolov8l.pt
- When training custom models, you'll need to collect and label your own dataset
- Consider using a USB camera or better webcam for consistent results

## Dependencies

- Python 3.13.5
- OpenCV 4.12.0
- Ultralytics YOLOv8 8.3.223
- PyTorch 2.9.0
- NumPy, Matplotlib, Pillow

## Troubleshooting

**Issue: Permission denied when activating venv**
- Run PowerShell as Administrator
- Or run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Issue: Webcam not working**
- Check if another application is using the webcam
- Try changing the camera index in webcam_detection.py (currently set to 0)

**Issue: Slow detection**
- The nano model (yolov8n.pt) is already optimized for speed
- Consider reducing camera resolution or frame rate

