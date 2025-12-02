# Modular Vision System for Robotic Arm Control

This project uses a **modular architecture** (like C++ header/implementation files) where each feature is separated into its own module for better organization and reusability.

## 📁 File Structure

```
EE175/
├── position_tracker.py           # Module: Position tracking from YOLO detections
├── color_filter.py               # Module: HSV color-based object detection
├── webcam_detection.py           # Script: YOLO detection with position tracking
├── webcam_color_detection.py    # Script: Pure color-based detection
├── webcam_combined_detection.py # Script: YOLO + Color detection combined
├── image_detection.py            # Script: Static image detection
└── yolov8n.pt                    # YOLO model file
```

---

## 🔧 Modules (Like C++ .h/.cpp files)

### 1. `position_tracker.py`
**Purpose:** Extract and visualize object positions from YOLO detections

**Key Features:**
- Extracts bounding box coordinates
- Calculates center points
- Draws crosshairs and coordinate labels
- Provides position data for robotics control

**Main Class:** `PositionTracker`

**Usage Example:**
```python
from position_tracker import PositionTracker

tracker = PositionTracker(show_crosshair=True, show_coords=True)
positions = tracker.process_detections(results, frame, model)
```

**Returns:** List of dicts with:
- `center`: (x, y) center coordinates
- `bbox`: (x1, y1, x2, y2) bounding box
- `width`, `height`, `area`: Object dimensions
- `class_name`, `confidence`: Detection info

---

### 2. `color_filter.py`
**Purpose:** Detect objects based on HSV color filtering

**Key Features:**
- Detects objects by color (red, blue, green, yellow, etc.)
- Uses HSV color space for robust color detection
- Filters out noise with morphological operations
- Provides position data for colored objects

**Main Class:** `ColorFilter`

**Supported Colors:**
- `red`, `blue`, `green`, `yellow`, `orange`, `purple`, `white`, `black`

**Usage Example:**
```python
from color_filter import ColorFilter

color_filter = ColorFilter(target_color='red', min_area=500)
objects = color_filter.find_colored_objects(frame)
annotated = color_filter.draw_colored_objects(frame, objects)
```

**Returns:** List of dicts with:
- `center`: (x, y) center coordinates
- `bbox`: (x1, y1, x2, y2) bounding box
- `area`: Object area in pixels
- `contour`: OpenCV contour data
- `color`: Detected color name

---

## 🎬 Scripts (Like C++ main.cpp files)

### 1. `webcam_detection.py`
**What it does:** YOLO object detection with position tracking
**Uses modules:** `position_tracker.py`
**Best for:** General object detection (person, car, ball, etc.)

**Run:**
```bash
python webcam_detection.py
```

**Controls:**
- `q` - Quit

**Output:**
- Green crosshairs on detected objects
- Center coordinates displayed on screen
- Console output with position data

---

### 2. `webcam_color_detection.py`
**What it does:** Pure color-based object detection
**Uses modules:** `color_filter.py`
**Best for:** Detecting objects by color (red ball, blue block, etc.)

**Run:**
```bash
python webcam_color_detection.py
```

**Controls:**
- `q` - Quit
- `c` - Cycle through colors
- `m` - Toggle mask view (shows what the filter sees)

**Output:**
- Magenta boxes around colored objects
- Center coordinates and labels
- Console output with position and area

---

### 3. `webcam_combined_detection.py`
**What it does:** YOLO + Color detection together
**Uses modules:** `position_tracker.py` + `color_filter.py`
**Best for:** Robotic arm applications (detect AND filter by color)

**Run:**
```bash
python webcam_combined_detection.py
```

**Controls:**
- `q` - Quit
- `c` - Cycle through target colors
- `m` - Toggle modes (YOLO Only / Color Only / Both)

**Modes:**
1. **YOLO Only** - Standard object detection
2. **Color Only** - Pure color filtering
3. **Both** - Combined detection (best for robotics!)

**Output:**
- Combined visualization with both detection types
- Statistics for both YOLO and color detection
- Position data from both sources

---

## 🤖 For Robotic Arm Control

### Recommended Workflow:

1. **Use `webcam_combined_detection.py` in "Both" mode**
   - YOLO identifies what the object is (ball, cup, etc.)
   - Color filter confirms it's the right color (red ball)
   - Position tracker gives you precise (x, y) coordinates

2. **Get position data:**
```python
# From YOLO + position_tracker
positions = tracker.process_detections(results, frame, model)
for pos in positions:
    if pos['class_name'] == 'sports ball':
        center = pos['center']  # (x, y)
        print(f"Ball at: {center}")

# From color_filter
colored_objects = color_filter.find_colored_objects(frame)
for obj in colored_objects:
    if obj['color'] == 'red':
        center = obj['center']  # (x, y)
        print(f"Red object at: {center}")
```

3. **Next steps (when you get the robot arm):**
   - Add `depth_estimator.py` module for distance (z-axis)
   - Add `inverse_kinematics.py` module for servo angle calculation
   - Add `servo_controller.py` module for hardware control

---

## 🎨 How Color Detection Works

Color detection uses **HSV (Hue, Saturation, Value)** color space instead of RGB because:
- **HSV is more robust** to lighting changes
- **Hue** represents the actual color (red, blue, green)
- **Saturation** represents color intensity
- **Value** represents brightness

### Color Ranges (HSV):
- **Red**: Hue 0-10 and 160-180 (red wraps around)
- **Blue**: Hue 100-130
- **Green**: Hue 40-80
- **Yellow**: Hue 20-30

The filter:
1. Converts frame to HSV
2. Creates a binary mask (white = color detected)
3. Cleans up noise with morphological operations
4. Finds contours (object boundaries)
5. Filters by minimum area
6. Returns position data

---

## 📊 Position Data Format

Both modules return position information in a consistent format:

```python
{
    'center': (320, 240),           # Center (x, y) in pixels
    'bbox': (100, 80, 540, 400),    # Bounding box (x1, y1, x2, y2)
    'area': 176000,                  # Object area in pixels²
    'class_name': 'sports ball',    # Object type (YOLO only)
    'color': 'red',                  # Color name (color_filter only)
    'confidence': 0.89               # Detection confidence (YOLO only)
}
```

Use this data to:
- Track object position over time
- Calculate movement/velocity
- Send to inverse kinematics solver
- Control robotic arm servos

---

## 🚀 Quick Start

**Test YOLO detection:**
```bash
python webcam_detection.py
```

**Test color detection (red ball):**
```bash
python webcam_color_detection.py
# Press 'c' to cycle to other colors if needed
```

**Test combined (best for robotics):**
```bash
python webcam_combined_detection.py
# Press 'm' to switch between YOLO/Color/Both
# Press 'c' to change target color
```

---

## 💡 Tips

1. **For red ball detection:** Use color filter with 'red', or combined mode
2. **Lighting matters:** Color detection works best with good, even lighting
3. **Adjust min_area:** Change `min_area` parameter to filter out noise
4. **Custom colors:** Edit `COLOR_RANGES` in `color_filter.py` to add custom colors
5. **Combine both:** Use YOLO to identify + color filter to confirm

---

## 🔮 Future Modules (When you get hardware)

- `depth_estimator.py` - Distance calculation (z-axis)
- `camera_calibration.py` - Pixel-to-real-world conversion
- `inverse_kinematics.py` - Calculate servo angles from (x,y,z)
- `servo_controller.py` - Send commands to robot arm
- `trajectory_planner.py` - Plan smooth motion paths
- `gripper_controller.py` - Open/close gripper

Each will be its own module, following the same C++-style organization!



