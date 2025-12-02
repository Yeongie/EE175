"""
Combined YOLO + Color Detection
Uses both position_tracker and color_filter modules
Perfect for robotic arm applications - detects objects AND filters by color
Press 'q' to quit, 'c' to cycle colors, 'm' to toggle modes
"""

import cv2
from ultralytics import YOLO
from position_tracker import PositionTracker
from color_filter import ColorFilter


def main():
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    # Initialize modules
    tracker = PositionTracker(show_crosshair=True, show_coords=True)
    available_colors = ColorFilter.get_available_colors()
    current_color_idx = 0  # Start with red
    color_filter = ColorFilter(target_color=available_colors[current_color_idx], min_area=500)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Detection modes
    modes = ['YOLO Only', 'Color Only', 'Both']
    current_mode_idx = 2  # Start with both
    
    print(f"Starting combined detection. Mode: {modes[current_mode_idx]}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Cycle through colors (for color detection)")
    print("  'm' - Toggle detection mode (YOLO/Color/Both)")
    print(f"Available colors: {', '.join(available_colors)}")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        mode = modes[current_mode_idx]
        annotated_frame = frame.copy()
        
        # YOLO Detection
        yolo_positions = []
        if mode in ['YOLO Only', 'Both']:
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            yolo_positions = tracker.process_detections(results, annotated_frame, model)
        
        # Color Detection
        colored_objects = []
        if mode in ['Color Only', 'Both']:
            colored_objects = color_filter.find_colored_objects(frame)
            annotated_frame = color_filter.draw_colored_objects(annotated_frame, colored_objects)
        
        # Display mode and statistics
        info_text = f"Mode: {mode} | Target Color: {available_colors[current_color_idx].upper()}"
        stats_text = f"YOLO: {len(yolo_positions)} | Color: {len(colored_objects)}"
        
        # Draw info background
        cv2.rectangle(annotated_frame, (5, 5), (600, 70), (0, 0, 0), -1)
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, stats_text, (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display the frame
        cv2.imshow('Combined Detection', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Cycle to next color
            current_color_idx = (current_color_idx + 1) % len(available_colors)
            color_filter.set_color(available_colors[current_color_idx])
            print(f"\nSwitched to detecting: {available_colors[current_color_idx].upper()}")
        elif key == ord('m'):
            # Cycle through modes
            current_mode_idx = (current_mode_idx + 1) % len(modes)
            print(f"\nMode changed to: {modes[current_mode_idx]}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Combined detection stopped.")


if __name__ == "__main__":
    main()



