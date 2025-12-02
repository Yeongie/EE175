"""
Real-time object detection using webcam with position tracking
Uses modular position_tracker module
Press 'q' to quit
"""

import cv2
from ultralytics import YOLO
from position_tracker import PositionTracker

def main():
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Using nano model for speed
    
    # Initialize position tracker
    tracker = PositionTracker(show_crosshair=True, show_coords=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting detection with position tracking. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        # Draw YOLO results on frame
        annotated_frame = results[0].plot()
        
        # Process detections and add position tracking
        positions = tracker.process_detections(results, annotated_frame, model)
        
        # Display the frame
        cv2.imshow('Object Detection with Position Tracking', annotated_frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()

