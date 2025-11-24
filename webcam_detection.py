"""
Real-time object detection using webcam
Press 'q' to quit
"""

import cv2
from ultralytics import YOLO

def main():
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')  # Using nano model for speed
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting detection. Press 'q' to quit.")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Run YOLOv8 inference
        results = model(frame, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Display the frame
        cv2.imshow('Object Detection', annotated_frame)
        
        # Print detection info
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            print(f"Detected: {class_name} ({conf:.2f})")
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()

