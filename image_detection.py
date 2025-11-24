"""
Object detection on static images
Usage: python image_detection.py <image_path>
"""

import cv2
from ultralytics import YOLO
import sys
import os

def main():
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python image_detection.py <image_path>")
        print("Example: python image_detection.py test_image.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image")
        return
    
    print(f"Loaded image: {image_path}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')
    
    # Run inference
    print("Running detection...")
    results = model(image, verbose=False)
    
    # Get detection information
    print(f"\nDetected {len(results[0].boxes)} objects:")
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  {i+1}. {class_name}: {conf:.2%} confidence at [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Draw results
    annotated_image = results[0].plot()
    
    # Save output
    output_path = "detected_" + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_image)
    print(f"\nSaved annotated image to: {output_path}")
    
    # Display result
    cv2.imshow('Object Detection', annotated_image)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

