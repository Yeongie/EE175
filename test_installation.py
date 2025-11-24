"""
Test script to verify OpenCV and YOLOv8 installation
Run this script to check if everything is working correctly
"""

import cv2
from ultralytics import YOLO
import numpy as np

def test_opencv():
    """Test OpenCV installation"""
    print("Testing OpenCV...")
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (10, 10), (90, 90), (0, 255, 0), 2)
    
    print("✓ OpenCV is working correctly!")
    return True

def test_yolov8():
    """Test YOLOv8 installation"""
    print("\nTesting YOLOv8...")
    
    try:
        # Load a pretrained YOLOv8 model
        model = YOLO('yolov8n.pt')  # nano model (smallest, fastest)
        print("✓ YOLOv8 model loaded successfully!")
        print(f"Model: {model.task}")
        
        # Create a dummy test image
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (50, 50), (590, 590), (255, 255, 255), -1)
        
        # Run inference
        results = model(test_image, verbose=False)
        print(f"✓ YOLOv8 inference completed!")
        print(f"Detected {len(results[0].boxes)} objects")
        
        return True
    except Exception as e:
        print(f"✗ YOLOv8 test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Computer Vision Setup")
    print("=" * 60)
    
    opencv_ok = test_opencv()
    yolov8_ok = test_yolov8()
    
    print("\n" + "=" * 60)
    if opencv_ok and yolov8_ok:
        print("✓ All tests passed! Your setup is ready.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. You can now use your webcam: python webcam_detection.py")
        print("2. Or detect objects in images: python image_detection.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("=" * 60)
        return False
    
    return True

if __name__ == "__main__":
    main()

