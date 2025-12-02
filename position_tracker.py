"""
Position Tracker Module
Extracts and visualizes object positions from YOLO detections
"""

import cv2
import numpy as np


class PositionTracker:
    """Tracks and visualizes object positions from YOLO detection results"""
    
    def __init__(self, show_crosshair=True, show_coords=True):
        """
        Initialize position tracker
        
        Args:
            show_crosshair: Whether to draw crosshair at center point
            show_coords: Whether to display coordinate text on frame
        """
        self.show_crosshair = show_crosshair
        self.show_coords = show_coords
    
    def extract_position(self, box):
        """
        Extract position information from a YOLO detection box
        
        Args:
            box: YOLO detection box object
            
        Returns:
            dict: Position information including center, bbox, class, confidence
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        
        # Calculate center point
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # Get class and confidence
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Calculate bounding box dimensions
        width = int(x2 - x1)
        height = int(y2 - y1)
        area = width * height
        
        return {
            'center': (center_x, center_y),
            'bbox': (int(x1), int(y1), int(x2), int(y2)),
            'width': width,
            'height': height,
            'area': area,
            'class_id': cls,
            'confidence': conf
        }
    
    def draw_position(self, frame, position_info, class_name):
        """
        Draw position markers and text on frame
        
        Args:
            frame: Image frame to draw on
            position_info: Position dict from extract_position()
            class_name: Name of the detected object class
        """
        center_x, center_y = position_info['center']
        x1, y1, x2, y2 = position_info['bbox']
        
        # Draw crosshair at center
        if self.show_crosshair:
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Green dot
            cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)  # Green crosshair
        
        # Draw coordinate text
        if self.show_coords:
            coord_text = f"Center: ({center_x}, {center_y})"
            text_x = x1
            text_y = y1 - 35  # Position above the bounding box
            
            # Add background rectangle for readability
            (text_width, text_height), _ = cv2.getTextSize(
                coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(frame, (text_x, text_y - text_height - 5), 
                         (text_x + text_width, text_y + 5), (0, 255, 0), -1)
            cv2.putText(frame, coord_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def process_detections(self, results, frame, model):
        """
        Process all detections and draw position information
        
        Args:
            results: YOLO detection results
            frame: Image frame to draw on
            model: YOLO model (for class names)
            
        Returns:
            list: List of position_info dicts for all detections
        """
        positions = []
        
        for box in results[0].boxes:
            # Extract position
            pos_info = self.extract_position(box)
            class_name = model.names[pos_info['class_id']]
            
            # Draw on frame
            self.draw_position(frame, pos_info, class_name)
            
            # Add class name to info
            pos_info['class_name'] = class_name
            positions.append(pos_info)
            
            # Print to console
            print(f"Detected: {class_name} ({pos_info['confidence']:.2f}) | "
                  f"Center: {pos_info['center']} | "
                  f"BBox: [{pos_info['bbox'][0]}, {pos_info['bbox'][1]}, "
                  f"{pos_info['bbox'][2]}, {pos_info['bbox'][3]}]")
        
        return positions



