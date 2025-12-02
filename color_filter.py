"""
Color Filter Module
Detects objects based on HSV color ranges
"""

import cv2
import numpy as np


class ColorFilter:
    """Filters and detects objects based on color using HSV color space"""
    
    # Predefined color ranges in HSV
    COLOR_RANGES = {
        'red': [
            # Red wraps around in HSV, so we need two ranges
            (np.array([0, 100, 100]), np.array([10, 255, 255])),      # Lower red
            (np.array([160, 100, 100]), np.array([180, 255, 255]))    # Upper red
        ],
        'blue': [
            (np.array([100, 100, 100]), np.array([130, 255, 255]))
        ],
        'green': [
            (np.array([40, 50, 50]), np.array([80, 255, 255]))
        ],
        'yellow': [
            (np.array([20, 100, 100]), np.array([30, 255, 255]))
        ],
        'orange': [
            (np.array([10, 100, 100]), np.array([20, 255, 255]))
        ],
        'purple': [
            (np.array([130, 50, 50]), np.array([160, 255, 255]))
        ],
        'white': [
            (np.array([0, 0, 200]), np.array([180, 30, 255]))
        ],
        'black': [
            (np.array([0, 0, 0]), np.array([180, 255, 50]))
        ]
    }
    
    def __init__(self, target_color='red', min_area=500):
        """
        Initialize color filter
        
        Args:
            target_color: Color to detect ('red', 'blue', 'green', etc.)
            min_area: Minimum contour area to consider (filters noise)
        """
        self.target_color = target_color.lower()
        self.min_area = min_area
        
        if self.target_color not in self.COLOR_RANGES:
            raise ValueError(f"Color '{target_color}' not supported. "
                           f"Available: {list(self.COLOR_RANGES.keys())}")
    
    def set_color(self, color):
        """Change target color"""
        self.target_color = color.lower()
        if self.target_color not in self.COLOR_RANGES:
            raise ValueError(f"Color '{color}' not supported. "
                           f"Available: {list(self.COLOR_RANGES.keys())}")
    
    def create_mask(self, frame):
        """
        Create a binary mask of pixels matching the target color
        
        Args:
            frame: BGR image frame
            
        Returns:
            Binary mask where white = target color detected
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color ranges for target color
        ranges = self.COLOR_RANGES[self.target_color]
        
        # Create mask (handle multiple ranges for colors like red)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        
        # Clean up mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
        
        return mask
    
    def find_colored_objects(self, frame, mask=None):
        """
        Find contours of colored objects in frame
        
        Args:
            frame: BGR image frame
            mask: Pre-computed mask (optional, will create if not provided)
            
        Returns:
            list: List of dicts with contour info (center, area, bbox, contour)
        """
        if mask is None:
            mask = self.create_mask(frame)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out small contours (noise)
            if area < self.min_area:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
            
            objects.append({
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, x + w, y + h),
                'contour': contour,
                'color': self.target_color
            })
        
        # Sort by area (largest first)
        objects.sort(key=lambda obj: obj['area'], reverse=True)
        
        return objects
    
    def draw_colored_objects(self, frame, objects, show_mask=False):
        """
        Draw detected colored objects on frame
        
        Args:
            frame: Image frame to draw on
            objects: List of object dicts from find_colored_objects()
            show_mask: Whether to overlay the color mask
            
        Returns:
            Annotated frame
        """
        output = frame.copy()
        
        # Optionally show mask overlay
        if show_mask:
            mask = self.create_mask(frame)
            colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            colored_mask[:, :, 0] = 0  # Remove blue channel
            colored_mask[:, :, 1] = 0  # Remove green channel
            output = cv2.addWeighted(output, 0.7, colored_mask, 0.3, 0)
        
        for obj in objects:
            cx, cy = obj['center']
            x1, y1, x2, y2 = obj['bbox']
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta box
            
            # Draw center point
            cv2.circle(output, (cx, cy), 8, (255, 0, 255), -1)
            cv2.drawMarker(output, (cx, cy), (255, 0, 255), 
                          cv2.MARKER_CROSS, 25, 2)
            
            # Add label
            label = f"{obj['color'].upper()} object"
            coord_text = f"Center: ({cx}, {cy})"
            
            # Draw label background
            text_y = y1 - 40
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(output, (x1, text_y - label_h - 5), 
                         (x1 + label_w, text_y + 5), (255, 0, 255), -1)
            cv2.putText(output, label, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw coordinates
            coord_y = y1 - 10
            (coord_w, coord_h), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, coord_y - coord_h - 5), 
                         (x1 + coord_w, coord_y + 5), (255, 0, 255), -1)
            cv2.putText(output, coord_text, (x1, coord_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output
    
    @staticmethod
    def get_available_colors():
        """Returns list of available color names"""
        return list(ColorFilter.COLOR_RANGES.keys())



