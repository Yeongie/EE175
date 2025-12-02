"""
Real-time color-based object detection using webcam
Uses modular color_filter module
Press 'q' to quit, 'c' to cycle through colors, 'm' to toggle mask view
"""

import cv2
from color_filter import ColorFilter


def main():
    # Initialize color filter (default: red)
    available_colors = ColorFilter.get_available_colors()
    current_color_idx = 0  # Start with red
    color_filter = ColorFilter(target_color=available_colors[current_color_idx], min_area=500)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    show_mask = False
    print(f"Starting color detection. Target color: {available_colors[current_color_idx].upper()}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'c' - Cycle through colors")
    print("  'm' - Toggle mask view")
    print(f"Available colors: {', '.join(available_colors)}")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Find colored objects
        colored_objects = color_filter.find_colored_objects(frame)
        
        # Draw detected objects
        annotated_frame = color_filter.draw_colored_objects(frame, colored_objects, show_mask=show_mask)
        
        # Display info on frame
        info_text = f"Detecting: {available_colors[current_color_idx].upper()} | Found: {len(colored_objects)} objects"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Print detection info
        if colored_objects:
            for i, obj in enumerate(colored_objects[:3]):  # Show top 3
                cx, cy = obj['center']
                print(f"  {i+1}. {obj['color'].upper()} object | "
                      f"Center: ({cx}, {cy}) | Area: {obj['area']} px²")
        
        # Display the frame
        cv2.imshow('Color Detection', annotated_frame)
        
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
            # Toggle mask view
            show_mask = not show_mask
            print(f"Mask view: {'ON' if show_mask else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Color detection stopped.")


if __name__ == "__main__":
    main()



