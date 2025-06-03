from ultralytics import YOLO
import cv2
import os
import time
from datetime import datetime
import numpy as np

# --- Configuration ---
# Get the absolute path of the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to save captured frames
SAVED_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'saved_images')
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)

# Path to save the output images
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'output_images')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Choose a YOLOv8 model
MODEL_NAME = 'yolov8n-seg.pt'

# Detection settings
CONFIDENCE_THRESHOLD = 0.5 # Lower confidence threshold for more detections
IMAGE_SIZE = 416  # Standard size for faster processing

# Custom colors for different classes (BGR format)
COLORS = {
    'person': (0, 255, 0),    # Green
    'car': (255, 0, 0),       # Blue
    'dog': (0, 0, 255),       # Red
    'cat': (255, 255, 0),     # Cyan
    'default': (0, 255, 255)  # Yellow
}

def draw_custom_box(img, box, label, conf, color):
    # Get box coordinates
    x1, y1, x2, y2 = map(int, box)
    
    # Draw thinner box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Prepare label text
    label_text = f"{label} {conf:.2f}"
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    
    # Draw label background
    cv2.rectangle(img, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
    
    # Draw label text in black and bold
    cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

print(f"--- Starting Real-time Object Detection with YOLOv8 ---")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Model: {MODEL_NAME}")

try:
    # 1. Load a pre-trained YOLOv8 model
    print(f"\nLoading model '{MODEL_NAME}'...")
    model = YOLO(MODEL_NAME)
    print("Model loaded successfully.")

    # 2. Initialize the camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        raise Exception("Could not open camera")

    print("Camera initialized successfully.")
    print("\nPress 'q' to quit, 's' to save current frame")

    frame_count = 0
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame for better detection
        resized_frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        # Perform YOLO detection on the frame
        results = model(resized_frame, conf=CONFIDENCE_THRESHOLD)
        
        # Process the results
        for r in results:
            # Create a copy of the original frame for drawing
            annotated_frame = frame.copy()
            
            # Draw custom boxes for each detection
            if len(r.boxes) > 0:
                print("\n--- Current Detections ---")
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    # Get color for this class
                    color = COLORS.get(class_name, COLORS['default'])
                    
                    # Get box coordinates and scale them back to original frame size
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Scale coordinates back to original frame size
                    scale_x = frame.shape[1] / IMAGE_SIZE
                    scale_y = frame.shape[0] / IMAGE_SIZE
                    x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                    y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                    
                    # Draw custom box
                    draw_custom_box(annotated_frame, (x1, y1, x2, y2), class_name, confidence, color)
                    
                    print(f"- Detected: {class_name} (Confidence: {confidence:.2f})")
            
            # Display the frame with detections
            cv2.imshow("Real-time Object Detection", annotated_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit if 'q' is pressed
            break
        elif key == ord('s'):  # Save frame if 's' is pressed
            # Generate timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_filename = f"frame_{timestamp}.jpg"
            output_filename = f"OP_frame_{timestamp}.jpg"
            
            # Save original frame
            frame_path = os.path.join(SAVED_IMAGES_FOLDER, frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"\nSaved original frame to: {frame_path}")
            
            # Save annotated frame
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved detected frame to: {output_path}")
            
            frame_count += 1

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Object Detection Finished ---")
    print(f"Total frames saved: {frame_count}")

except Exception as e:
    print(f"An error occurred: {e}")
    # Make sure to release the camera if there's an error
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows() 