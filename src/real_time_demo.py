import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from inference import predict_mask
import time

def real_time_mask_detection():
    """Run real-time mask detection using webcam."""
    try:
        # Check if model exists
        model_path = "models/mobilenetv3_mask_detection.pth"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Please run train.py first!")
            return
        
        print("Loading model...")
        model = models.mobilenet_v3_small(pretrained=False)
        num_classes = 2
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Load face cascade with your custom path
        cascade_path = "C:/Users/Bart/Desktop/Projects/Face_mask_detection/haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            print("❌ Haar cascade file not found!")
            print("Please ensure haarcascade_frontalface_default.xml is in the correct path")
            return
            
        faceCascade = cv.CascadeClassifier(cascade_path)
        
        # Start webcam
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
        
        # Set camera properties for better performance
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv.CAP_PROP_FPS, 30)
        
        print("✅ Starting real-time detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read frame from webcam!")
                break
            
            frame_count += 1
            
            # Calculate FPS every frame for smooth updates
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            
            # Better face detection parameters
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Apply histogram equalization for better detection
            gray = cv.equalizeHist(gray)
            
            # Detect faces with improved parameters
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,        # How much the image size is reduced at each scale
                minNeighbors=5,         # How many neighbors each candidate rectangle should retain
                minSize=(50, 50),       # Minimum possible face size
                maxSize=(300, 300),     # Maximum possible face size
                flags=cv.CASCADE_SCALE_IMAGE
            )
            
            # Filter overlapping detections
            if len(faces) > 1:
                # Remove overlapping detections
                faces = filter_overlapping_faces(faces, overlap_threshold=0.3)
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Add padding around face for better detection
                padding = 20
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(frame.shape[1], x + w + padding)
                y_end = min(frame.shape[0], y + h + padding)
                
                # Extract face region with padding
                face_roi = frame[y_start:y_end, x_start:x_end]
                
                # Skip if face is too small
                if face_roi.shape[0] < 50 or face_roi.shape[1] < 50:
                    continue
                
                try:
                    # Get prediction for this face
                    prediction, confidence = predict_mask(face_roi, model, device)
                    
                    #q Color coding and confidence thresholding
                    if prediction == "Mask":
                        color = (0, 255, 0)  # Green for mask
                        text_color = (0, 255, 0)
                    elif prediction == "No Mask":
                        color = (0, 0, 255)  # Red for no mask
                        text_color = (0, 0, 255)
                    else:  # Uncertain
                        color = (0, 255, 255)  # Yellow for uncertain
                        text_color = (0, 255, 255)
                    
                    # Draw rectangle around face
                    thickness = 3 if confidence > 0.8 else 2
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Add prediction text with better formatting
                    text = f"{prediction}: {confidence:.2f}"
                    font_scale = 0.7
                    font_thickness = 2
                    
                    # Get text size for background rectangle
                    (text_width, text_height), _ = cv.getTextSize(
                        text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                    )
                    
                    # Draw background rectangle for text
                    cv.rectangle(frame, (x, y-text_height-10), 
                               (x+text_width, y), color, -1)
                    
                    # Add text
                    cv.putText(frame, text, (x, y-5), 
                              cv.FONT_HERSHEY_SIMPLEX, font_scale, 
                              (255, 255, 255), font_thickness)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    # Draw red rectangle for error
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv.putText(frame, "Error", (x, y-10), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add permanent FPS counter in top left with color coding
            fps_text = f"FPS: {fps:.1f}"
            
            # Color coding based on FPS value
            if fps < 10:
                fps_color = (0, 0, 255)      # Red for < 10 FPS
            elif 11 <= fps <= 14:
                fps_color = (0, 255, 255)    # Yellow for 11-14 FPS
            else:  # fps >= 15
                fps_color = (0, 255, 0)      # Green for >= 15 FPS
            
            # Get text size for background rectangle
            font_scale = 0.7
            font_thickness = 2
            (fps_text_width, fps_text_height), baseline = cv.getTextSize(
                fps_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw semi-transparent background rectangle for FPS
            overlay = frame.copy()
            cv.rectangle(overlay, (5, 5), (fps_text_width + 15, fps_text_height + 15), (0, 0, 0), -1)
            cv.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw FPS text
            cv.putText(frame, fps_text, (10, fps_text_height + 10), 
                      cv.FONT_HERSHEY_SIMPLEX, font_scale, fps_color, font_thickness)
            
            
            # Add instructions
            cv.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0]-10), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv.imshow("Face Mask Detection", frame)
            
            # Handle key presses
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"mask_detection_{timestamp}.jpg"
                cv.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv.destroyAllWindows()
        print("✅ Real-time detection ended.")
        
    except Exception as e:
        print(f"❌ Real-time detection failed: {e}")
        import traceback
        traceback.print_exc()

def filter_overlapping_faces(faces, overlap_threshold=0.3):
    """Filter out overlapping face detections"""
    if len(faces) <= 1:
        return faces
    
    # Convert to list for easier manipulation
    faces_list = list(faces)
    filtered_faces = []
    
    # Sort by area (largest first)
    faces_list.sort(key=lambda f: f[2] * f[3], reverse=True)
    
    while faces_list:
        # Take the largest remaining face
        current_face = faces_list.pop(0)
        filtered_faces.append(current_face)
        
        # Remove faces that significantly overlap with current face
        remaining_faces = []
        for face in faces_list:
            if calculate_overlap(current_face, face) < overlap_threshold:
                remaining_faces.append(face)
        
        faces_list = remaining_faces
    
    return np.array(filtered_faces)

def calculate_overlap(face1, face2):
    """Calculate overlap ratio between two face rectangles"""
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    face1_area = w1 * h1
    face2_area = w2 * h2
    
    # Return overlap ratio (intersection over minimum area)
    min_area = min(face1_area, face2_area)
    return intersection_area / min_area if min_area > 0 else 0.0

if __name__ == "__main__":
    real_time_mask_detection()
