import cv2
from deepface import DeepFace
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np

# === Configuration ===
DB_PATH = "/home/aleema/facerecognition-realtime/database"
CSV_FILE = "/home/aleema/facerecognition-realtime/attendance.csv"
EXIT_TIMEOUT_SEC = 10
MIN_CONFIDENCE = 0.85  # Increased threshold for better accuracy
KNOWN_PERSON = "inzamam"  # The only person we want to recognize

# Face detection backends to test
DETECTORS = ['mtcnn', 'ssd', 'retinaface', 'yolov8', 'fastmtcnn']

# === State Tracking ===
attendance = {}
current_detections = set()
status_text = "Waiting for detection..."
last_status_change = datetime.now()

# === Initialize CSV ===
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Name", "Status", "Time", "Duration"]).to_csv(CSV_FILE, index=False)

# === Face Detection ===
def detect_faces(frame, detector_backend):
    try:
        face_objs = DeepFace.extract_faces(
            frame,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )
        return [f for f in face_objs if f['confidence'] > 0.95]  # Very high confidence
    except Exception as e:
        print(f"[{detector_backend.upper()} ERROR] {str(e)}")
        return []

# === Face Recognition ===
def recognize_face(face_img):
    try:
        df = DeepFace.find(
            face_img,
            db_path=DB_PATH,
            enforce_detection=False,
            model_name='Facenet',
            detector_backend='skip',  # We already detected faces
            silent=True,
            distance_metric='cosine'
        )
        
        if len(df) > 0 and len(df[0]) > 0:
            best_match = df[0].iloc[0]
            similarity = 1 - best_match['distance']
            if similarity >= MIN_CONFIDENCE:
                identity_path = best_match['identity']
                name = os.path.basename(os.path.dirname(identity_path))
                return name, similarity
        return "Unknown", 0
    except Exception as e:
        print(f"[RECOG ERROR] {str(e)}")
        return "Unknown", 0

# === Attendance Logging ===
def log_attendance(name, status, duration=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "Name": name,
        "Status": status,
        "Time": timestamp,
        "Duration": str(duration) if duration else ""
    }
    pd.DataFrame([entry]).to_csv(CSV_FILE, mode='a', header=False, index=False)
    print(f"[LOG] {status}: {name} at {timestamp}" + (f" (Duration: {duration})" if duration else ""))

# === Main Loop ===
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("[ERROR] Camera not accessible.")
    exit()

print("[INFO] System started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame read failed.")
            continue

        # Create a black sidebar for status info
        sidebar = np.zeros((frame.shape[0], 300, 3), dtype=np.uint8)
        
        current_detections.clear()
        current_detector = DETECTORS[0]  # Start with MTCNN
        
        faces = detect_faces(frame, current_detector)
        
        for face in faces:
            facial_area = face['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            # Crop with padding for better recognition
            padding = 30
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(frame.shape[1], x+w+padding), min(frame.shape[0], y+h+padding)
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                continue
                
            name, confidence = recognize_face(face_img)
            current_detections.add(name)
            
            # Only process known person
            if name == KNOWN_PERSON:
                # Draw bounding box and label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Update attendance
                now = datetime.now()
                if name not in attendance:
                    attendance[name] = {
                        "first_seen": now,
                        "last_seen": now,
                        "logged": False
                    }
                    status_text = f"{KNOWN_PERSON} is present since {now.strftime('%H:%M:%S')}"
                    last_status_change = now
                else:
                    attendance[name]["last_seen"] = now
                    duration = now - attendance[name]["first_seen"]
                    status_text = (f"{KNOWN_PERSON} is present\n"
                                 f"Since: {attendance[name]['first_seen'].strftime('%H:%M:%S')}\n"
                                 f"Duration: {str(duration).split('.')[0]}")
                    
                    # Log only once per session
                    if not attendance[name]["logged"]:
                        log_attendance(name, "PRESENT")
                        attendance[name]["logged"] = True
            else:
                # Draw red box for unknown
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Check for exits
        now = datetime.now()
        for name in list(attendance.keys()):
            if name not in current_detections:
                absence = (now - attendance[name]["last_seen"]).total_seconds()
                if absence > EXIT_TIMEOUT_SEC:
                    duration = attendance[name]["last_seen"] - attendance[name]["first_seen"]
                    log_attendance(name, "LEFT", duration)
                    del attendance[name]
                    status_text = f"{KNOWN_PERSON} left at {now.strftime('%H:%M:%S')}"
                    last_status_change = now

        # Display status in sidebar
        y_offset = 30
        for line in status_text.split('\n'):
            cv2.putText(sidebar, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 30
        
        # Add detector info
        cv2.putText(sidebar, f"Detector: {current_detector.upper()}", (10, y_offset+30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Combine frames
        combined = np.hstack((frame, sidebar))
        
        cv2.imshow("Attendance System", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Finalize any remaining sessions
    for name in attendance:
        duration = datetime.now() - attendance[name]["first_seen"]
        log_attendance(name, "SYSTEM CLOSED", duration)
    
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] System stopped. Attendance log saved.")

