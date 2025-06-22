import cv2
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import logging
# logging.getLogger('deepface').setLevel(logging.ERROR)

# Paths
DB_PATH = "path/to/face_database"  # Change to your face database path
CSV_FILE = "/path/to/attendance.csv"  # Change to your desired path

# Parameters
EXIT_TIMEOUT_SEC = 10

# Attendance tracking
attendance = {}
last_seen = {}

# Initialize CSV
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
    df.to_csv(CSV_FILE, index=False)
else:
    df = pd.read_csv(CSV_FILE)

# Start camera
cap = cv2.VideoCapture(0)  # Change index if needed

print("[INFO] Starting attendance system...")
# frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        res = DeepFace.find(
            frame, 
            db_path=DB_PATH,
            enforce_detection=False,
            model_name='Facenet',
            detector_backend='opencv'
        )

        if len(res) > 0 and len(res[0]) > 0:
            # Best match
            best_match = res[0].iloc[0]
            identity_path = best_match['identity']
            # Extract name from path
            name = identity_path.split('/')[-2]

            # Coordinates
            xmin = int(best_match['source_x'])
            ymin = int(best_match['source_y'])
            xmax = int(xmin + best_match['source_w'])
            ymax = int(ymin + best_match['source_h'])

            # Draw bounding box + label
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            # Attendance mark
            if name not in attendance:
                entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                attendance[name] = {"entry": entry_time, "exit": None}
                print(f"[INFO] Entry marked: {name} at {entry_time}")

            last_seen[name] = datetime.now()

        # Handle exit
        now = datetime.now()
        for name in list(attendance.keys()):
            if name in last_seen:
                seconds_since_seen = (now - last_seen[name]).total_seconds()
                if seconds_since_seen > EXIT_TIMEOUT_SEC and attendance[name]["exit"] is None:
                    exit_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    attendance[name]["exit"] = exit_time
                    print(f"[INFO] Exit marked: {name} at {exit_time}")

                    # Save to CSV
                    df = pd.read_csv(CSV_FILE)
                    df = pd.concat([df, pd.DataFrame([{
                        "Name": name,
                        "Entry Time": attendance[name]["entry"],
                        "Exit Time": attendance[name]["exit"]
                    }])], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)

                    # Remove from tracking
                    del attendance[name]
                    del last_seen[name]

    except Exception as e:
        print("[WARN]", e)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



