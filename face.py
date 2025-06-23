import cv2
from deepface import DeepFace
from datetime import datetime
import pandas as pd
import os
import time

# === Configuration ===
DB_PATH = "F:/CCTV ATTENDANCE/Fatima Saud Work/facerecognition-realtime/database"
CSV_FILE = "F:/CCTV ATTENDANCE/Fatima Saud Work/facerecognition-realtime/attendance.csv"
EXIT_TIMEOUT_SEC = 10

# === State Tracking ===
attendance = {}
last_seen = {}

# === Load or Create CSV ===
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
    df.to_csv(CSV_FILE, index=False)

# === Open Camera ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Camera not accessible.")
    exit()

print("[INFO] Attendance system started...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame.")
            break

        try:
            results = DeepFace.find(
                frame,
                db_path=DB_PATH,
                enforce_detection=False,
                model_name='Facenet',
                detector_backend='opencv'
            )

            if len(results) > 0 and len(results[0]) > 0:
                best_match = results[0].iloc[0]
                identity_path = best_match['identity']
                name = identity_path.split(os.path.sep)[-2]

                xmin = int(best_match['source_x'])
                ymin = int(best_match['source_y'])
                xmax = int(xmin + best_match['source_w'])
                ymax = int(ymin + best_match['source_h'])

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                now = datetime.now()

                # Entry
                if name not in attendance:
                    entry_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    attendance[name] = {"entry": entry_time, "exit": None}
                    print(f"[INFO] Entry marked: {name} at {entry_time}")
                
                last_seen[name] = now

        except Exception as e:
            print("[ERROR]", str(e))

        # Exit Logic
        now = datetime.now()
        for name in list(attendance.keys()):
            if name in last_seen:
                seconds_absent = (now - last_seen[name]).total_seconds()
                if seconds_absent > EXIT_TIMEOUT_SEC and attendance[name]["exit"] is None:
                    exit_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    attendance[name]["exit"] = exit_time
                    print(f"[INFO] Exit marked: {name} at {exit_time}")

                    # Append to CSV
                    df = pd.concat([df, pd.DataFrame([{
                        "Name": name,
                        "Entry Time": attendance[name]["entry"],
                        "Exit Time": attendance[name]["exit"]
                    }])], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)

                    del attendance[name]
                    del last_seen[name]

        # Show camera feed
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")

finally:
    # Handle unrecorded exits
    print("[INFO] Saving remaining sessions...")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name, record in attendance.items():
        if record["exit"] is None:
            record["exit"] = now
            df = pd.concat([df, pd.DataFrame([{
                "Name": name,
                "Entry Time": record["entry"],
                "Exit Time": record["exit"]
            }])], ignore_index=True)

    df.to_csv(CSV_FILE, index=False)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done. CSV updated.")
