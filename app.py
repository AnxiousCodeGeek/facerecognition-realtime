import tkinter as tk
from tkinter import messagebox
import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
from PIL import Image, ImageTk

# Paths
DB_PATH = "database"
CSV_FILE = "attendance.csv"
EXIT_TIMEOUT_SEC = 10

# Attendance state
attendance = {}
last_seen = {}

# Ensure folders
os.makedirs(DB_PATH, exist_ok=True)
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
    df.to_csv(CSV_FILE, index=False)

# === GUI Setup ===
root = tk.Tk()
root.title("Face Attendance System")

# Webcam Feed
camera_label = tk.Label(root)
camera_label.grid(row=0, column=0, columnspan=3)

# Name Entry
name_label = tk.Label(root, text="Name:")
name_label.grid(row=1, column=0, padx=5, pady=5)
name_entry = tk.Entry(root)
name_entry.grid(row=1, column=1, padx=5, pady=5)

# Initialize Camera
cap = cv2.VideoCapture(0)

# Update Camera Frame
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, update_frame)

# Capture and Save Face
def capture_face():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to capture image.")
        return
    person_path = os.path.join(DB_PATH, name)
    os.makedirs(person_path, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(person_path, filename)
    cv2.imwrite(filepath, frame)
    messagebox.showinfo("Success", f"Image saved to {filepath}")

# Start Attendance Monitoring
def start_attendance():
    global attendance, last_seen
    print("[INFO] Attendance system started...")

    def detect_loop():
        ret, frame = cap.read()
        if not ret:
            root.after(1000, detect_loop)
            return

        try:
            res = DeepFace.find(
                frame,
                db_path=DB_PATH,
                enforce_detection=False,
                model_name='Facenet',
                detector_backend='opencv'
            )

            if len(res) > 0 and len(res[0]) > 0:
                best = res[0].iloc[0]
                name = best['identity'].split(os.path.sep)[-2]

                now = datetime.now()
                if name not in attendance:
                    attendance[name] = {"entry": now.strftime("%Y-%m-%d %H:%M:%S"), "exit": None}
                    print(f"[ENTRY] {name} at {attendance[name]['entry']}")

                last_seen[name] = now

        except Exception as e:
            print("[WARN]", str(e))

        now = datetime.now()
        for name in list(attendance.keys()):
            if name in last_seen:
                if (now - last_seen[name]).total_seconds() > EXIT_TIMEOUT_SEC and attendance[name]["exit"] is None:
                    attendance[name]["exit"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[EXIT] {name} at {attendance[name]['exit']}")

                    # Save to CSV
                    df = pd.read_csv(CSV_FILE)
                    df = pd.concat([df, pd.DataFrame([{
                        "Name": name,
                        "Entry Time": attendance[name]["entry"],
                        "Exit Time": attendance[name]["exit"]
                    }])], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)

                    del attendance[name]
                    del last_seen[name]

        root.after(1000, detect_loop)

    detect_loop()

# Buttons
tk.Button(root, text="Capture & Save", command=capture_face).grid(row=1, column=2, padx=5, pady=5)
tk.Button(root, text="Start Attendance", command=start_attendance).grid(row=2, column=0, columnspan=3, pady=10)

# Start webcam loop
update_frame()

# Exit cleanup
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
