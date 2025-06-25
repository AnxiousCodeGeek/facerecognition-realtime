import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import DateEntry
import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
from PIL import Image, ImageTk
import shutil

# === Configuration ===
DB_PATH = "database"
CSV_FILE = "attendance.csv"
EXIT_TIMEOUT_SEC = 10
MODELS = ["Facenet", "VGG-Face", "ArcFace", "DeepFace"]

# Ensure directories
os.makedirs(DB_PATH, exist_ok=True)
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["Name", "Entry Time", "Exit Time"])
    df.to_csv(CSV_FILE, index=False)

# === GUI Setup ===
root = tk.Tk()
root.title("Face Attendance System")
root.geometry("1000x700")

# State
cap = cv2.VideoCapture(0)
attendance = {}
last_seen = {}
selected_model = tk.StringVar(value="Facenet")

# === Frames ===
video_frame = tk.Label(root)
video_frame.pack(pady=5)

form_frame = tk.Frame(root)
form_frame.pack(pady=5)

# === Name Entry and Buttons ===
tk.Label(form_frame, text="Name:").grid(row=0, column=0, padx=5)
name_entry = tk.Entry(form_frame)
name_entry.grid(row=0, column=1, padx=5)
tk.Button(form_frame, text="Capture & Save", command=lambda: capture_face()).grid(row=0, column=2, padx=5)
tk.Button(form_frame, text="Start Attendance", command=lambda: start_attendance()).grid(row=0, column=3, padx=5)

# === Model Selection ===
tk.Label(form_frame, text="Model:").grid(row=0, column=4, padx=5)
model_menu = ttk.Combobox(form_frame, textvariable=selected_model, values=MODELS)
model_menu.grid(row=0, column=5, padx=5)

# === Attendance Table ===
tree_frame = tk.Frame(root)
tree_frame.pack(pady=10)

columns = ("Name", "Entry Time", "Exit Time")
attendance_table = ttk.Treeview(tree_frame, columns=columns, show='headings', height=10)
for col in columns:
    attendance_table.heading(col, text=col)
    attendance_table.column(col, width=250)
attendance_table.pack()

def update_table():
    for row in attendance_table.get_children():
        attendance_table.delete(row)
    df = pd.read_csv(CSV_FILE)
    for _, row in df.iterrows():
        attendance_table.insert('', 'end', values=(row["Name"], row["Entry Time"], row["Exit Time"]))

# === Export by Date ===
export_frame = tk.Frame(root)
export_frame.pack(pady=10)

export_date = DateEntry(export_frame, date_pattern='yyyy-mm-dd')
export_date.grid(row=0, column=0, padx=5)
tk.Button(export_frame, text="Export by Date", command=lambda: export_by_date()).grid(row=0, column=1, padx=5)

def export_by_date():
    date_str = export_date.get()
    df = pd.read_csv(CSV_FILE)
    filtered = df[df['Entry Time'].str.startswith(date_str)]
    out_path = f"attendance_{date_str}.csv"
    filtered.to_csv(out_path, index=False)
    messagebox.showinfo("Exported", f"Data exported to {out_path}")

# === Manage People ===
manage_frame = tk.Frame(root)
manage_frame.pack(pady=10)

def refresh_people():
    people = [d for d in os.listdir(DB_PATH) if os.path.isdir(os.path.join(DB_PATH, d))]
    person_menu['values'] = people

def delete_person():
    person = person_var.get()
    if not person:
        return
    path = os.path.join(DB_PATH, person)
    if os.path.exists(path):
        shutil.rmtree(path)
        messagebox.showinfo("Deleted", f"Deleted {person}")
        refresh_people()

def update_person():
    person = person_var.get()
    if not person:
        return
    ret, frame = cap.read()
    if ret:
        path = os.path.join(DB_PATH, person)
        os.makedirs(path, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(path, filename), frame)
        messagebox.showinfo("Updated", f"Image added to {person}")

person_var = tk.StringVar()
person_menu = ttk.Combobox(manage_frame, textvariable=person_var)
person_menu.grid(row=0, column=0, padx=5)
tk.Button(manage_frame, text="Delete Person", command=delete_person).grid(row=0, column=1, padx=5)
tk.Button(manage_frame, text="Update Photo", command=update_person).grid(row=0, column=2, padx=5)
refresh_people()

# === Core Functions ===
def capture_face():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return
    ret, frame = cap.read()
    if not ret:
        return
    person_path = os.path.join(DB_PATH, name)
    os.makedirs(person_path, exist_ok=True)
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(person_path, filename), frame)
    messagebox.showinfo("Saved", f"Image saved to {person_path}")
    refresh_people()

def start_attendance():
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
                model_name=selected_model.get(),
                detector_backend='opencv'
            )
            if len(res) > 0 and len(res[0]) > 0:
                best = res[0].iloc[0]
                name = best['identity'].split(os.path.sep)[-2]
                now = datetime.now()
                if name not in attendance:
                    attendance[name] = {"entry": now.strftime("%Y-%m-%d %H:%M:%S"), "exit": None}
                last_seen[name] = now

        except Exception as e:
            print("[WARN]", e)

        now = datetime.now()
        for name in list(attendance.keys()):
            if name in last_seen:
                if (now - last_seen[name]).total_seconds() > EXIT_TIMEOUT_SEC and attendance[name]["exit"] is None:
                    attendance[name]["exit"] = now.strftime("%Y-%m-%d %H:%M:%S")
                    df = pd.read_csv(CSV_FILE)
                    df = pd.concat([df, pd.DataFrame([{
                        "Name": name,
                        "Entry Time": attendance[name]["entry"],
                        "Exit Time": attendance[name]["exit"]
                    }])], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)
                    update_table()
                    del attendance[name]
                    del last_seen[name]

        root.after(1000, detect_loop)

    detect_loop()

# === Live Camera Feed ===
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_frame.imgtk = imgtk
        video_frame.configure(image=imgtk)
    root.after(10, update_frame)

update_table()
update_frame()

# === Graceful Exit ===
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
