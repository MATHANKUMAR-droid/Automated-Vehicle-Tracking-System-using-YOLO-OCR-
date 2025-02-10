import cv2
import numpy as np
import pytesseract
import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog, ttk
from ultralytics import YOLO
from datetime import datetime
from PIL import Image, ImageTk
import os

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
model = YOLO("yolov8n.pt")

# Centroid Tracker
class CentroidTracker:
    def __init__(self):
        self.objects = {}
        self.next_object_id = 0

    def update(self, detections):
        updated_objects = {}
        for detection in detections:
            x, y, w, h = detection
            updated_objects[self.next_object_id] = (x, y, w, h)
            self.next_object_id += 1
        self.objects = updated_objects
        return updated_objects

# Extract license plate text
def extract_license_plate_text(image, bbox):
    x, y, w, h = bbox
    plate = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, config='--psm 7')
    return text.strip()

# Determine car color
def get_car_color(image, bbox):
    x, y, w, h = bbox
    car_roi = image[y:y+h, x:x+w]
    avg_color = np.mean(car_roi, axis=(0, 1))
    colors = {"Red": (200, 50, 50), "Blue": (50, 50, 200), "Green": (50, 200, 50), "White": (200, 200, 200), "Black": (50, 50, 50)}
    
    closest_color = min(colors, key=lambda c: np.linalg.norm(np.array(colors[c]) - avg_color))
    return closest_color

# GUI application
class VehicleTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Tracking System")
        
        # Upload Button
        tk.Label(root, text="Select an option:", font=("Arial", 14)).pack(pady=10)
        tk.Button(root, text="Upload Video", command=self.choose_video, font=("Arial", 12)).pack(pady=10)

        # Video Display Frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack(pady=10)

        # Table for analysis
        self.tree = ttk.Treeview(root, columns=("Timestamp", "Plate Number", "Car Color", "Speed"), show="headings")
        for col in ("Timestamp", "Plate Number", "Car Color", "Speed"):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150)
        self.tree.pack(pady=10)

        # Data storage
        self.data = []
        self.cap = None

    def choose_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.process_video(file_path)

    def process_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        tracker = CentroidTracker()

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            results = model(frame)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2 - x1, y2 - y1))

            objects = tracker.update(detections)

            for obj_id, bbox in objects.items():
                text = extract_license_plate_text(frame, bbox)
                color = get_car_color(frame, bbox)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                speed = "N/A"

                # Store in data list
                self.data.append([timestamp, text, color, speed])

                # Update table
                self.tree.insert("", "end", values=(timestamp, text, color, speed))

                # Draw boxes
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{text}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show video in Tkinter
            self.display_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Save to Excel
        df = pd.DataFrame(self.data, columns=["Timestamp", "Plate Number", "Car Color", "Speed"])
        df.to_excel("Vehicle_Tracking.xlsx", index=False)
        print("✅ Data saved to Vehicle_Tracking.xlsx")

    def display_frame(self, frame):
        frame = cv2.resize(frame, (500, 300))  # Resize video to fit Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)
        self.root.update_idletasks()

# Start GUI
root = tk.Tk()
app = VehicleTrackingApp(root)
root.mainloop()
