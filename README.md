# Automated Vehicle Tracking System using YOLO & OCR

## Overview
The **Automated Vehicle Tracking System** is a computer vision-based solution designed to detect vehicles, recognize license plates, extract plate numbers, and log essential details such as timestamp, vehicle color, and speed. This project is built using **YOLOv8** for object detection and **Tesseract OCR** for text recognition, making it a highly efficient and automated system for vehicle monitoring applications.
This system can be integrated into **traffic surveillance, toll systems, and security applications** to enhance vehicle monitoring and automation.

**Features**
✅ **Advanced Object Detection**: Utilizes **YOLOv8** for accurate vehicle and license plate detection.
✅ **Optical Character Recognition (OCR)**: Implements **Tesseract OCR** to extract license plate numbers with high accuracy.
✅ **Car Color Identification**: Uses **OpenCV** and image processing techniques to identify vehicle colors.
✅ **Interactive GUI Interface**: Developed a **Tkinter**-based GUI to display live video feed, process images, and present analysis results.
✅ **Automated Data Logging**: Stores timestamp, license plate number, car color, and speed in an **Excel sheet** for record-keeping and future analysis.
✅ **Real-time & Batch Processing**: Supports live camera feed and pre-recorded video processing for flexible usage.

**Tech Stack**
- **Python** - Core programming language
- **OpenCV** - Image and video processing
- **YOLOv8** - Object detection for vehicle and license plate recognition
- **Tesseract OCR** - Optical Character Recognition to extract text from plates
- **Tkinter** - GUI development for user interaction
- **Pandas** - Data handling and storage in an Excel sheet
- **NumPy** - Efficient numerical operations

**Installation & Setup**
**Prerequisites**
Ensure you have the following installed on your system:
- Python 3.8+
- pip (Python package manager)
- Git

 Step 1: Clone the Repository
```bash
$ git clone https://github.com/yourusername/Automated-Vehicle-Tracking-System.git
$ cd Automated-Vehicle-Tracking-System
```

Step 2: Install Required Dependencies
```bash
$ pip install -r requirements.txt
```

 Step 3: Download YOLOv8 Model
Download the **YOLOv8 weights** from [Ultralytics YOLOv8 GitHub](https://github.com/ultralytics/ultralytics) and place them in the `models/` directory.

 Step 4: Run the Application
For **live video processing** using a webcam:
```bash
$ python main.py --mode live
```
For **processing a pre-recorded video**:
```bash
$ python main.py --mode video --input path/to/video.mp4
```


**Project Structure**
```
Automated-Vehicle-Tracking-System/
│── models/                    # YOLOv8 model weights
│── data/                      # Sample images/videos
│── output/                    # Processed results
│── src/                       # Source code
│   │── vehicle_detection.py   # YOLOv8 for vehicle detection
│   │── plate_recognition.py   # OCR for extracting plate numbers
│   │── color_identification.py# Image processing for color detection
│   │── gui.py                 # Tkinter-based interactive GUI
│   │── logger.py              # Data logging into an Excel sheet
│── main.py                    # Entry point to run the program
│── requirements.txt           # List of dependencies
│── README.md                  # Project documentation
```

** How It Works**1. **Vehicle Detection**: The system detects vehicles and license plates in a live feed or uploaded video using YOLOv8.
2. **License Plate Recognition**: Extracts text from the detected plates using Tesseract OCR.
3. **Car Color Identification**: Determines vehicle color using image processing techniques in OpenCV.
4. **Data Logging**: Stores extracted details (timestamp, license plate number, car color, speed) in an Excel file.
5. **Interactive GUI**: Displays live results, allows users to choose between live and recorded video input.

**Sample Output**
| Timestamp       | License Plate | Car Color | Speed (km/h) |
|----------------|--------------|-----------|--------------|
| 2025-02-10 14:30:01 | ABC1234     | Red       | 60           |
| 2025-02-10 14:30:05 | XYZ5678     | Blue      | 45           |

** Use Cases**
- **Traffic Surveillance**: Monitor traffic and enforce road laws.
- **Toll Booth Automation**: Automatically capture vehicle details for toll payments.
- **Parking Management**: Track vehicles entering and exiting a parking lot.
- **Security & Law Enforcement**: Detect stolen vehicles by cross-checking with a database.

**Future Enhancements**
🔹 Integrate **DeepSORT** for vehicle tracking across multiple frames.
🔹 Implement a **database system** for enhanced data storage and retrieval.
🔹 Deploy the system as a **web-based application** using Flask or FastAPI.
🔹 Improve OCR accuracy using **deep learning-based text recognition**.



