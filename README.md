# 🧠 Smart Face Recognition Attendance System  

A super cool **AI-powered Face Recognition Attendance System** built with **Python & OpenCV**.  
This system captures face images, trains a recognition model, and automatically marks attendance when a registered person is recognized.  

---

## 🚀 Features  

- 📸 **Face Capture**: Captures **100 grayscale images** per user for training.  
- 🧑‍💻 **Model Training**: Uses **LBPH (Local Binary Patterns Histograms)** algorithm from OpenCV.  
- 👀 **Real-time Recognition**: Detects faces via webcam, draws a bounding box, and displays the name.  
- 📝 **Attendance Marking**: Marks attendance **only once per day** for each user.  
- 📂 **Dataset Management**: Stores images in a structured dataset folder.  
- 💾 **CSV Attendance File**: Saves daily attendance in `.csv` format.  

---

## 🛠️ Tech Stack  

- **Python 3.x**  
- **OpenCV** (for face detection & recognition)  
- **NumPy** (for array operations)  
- **CSV Module** (for attendance logging)  
- **Datetime Module** (for date-based attendance)  

---

## 📂 Project Structure  

```plaintext
smart_face_recognition_attendance_marking_system/
├── face_recognition_attendance_ai_system/
│   ├── dataset/                  # Stores captured face images
│   ├── trainer/                  # Stores the trained model (yml file)
│   ├── haarcascade_frontalface_default.xml  # Haarcascade file for face detection
│   ├── face_dataset.py           # Captures and stores images
│   ├── training.py               # Trains the LBPH face recognizer
│   ├── face_recognition.py       # Recognizes faces and marks attendance
│   ├── attendance.csv            # Attendance log file
│   ├── names.txt                 # Stores ID-to-Name mapping
│   └── utils.py                  # (Optional) Helper functions
├── README.md                     # Project documentation
└── requirements.txt              # Dependencies list
```
---

## ⚡ How It Works  

1. **Capture Images**  
   - Run `face_dataset.py`  
   - Enter user name → System captures **100 grayscale images** of the face.  

2. **Train Model**  
   - Run `training.py`  
   - It trains the **LBPH recognizer** with captured images.  

3. **Recognize & Mark Attendance**  
   - Run `face_recognition.py`  
   - The camera opens → Detects & recognizes faces.  
   - Marks attendance **once per day per user** in a CSV file.  

---

## 📊 Sample Attendance CSV  

| Name     | Date       | Time     |
|----------|-----------|----------|
| Aniket   | 2025-08-16 | 09:45 AM |
| Ramesh   | 2025-08-16 | 09:47 AM |

---

## 🎯 Future Enhancements  

- Add **GUI** for better usability.  
- Store attendance in **databases** like MySQL / Firebase.  
- Enable **email/SMS notification** on attendance.  
- Improve recognition with **deep learning models** (e.g., FaceNet).  

---

## 🤝 Contribution  

Feel free to **fork** this repo, improve it, and send a PR 🚀  

---

## 📜 License  

This project is licensed under the **MIT License**.  

---

💡 *Built with ❤️ by Aniket & AI*  
