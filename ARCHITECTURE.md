# Face Recognition Visitor Management System - Architecture Documentation

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Key Features](#key-features)
7. [API Endpoints](#api-endpoints)
8. [Database & Storage](#database--storage)
9. [Security Features](#security-features)
10. [How It Works](#how-it-works)

---

## 🎯 System Overview

This is a **Face Recognition-based Visitor Management System** that:
- Tracks employee/visitor attendance using facial recognition
- Detects and alerts for unknown faces
- Provides admin dashboard for user management
- Works in low-light conditions with image enhancement
- Sends email alerts for security breaches 

**Primary Use Case**: Industries tracking employee attendance or visitor access control

---

## 🛠 Technology Stack

### Backend
- **Flask** - Web framework (Python)
- **OpenCV (cv2)** - Image processing and webcam capture
- **PyTorch** - Deep learning framework for image enhancement
- **InsightFace** - Face detection and recognition (ArcFace model)
- **YOLO v8** - Face detection (optional, for bounding boxes)
- **scikit-learn** - Cosine similarity for face matching

### Frontend
- **HTML/CSS/Bootstrap 5** - UI design
- **JavaScript** - Camera capture, dynamic content

### AI Models
- **InsightFace (buffalo_l)** - Face embedding extraction (512-dim vectors)
- **Zero-DCE** - Low-light image enhancement
- **YOLOv8n-face** - Face detection

### Storage
- **CSV files** - Attendance logs, user data
- **Pickle files** - Face encodings storage
- **File system** - User images organized in folders

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System-main/
│
├── app.py                          # Main Flask application
├── encodings.pkl                   # Stored face embeddings
├── users.csv                       # User database (name, phone)
├── credentials.json                # Google API credentials (if used)
├── yolov8n-face-lindevs.pt        # YOLO face detection model
│
├── Attendance/                     # Daily attendance CSV files
│   ├── Attendance-01_15_25.csv
│   └── ...
│
├── static/                         # Static assets
│   ├── faces/                      # User face images (organized by user)
│   │   ├── Bibek_Meher_8260889508/
│   │   │   ├── Bibek_Meher1.jpg
│   │   │   └── ...
│   │   └── ...
│   ├── unknowns/                   # Captured unknown faces
│   ├── images/                     # UI assets (logos, etc.)
│   ├── style.css                   # Main stylesheet
│   └── admin.css                   # Admin panel styles
│
├── templates/                      # HTML templates
│   ├── home.html                   # Main dashboard
│   ├── start.html                  # Live video feed page
│   ├── sign.html                   # User registration
│   ├── login.html                  # User login
│   ├── adminlogin.html             # Admin login
│   └── admin.html                  # Admin dashboard
│
├── model/                          # AI models
│   ├── enhance_net_nopool.py       # Zero-DCE network architecture
│   ├── dce_model.pth               # Pre-trained enhancement model
│   └── dce_model_with_denoiser.pth
│
└── demo/                           # Screenshots/demo images
```

---

## 🧩 Core Components

### 1. **Face Detection & Recognition Engine**
- **InsightFace (ArcFace)**: Extracts 512-dimensional face embeddings
- **Cosine Similarity**: Matches faces with threshold > 0.45
- **YOLO v8**: Detects face bounding boxes

### 2. **Image Enhancement Module**
- **Zero-DCE Model**: Enhances low-light images (nighttime 7PM-6AM)
- **Gaussian Blur**: Reduces noise
- **Automatic Trigger**: Activates when brightness < 60 (grayscale mean)

### 3. **Attendance System**
- **Real-time Marking**: Marks attendance when face is recognized
- **Duplicate Prevention**: Checks if already marked today
- **CSV Logging**: Stores Name, Phone, Time

### 4. **Unknown Face Handler**
- **Snapshot Capture**: Saves unknown faces to `static/unknowns/`
- **Email Alerts**: Sends image to admin via Gmail SMTP
- **Rate Limiting**: Max 3 alerts per face per 3 hours
- **Persistent Labeling**: Assigns unique IDs (unknown1, unknown2, etc.)

### 5. **User Management**
- **Registration**: Upload/capture 1-50 images per user
- **Encoding Generation**: Extracts face embeddings and stores in `encodings.pkl`
- **Admin Controls**: Delete users, view logs

### 6. **Authentication System**
- **User Login**: `user` / `user@123`
- **Admin Login**: `Bibek35` / `8260889508`
- **Session Management**: Flask sessions

---

## 🔄 Data Flow

### Registration Flow
```
User fills form (name, phone) 
  → Uploads/captures images 
  → Images saved to static/faces/{Name}_{Phone}/ 
  → Face embeddings extracted 
  → Stored in encodings.pkl 
  → User added to users.csv
```

### Attendance Flow
```
Webcam captures frame 
  → Low-light enhancement (if needed) 
  → Face detection (InsightFace) 
  → Extract embedding 
  → Compare with known_encodings (cosine similarity) 
  → If match > 0.45: Mark attendance 
  → If match < 0.45: Trigger unknown face handler
```

### Unknown Face Flow
```
Unknown face detected 
  → Assign persistent label (unknown1, unknown2...) 
  → Save snapshot to static/unknowns/ 
  → Send email alert with image 
  → Log to unknown_log 
  → Rate limit: 3 alerts per 3 hours
```

---

## ✨ Key Features

### 1. **Real-time Face Recognition**
- Live video feed at 1280x720 resolution
- 30 FPS processing with 0.03s delay
- Multi-face detection support

### 2. **Low-Light Enhancement**
- Automatic activation during night hours (7PM-6AM)
- Zero-DCE neural network enhancement
- Brightness threshold: < 60 (grayscale mean)

### 3. **Security Alerts**
- Email notifications for unknown faces
- Snapshot storage with timestamps
- Alert rate limiting to prevent spam

### 4. **User Registration**
- Multiple image capture (1-50 images)
- Webcam integration with live preview
- File upload support

### 5. **Admin Dashboard**
- View all registered users
- Delete users (auto-rebuilds encodings)
- View unknown face logs

### 6. **Attendance Tracking**
- Daily CSV files (format: `Attendance-MM_DD_YY.csv`)
- Duplicate prevention
- Phone number extraction from username

---

## 🌐 API Endpoints

| Route | Method | Description | Auth Required |
|-------|--------|-------------|---------------|
| `/` | GET | Home dashboard | User login |
| `/login` | GET/POST | User login page | No |
| `/adminlogin` | GET/POST | Admin login page | No |
| `/admin` | GET | Admin dashboard | Admin login |
| `/register` | GET/POST | User registration | No |
| `/start` | GET | Live video feed page | No |
| `/video_feed` | GET | Video stream endpoint | No |
| `/mark-attendance` | POST | Manual attendance marking | No |
| `/delete_user/<user_id>` | GET | Delete user | Admin login |
| `/logout` | GET | Logout session | No |
| `/capture_image` | GET | Capture single frame | No |
| `/fix-missing-phones` | GET | Utility to fix users.csv | No |

---

## 💾 Database & Storage

### CSV Files
1. **Attendance/Attendance-{date}.csv**
   - Columns: `Name, Phone, Time`
   - Created daily automatically
   - Example: `Bibek_Meher_8260889508,8260889508,14:30:45`

2. **users.csv**
   - Columns: `Name_Phone, Phone`
   - Maps user folders to phone numbers
   - Example: `Bibek_Meher_8260889508,8260889508`

3. **admin_logs.csv** (if exists)
   - Admin activity logs

### Pickle Files
1. **encodings.pkl**
   - Stores: `(known_encodings, known_names)`
   - `known_encodings`: List of 512-dim numpy arrays
   - `known_names`: List of strings (format: `Name_Phone`)

### File System
1. **static/faces/{Name}_{Phone}/**
   - User images: `{Name}1.jpg`, `{Name}2.jpg`, etc.
   - Max 50 images per user

2. **static/unknowns/**
   - Unknown face snapshots
   - Format: `unknown{N}_{YYYYMMDD_HHMMSS}.jpg`

---

## 🔒 Security Features

### 1. **Authentication**
- Session-based login (Flask sessions)
- Separate user and admin roles
- Hardcoded credentials (should be moved to env variables)

### 2. **Unknown Face Detection**
- Real-time monitoring
- Email alerts to admin
- Snapshot evidence storage

### 3. **Rate Limiting**
- Max 3 email alerts per unknown face per 3 hours
- Prevents email spam

### 4. **Data Privacy**
- Face embeddings stored locally (not images in memory)
- CSV files for audit trails

---

## 🔍 How It Works

### Face Recognition Algorithm

1. **Face Detection**
   ```python
   face_app = FaceAnalysis(name='buffalo_l')
   results = face_app.get(rgb_image)
   # Returns: face.bbox, face.embedding
   ```

2. **Embedding Extraction**
   - 512-dimensional vector per face
   - Represents unique facial features

3. **Face Matching**
   ```python
   matches = cosine_similarity([embedding], known_encodings)[0]
   best_match_index = np.argmax(matches)
   match_score = matches[best_match_index]
   
   if match_score > 0.45:
       # Face recognized
   else:
       # Unknown face
   ```

4. **Threshold Logic**
   - **> 0.45**: Recognized (mark attendance)
   - **< 0.45**: Unknown (trigger alert)

### Low-Light Enhancement

```python
# Check conditions
is_night_time = (hour >= 19 or hour < 6)
is_dark = np.mean(grayscale_image) < 60

if is_night_time and is_dark:
    enhanced_frame = enhance_image(frame)
```

**Zero-DCE Process**:
1. Resize to 512x512
2. Normalize to [0, 1]
3. Pass through DCE neural network
4. Apply 8 curve adjustments
5. Denormalize to [0, 255]

### Email Alert System

```python
def send_unknown_face_alert(image):
    # SMTP SSL connection to Gmail
    # Attach JPEG image
    # Send to admin email
```

**Requirements**:
- `.env` file with `FR_APP_PASSWORD` (Gmail app password)
- Gmail account with 2FA enabled

---

## 🚀 Startup Sequence

1. **Load Models**
   - YOLO v8 face detector
   - InsightFace ArcFace model
   - Zero-DCE enhancement model

2. **Load Encodings**
   - Read `encodings.pkl`
   - Populate `known_encodings` and `known_names`

3. **Initialize Webcam**
   - Open camera (index 0)
   - Set resolution: 1280x720

4. **Create Directories**
   - `Attendance/` folder
   - Today's CSV file

5. **Start Flask Server**
   - Host: `0.0.0.0` (accessible on network)
   - Port: `5050`
   - Debug mode: `True`

---

## 📊 Performance Considerations

### Optimization Techniques
1. **Threading**: Attendance marking runs in separate thread
2. **Frame Rate Limiting**: 0.03s delay (≈30 FPS)
3. **JPEG Compression**: 90% quality for streaming
4. **Conditional Enhancement**: Only in low-light conditions

### Resource Usage
- **CPU**: High (face detection + recognition)
- **Memory**: ~500MB (models + encodings)
- **Disk**: Grows with user images and attendance logs

---

## 🐛 Known Issues & Limitations

1. **Hardcoded Credentials**: Admin/user passwords in code
2. **No Database**: Uses CSV files (not scalable)
3. **Single Camera**: No multi-camera support
4. **No Face Liveness**: Can be fooled by photos
5. **Email Dependency**: Requires Gmail app password
6. **No HTTPS**: Insecure for production

---

## 🔧 Configuration

### Environment Variables (.env)
```
FR_APP_PASSWORD=your_gmail_app_password
SECRET_KEY=your_flask_secret_key
```

### Adjustable Parameters (in app.py)
- **Face match threshold**: `0.45` (line ~230)
- **Camera resolution**: `1280x720` (line ~180)
- **Alert cooldown**: `3 hours` (line ~250)
- **Max images per user**: `50` (line ~450)
- **Low-light threshold**: `60` (line ~210)

---

## 📝 Future Enhancements

1. **Database Integration**: PostgreSQL/MongoDB
2. **Multi-camera Support**: RTSP streams
3. **Face Liveness Detection**: Anti-spoofing
4. **REST API**: Mobile app integration
5. **Cloud Storage**: AWS S3 for images
6. **Analytics Dashboard**: Attendance reports
7. **Role-based Access Control**: Multiple admin levels
8. **Notification System**: SMS/WhatsApp alerts

---

## 🎓 Learning Resources

- **InsightFace**: https://github.com/deepinsight/insightface
- **Zero-DCE**: https://github.com/Li-Chongyi/Zero-DCE
- **Flask**: https://flask.palletsprojects.com/
- **OpenCV**: https://docs.opencv.org/

---

## 📞 Support

For issues or questions:
- Email: bibekmeher35@gmail.com
- GitHub: https://github.com/Bibekmeher35/Face-Recognition-VMS

---

**Last Updated**: January 2025
**Version**: 1.0
**Author**: Bibek Meher
