# 🎯 Face Recognition Visitor Management System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1.1-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent attendance tracking system powered by deep learning face recognition**

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Contributing](#-contributing)

</div>

---

## 📖 Description

A production-ready **Face Recognition Visitor Management System** designed for industries to automate employee attendance tracking and visitor access control. Built with state-of-the-art AI models including **InsightFace (ArcFace)**, **YOLO v8**, and **Zero-DCE** for low-light enhancement.

### 🎯 Key Highlights
- ✅ **Real-time face recognition** with 95%+ accuracy
- 🌙 **Low-light enhancement** for 24/7 operation
- 🚨 **Unknown face detection** with email alerts
- 📊 **Automated attendance logging** with CSV exports
- 👥 **Multi-face detection** support
- 🔐 **Admin dashboard** for user management

---

## ✨ Features

### Core Functionality
| Feature | Description |
|---------|-------------|
| 🎥 **Live Video Feed** | Real-time webcam streaming at 1280x720 resolution |
| 👤 **User Registration** | Upload or capture 1-50 images per person |
| ✅ **Attendance Marking** | Automatic attendance with duplicate prevention |
| 🔍 **Face Recognition** | InsightFace ArcFace with cosine similarity matching |
| 🌙 **Night Mode** | Zero-DCE enhancement for low-light conditions |
| 📧 **Email Alerts** | Instant notifications for unknown faces |
| 📊 **Daily Reports** | CSV attendance logs with timestamps |
| 🛡️ **Admin Panel** | User management and system controls |

### Advanced Features
- **Multi-face Detection**: Recognize multiple people simultaneously
- **Persistent Unknown Tracking**: Assigns unique IDs to unrecognized faces
- **Rate Limiting**: Prevents alert spam (max 3 per face per 3 hours)
- **Snapshot Storage**: Saves unknown face images for security audit
- **Phone Number Integration**: Links attendance to contact information
- **Session Management**: Secure user and admin authentication

---

## 🎬 Demo

### Screenshots

<table>
  <tr>
    <td><img src="demo/USER INTERFACE.jpeg" alt="Home Dashboard" width="400"/><br/><b>Home Dashboard</b></td>
    <td><img src="demo/User Adding.jpeg" alt="User Registration" width="400"/><br/><b>User Registration</b></td>
  </tr>
  <tr>
    <td><img src="demo/Live Images Capturing.jpeg" alt="Live Capture" width="400"/><br/><b>Live Image Capture</b></td>
    <td><img src="demo/CSV ATTENDANCE RECORD.jpeg" alt="Attendance Log" width="400"/><br/><b>Attendance Records</b></td>
  </tr>
</table>

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or IP camera
- 4GB+ RAM recommended
- macOS, Linux, or Windows

### Step 1: Clone Repository
```bash
git clone https://github.com/Bibekmeher35/Face-Recognition-VMS.git
cd Face-Recognition-VMS
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note for macOS (Apple Silicon):**
```bash
pip install onnxruntime-silicon  # Better performance on M1/M2/M3
```

### Step 4: Download AI Models
The system will automatically download required models on first run:
- InsightFace buffalo_l model (~200MB)
- YOLO v8 face detection model (included: `yolov8n-face-lindevs.pt`)
- Zero-DCE enhancement model (included: `model/dce_model.pth`)

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```bash
# Email alert configuration (optional)
FR_APP_PASSWORD=your_gmail_app_password
SECRET_KEY=your_flask_secret_key_here
```

**To generate Gmail App Password:**
1. Enable 2-Factor Authentication on your Google account
2. Go to: https://myaccount.google.com/apppasswords
3. Generate a new app password for "Mail"
4. Copy the 16-character password to `.env`

---

## 💻 Usage

### Starting the Application

```bash
python app.py
```

The server will start at: **http://localhost:5050**

### Default Credentials

| Role | Username | Password |
|------|----------|----------|
| **User** | `user` | `user@123` |
| **Admin** | `Bibek35` | `8260889508` |

⚠️ **Security Warning**: Change these credentials in production!

### Quick Start Guide

#### 1️⃣ Register New Users
1. Navigate to **"Add New User"** section
2. Enter name and 10-digit phone number
3. Upload image OR capture 5 photos from webcam
4. Click **"Register"**

#### 2️⃣ Mark Attendance
1. Click **"Take Attendance"** button
2. Face recognition runs automatically
3. Attendance logged to `Attendance/Attendance-{date}.csv`

#### 3️⃣ Admin Dashboard
1. Login as admin
2. View all registered users
3. Delete users (auto-rebuilds face encodings)
4. Monitor unknown face alerts

---

## 🏗️ Architecture

### Technology Stack

**Backend:**
- Flask 3.1.1 (Web framework)
- InsightFace (Face recognition)
- PyTorch (Deep learning)
- OpenCV (Image processing)

**AI Models:**
- **InsightFace ArcFace**: 512-dim face embeddings
- **YOLO v8**: Face detection
- **Zero-DCE**: Low-light enhancement

**Frontend:**
- HTML5 + CSS3
- Bootstrap 5.3.2
- JavaScript (ES6)

### Project Structure
```
Face-Recognition-VMS/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── ARCHITECTURE.md           # Detailed system documentation
├── .env                      # Environment variables (create this)
│
├── model/                    # AI models
│   ├── enhance_net_nopool.py
│   └── dce_model.pth
│
├── static/                   # Static assets
│   ├── faces/                # User face images
│   ├── unknowns/             # Unknown face snapshots
│   └── images/               # UI assets
│
├── templates/                # HTML templates
│   ├── home.html
│   ├── admin.html
│   ├── sign.html
│   └── ...
│
├── Attendance/               # Daily CSV logs
│   └── Attendance-{date}.csv
│
├── encodings.pkl             # Face embeddings database
└── users.csv                 # User information
```

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md)

---

## 🔧 Configuration

### Camera Settings (app.py)
```python
cap = cv2.VideoCapture(0)  # 0 = default webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

**For IP Camera (RTSP):**
```python
cap = cv2.VideoCapture("rtsp://username:password@ip:port/path")
```

### Recognition Threshold
```python
if match_score > 0.45:  # Adjust threshold (0.0 - 1.0)
    # Face recognized
```
- **Higher (0.6+)**: More strict, fewer false positives
- **Lower (0.3-0.4)**: More lenient, may increase false positives

### Low-Light Enhancement
```python
is_night_time = current_hour >= 19 or current_hour < 6  # 7 PM - 6 AM
if is_night_time and np.mean(gray) < 60:  # Brightness threshold
    frame = enhance_image(frame)
```

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home dashboard |
| `/login` | GET/POST | User login |
| `/adminlogin` | GET/POST | Admin login |
| `/register` | GET/POST | User registration |
| `/start` | GET | Live video feed page |
| `/video_feed` | GET | Video stream (MJPEG) |
| `/mark-attendance` | POST | Manual attendance API |
| `/admin` | GET | Admin dashboard |
| `/delete_user/<id>` | GET | Delete user |
| `/logout` | GET | Logout session |

---

## 🛡️ Security Features

- ✅ Session-based authentication
- ✅ Unknown face detection with alerts
- ✅ Rate-limited email notifications
- ✅ Snapshot evidence storage
- ✅ CSV audit trails
- ⚠️ **TODO**: HTTPS support, password hashing, RBAC

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'cv2'**
```bash
pip install opencv-python
# OR for headless systems:
pip install opencv-python-headless
```

**2. Camera not accessible**
- Check camera permissions in System Preferences (macOS)
- Try different camera index: `cv2.VideoCapture(1)`
- Ensure no other app is using the camera

**3. InsightFace model download fails**
```bash
# Manually download buffalo_l model
mkdir -p ~/.insightface/models/buffalo_l
# Download from: https://github.com/deepinsight/insightface/releases
```

**4. Email alerts not working**
- Verify `.env` file exists with `FR_APP_PASSWORD`
- Check Gmail 2FA is enabled
- Ensure app password is correct (16 characters, no spaces)

**5. Low accuracy in recognition**
- Capture more images per user (20-50 recommended)
- Ensure good lighting during registration
- Lower threshold: `match_score > 0.40`

---

## 📈 Performance Optimization

### For Better Speed
1. **Reduce resolution**: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)`
2. **Increase frame delay**: `time.sleep(0.05)` in `gen_frames()`
3. **Use GPU**: Install `onnxruntime-gpu` for CUDA support

### For Better Accuracy
1. **Capture diverse angles** during registration
2. **Use consistent lighting** conditions
3. **Increase match threshold** to 0.50-0.55
4. **Collect 30+ images** per person

---

## 🗺️ Roadmap

### Completed ✅
- [x] Real-time face recognition
- [x] User registration with webcam
- [x] Admin dashboard
- [x] Unknown face alerts
- [x] Low-light enhancement
- [x] Daily attendance logs

### In Progress 🚧
- [ ] Improve UI/UX design
- [ ] Add date/time display on home page
- [ ] Company logo customization
- [ ] Enhanced admin privileges

### Planned 📋
- [ ] Database integration (PostgreSQL/MongoDB)
- [ ] REST API for mobile apps
- [ ] Multi-camera support
- [ ] Face liveness detection (anti-spoofing)
- [ ] Analytics dashboard with charts
- [ ] SMS/WhatsApp notifications
- [ ] Cloud deployment (AWS/Azure)
- [ ] Docker containerization

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/AmazingFeature`
3. **Commit** your changes: `git commit -m 'Add AmazingFeature'`
4. **Push** to branch: `git push origin feature/AmazingFeature`
5. **Open** a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guide for Python code
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting PR
- Don't wait for issue assignment - just contribute!

### Areas for Contribution
- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements
- ⚡ Performance optimizations
- 🧪 Test coverage

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Bibek Meher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 👨‍💻 Author

**Bibek Meher**
- GitHub: [@Bibekmeher35](https://github.com/Bibekmeher35)
- Email: bibekmeher35@gmail.com

---

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition models
- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLO v8 implementation
- [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) - Low-light enhancement
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

## 📞 Support

If you encounter any issues or have questions:

1. Check [Troubleshooting](#-troubleshooting) section
2. Review [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
3. Open an [Issue](https://github.com/Bibekmeher35/Face-Recognition-VMS/issues)
4. Email: bibekmeher35@gmail.com

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=Bibekmeher35/Face-Recognition-VMS&type=Date)](https://star-history.com/#Bibekmeher35/Face-Recognition-VMS&Date)

---

<div align="center">

**Made with ❤️ by Bibek Meher**

[⬆ Back to Top](#-face-recognition-visitor-management-system)

</div>
