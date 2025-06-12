# 📁 Imports and Library Setup
# Ensure the following packages are installed:
# pip install ultralytics scikit-learn insightface torch opencv-python
import os
import cv2
import torch
import numpy as np
import pandas as pd
from datetime import date, datetime
from flask import Flask, jsonify, request, render_template, redirect, session, url_for, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import pickle
import time
from model.enhance_net_nopool import enhance_net_nopool
import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import logging
logging.basicConfig(level=logging.INFO)
unknown_face_labels = {}
unknown_counter = 1

# --- Unknown embeddings and names for embedding-based unknown matching ---
unknown_embeddings = []
unknown_names = []

# Load or initialize unknown face labels
if os.path.exists("unknown.csv"):
    with open("unknown.csv", "r") as f:
        for line in f.readlines()[1:]:
            label, face_id = line.strip().split(',')
            unknown_face_labels[face_id] = label
            unknown_counter = max(unknown_counter, int(label.replace("unknown", "")) + 1)
else:
    with open("unknown.csv", "w") as f:
        f.write("Label,FaceID\n")

# 🔧 Flask Configuration Class
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key')
    UPLOAD_FOLDER = 'static/faces'

# 🚀 Initialize Flask App
app = Flask(__name__)
app.config.from_object(Config)

# 📅 Date Configuration and Attendance File Setup
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

# Attendance folder setup
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Phone,Time')

# 🤖 Load YOLO and InsightFace Models
# Load YOLOv8 model for face detection (optional, for bounding box)
model = YOLO("yolov8n-face-lindevs.pt")

# InsightFace ArcFace model setup
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

# Load Zero-DCE model for low-light enhancement
dce_model = enhance_net_nopool()
dce_model.load_state_dict(torch.load("model/dce_model.pth", map_location='cpu'))
dce_model.eval()

def enhance_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    input_tensor = torch.from_numpy(img_resized / 255.0).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        enhanced = dce_model(input_tensor)[0]
    enhanced = enhanced.squeeze().permute(1, 2, 0).numpy()
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

# 📧 Utility function to send email alert for unknown face
def send_unknown_face_alert(image):
    sender_email = "bibekmeher35@gmail.com"
    receiver_email = "jayrudra896@gmail.com"
    password = "relkwtvaccyetand"

    msg = MIMEMultipart()
    msg['Subject'] = 'Unknown Person Detected'
    msg['From'] = sender_email
    msg['To'] = receiver_email

    text = MIMEText("An unknown person was detected by the attendance system.")
    msg.attach(text)

    img_data = cv2.imencode('.jpg', image)[1].tobytes()
    image_attachment = MIMEImage(img_data, name="unknown.jpg")
    msg.attach(image_attachment)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.send_message(msg)
            # Keep print for email section as per instruction
            print("[Email Sent] Unknown face alert sent successfully.")
    except Exception as e:
        # Keep print for email section as per instruction
        print(f"[Email Error] Failed to send unknown face alert: {e}")

# Utility function to handle unknown face: snapshot, alert, log
def handle_unknown_face(image, face_id, now):
    global unknown_counter
    try:
        # Assign a persistent label for this face_id
        new_face = False
        if face_id not in unknown_face_labels:
            unknown_face_labels[face_id] = f"unknown{unknown_counter}"
            unknown_counter += 1
            new_face = True
        label = unknown_face_labels[face_id]

        # Save new unknown face entry to CSV
        if new_face:
            with open("unknown.csv", "a") as f:
                f.write(f"{label},{face_id}\n")

        # Save snapshot
        if not os.path.exists("static/unknowns"):
            os.makedirs("static/unknowns")
        snapshot_path = os.path.join("static/unknowns", f"{label}_{now.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(snapshot_path, image)

        # Send email alert
        send_unknown_face_alert(image)

        # Update alert tracking and log
        unknown_face_alerted[face_id] = {
            'time': now,
            'count': unknown_face_alerted.get(face_id, {}).get('count', 0) + 1,
            'label': label
        }
        unknown_log.append({
            'time': now.strftime('%Y-%m-%d %H:%M:%S'),
            'bbox': face_id.replace("_", ","),
            'label': label
        })

        logging.info("[Unknown Handled] Snapshot saved, email sent, log updated for %s (%s)", face_id, label)
    except Exception as e:
        logging.error("[Handler Error] %s", e)

# 🧠 Load Previously Stored Face Encodings
# Load known face embeddings
if os.path.exists('encodings.pkl'):
    try:
        with open('encodings.pkl', 'rb') as f:
            known_encodings, known_names = pickle.load(f)
    except Exception as e:
        logging.error("[Load Encodings Error] %s", e)
        known_encodings, known_names = [], []
else:
    known_encodings, known_names = [], []

unknown_face_alerted = {}
unknown_log = []

# 📝 Attendance Logging Function
# Attendance marker
def mark_attendance(name):
    import re
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if name not in df['Name'].values:
        now = datetime.now()
        time_now = now.strftime('%H:%M:%S')

        # Default phone number as "-"
        phone = "-"
        if os.path.exists('users.csv'):
            with open('users.csv', 'r') as f:
                phone_dict = dict(line.strip().split(',') for line in f if ',' in line)
                phone = phone_dict.get(name, "-")

        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f"\n{name},{phone},{time_now}")

# 📸 Webcam Frame Generation and Real-Time Face Recognition
# Webcam stream
# 📸 Webcam Frame Generation and Real-Time Face Recognition
# Webcam stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap = cv2.VideoCapture(admin:isdr@430:8000)
#cap = cv2.VideoCapture("rtsp://username:password@camera_ip:port/path")
#cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.1.100:554/Streaming/Channels/101/")
#cap = cv2.VideoCapture("rtsp://admin:isdr%40430@192.168.49.247:8000/Streaming/Channels/101/")
#cap = cv2.VideoCapture("http://192.168.49.247:8000/video")


def gen_frames():
    global known_encodings, known_names, unknown_counter
    from datetime import datetime, timedelta
    from threading import Thread
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            logging.error("[Camera Error] Frame not read")
            continue

        # Resize frame to 1280x720
        #frame = cv2.resize(frame, (640, 480))
        frame = cv2.resize(frame, (1280, 720))

        # Enhance frame for low-light conditions during nighttime hours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise after grayscale conversion
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        enhanced = False
        current_hour = datetime.now().hour
        is_night_time = current_hour >= 19 or current_hour < 6  # 7 PM to 6 AM
        if is_night_time and np.mean(gray) < 60:
            frame = enhance_image(frame)
            enhanced = True

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            results = face_app.get(rgb)
        except Exception as e:
            logging.error("[Face Detection Error] %s", e)
            continue

        for face in results:
            embedding = face.embedding
            x1, y1, x2, y2 = map(int, face.bbox)

            if known_encodings:
                matches = cosine_similarity([embedding], known_encodings)[0]
                best_match_index = np.argmax(matches)
                match_score = matches[best_match_index]
                if match_score > 0.45:
                    name = known_names[best_match_index]
                    Thread(target=mark_attendance, args=(name,)).start()
                    # Extract display name using new convention
                    display_name = name.rsplit('_', 1)[0].replace('_', ' ')
                    label = f"{display_name} ({match_score:.2f})"
                else:
                    # --- Embedding-based unknown matching ---
                    now = datetime.now()
                    match_scores = cosine_similarity([embedding], unknown_embeddings)[0] if unknown_embeddings else []
                    best_match_idx = np.argmax(match_scores) if len(match_scores) > 0 else -1
                    if len(match_scores) > 0 and match_scores[best_match_idx] > 0.5:
                        label = unknown_names[best_match_idx]
                    else:
                        label = f"unknown{unknown_counter}"
                        unknown_counter += 1
                        unknown_embeddings.append(embedding)
                        unknown_names.append(label)
                        with open("unknown.csv", "a") as f:
                            f.write(f"{label},{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    # Avoid sending too many emails for the same unknown face label
                    face_id = f"{x1}_{y1}_{x2}_{y2}"
                    alert_info = unknown_face_alerted.get(face_id)
                    alert_count = alert_info.get('count', 0) if alert_info else 0
                    last_alert_time = alert_info.get('time') if alert_info else None
                    if not last_alert_time or (now - last_alert_time) > timedelta(hours=3):
                        # Reset count if time threshold passed
                        alert_count = 0
                    if alert_count < 3:
                        import threading
                        threading.Thread(target=handle_unknown_face, args=(frame.copy(), face_id, now)).start()
            else:
                label = "No registered faces"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if enhanced:
                cv2.putText(frame, "🔦 Enhanced", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame = buffer.tobytes()
        # Limit frame rate
        time.sleep(0.03)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 🌐 Route: Home Page (Attendance Summary + Registered Users)
@app.route('/')
def index():
    if not session.get('user_logged_in'):
        return redirect(url_for('login'))
    phone_dict = {}
    if os.path.exists('users.csv'):
        with open('users.csv', 'r') as f:
            for line in f:
                n, p = line.strip().split(',')
                phone_dict[n] = p
    users = []
    for user_folder in os.listdir(app.config['UPLOAD_FOLDER']):
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], user_folder)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                continue
            latest_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
            latest_file_path = os.path.join(folder_path, latest_file)
            rel_path = os.path.relpath(latest_file_path, start=app.root_path)

            name = user_folder.rsplit('_', 1)[0].replace('_', ' ')
            phone = phone_dict.get(user_folder, "-")
            timestamp = time.ctime(os.path.getmtime(latest_file_path))

            users.append({
                'name': name,
                'phone': phone,
                'time': timestamp,
                'img': rel_path
            })
    attendance_data = []
    csv_path = f'Attendance/Attendance-{datetoday}.csv'
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            next(f)  # Skip the header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    name, phone, time_ = parts
                    attendance_data.append({
                        'name': name.split('__')[0],
                        'phone': phone,
                        'time': time_
                    })
    return render_template('home.html', datetoday2=datetoday2, users=users, data=attendance_data)

# 🌐 Route: Webcam Start Page
@app.route('/start')
def start():
    return render_template('start.html')  # A new template that shows the stream

# 🌐 Route: Live Video Feed (for Streaming)
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🌐 Route: API to Mark Attendance via Single Frame Capture
@app.route('/mark-attendance', methods=['POST'])
def mark_attendance_api():
    success, frame = cap.read()
    if not success:
        return jsonify({'message': '❌ Failed to access camera.'})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_app.get(rgb)

    if not results:
        return jsonify({'message': '😕 No face detected.'})

    embedding = results[0].embedding
    name = "Unknown"
    message = "😕 Face not recognized."

    if known_encodings:
        matches = cosine_similarity([embedding], known_encodings)[0]
        best_match_index = np.argmax(matches)
        match_score = matches[best_match_index]
        if match_score > 0.45:
            name = known_names[best_match_index]

            # Check if already marked
            df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
            if name in df['Name'].values:
                message = f"ℹ️ {name.split('__')[0]}'s attendance already marked."
            else:
                now = datetime.now()
                time_now = now.strftime('%H:%M:%S')
                phone = name.split('__')[1] if '__' in name else "-"
                with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
                    f.write(f"\n{name},{phone},{time_now}")
                message = f"✅ {name.split('__')[0]}'s attendance marked."

            return jsonify({'message': message})

    return jsonify({'message': message})

# 🌐 Route: Admin Login Page
@app.route('/adminlogin', methods=['GET', 'POST'])
def adminlogin():
    error = ''
    if request.method == 'POST':
        username = request.form['userName']
        password = request.form['password']
        if username == 'Bibek35' and password == '8260889508':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid Credentials'
    return render_template('adminlogin.html', error=error)

# 🌐 Route: Admin Dashboard (View Registered Users)
@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('adminlogin'))
    phone_dict = {}
    if os.path.exists('users.csv'):
        with open('users.csv', 'r') as f:
            for line in f:
                n, p = line.strip().split(',')
                phone_dict[n] = p
    users = []
    for user_folder in os.listdir(app.config['UPLOAD_FOLDER']):
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], user_folder)
        if os.path.isdir(folder_path):
            # Get list of image files in this folder
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                continue

            # Find the most recently modified image
            latest_file = max(image_files, key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
            latest_file_path = os.path.join(folder_path, latest_file)
            rel_path = os.path.relpath(latest_file_path, start=app.root_path)

            # Parse name and phone
            name = user_folder.rsplit('_', 1)[0].replace('_', ' ')
            phone = phone_dict.get(user_folder, "-")
            timestamp = time.ctime(os.path.getmtime(latest_file_path))

            users.append({
                'name': name,
                'phone': phone,
                'datetime': timestamp,
                'img': rel_path,
                'folder': user_folder
            })
    return render_template('admin.html', users=users, unknowns=unknown_log)

# 🗑 Route: Delete Registered User and Rebuild Encodings
@app.route('/delete_user/<user_id>')
def delete_user(user_id):
    import shutil
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
        # Rebuild encodings
        known_encodings.clear()
        known_names.clear()
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    path = os.path.join(root, file)
                    img = cv2.imread(path)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = face_app.get(rgb)
                    if results:
                        known_encodings.append(results[0].embedding)
                        known_names.append(os.path.splitext(file)[0])
        with open('encodings.pkl', 'wb') as f:
            pickle.dump((known_encodings, known_names), f)
    return redirect(url_for('admin_dashboard'))

# 🔓 Route: Logout Admin Session
@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    session.pop('user_logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))

import base64

# ✍️ Route: User Registration with Image and Phone
@app.route('/register', methods=['GET', 'POST'])
def register():
    global known_encodings, known_names
    message = ''
    if request.method == 'POST':
        name = request.form['name']
        phone = request.form['phone']
        file = request.files.get('image')
        if not name or not phone:
            message = "Please fill in all fields."
            return render_template('sign.html', message=message)
        if not phone.isdigit() or len(phone) != 10:
            message = "📱 Phone number must be exactly 10 digits."
            return render_template('sign.html', message=message)

        base_name = name.strip().replace(" ", "_")
        person_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_{phone}")
        os.makedirs(person_folder, exist_ok=True)
        existing = [f for f in os.listdir(person_folder) if f.endswith('.jpg') or f.endswith('.png')]

        if len(existing) >= 50:
            message = "Maximum 50 images allowed per user."
            return render_template('sign.html', message=message)

        saved_images = 0
        if file and file.filename:
            count = len(existing) + 1
            filename = secure_filename(f"{base_name}{count}.jpg")
            path = os.path.join(person_folder, filename)
            file.save(path)
            saved_images += 1
            existing.append(filename)
        else:
            for i in range(1, 6):
                b64data = request.form.get(f'captured_image{i}')
                if b64data:
                    try:
                        header, encoded = b64data.split(',', 1)
                        image_data = base64.b64decode(encoded)
                        count = len(existing) + 1
                        img_path = os.path.join(person_folder, f"{base_name}{count}.jpg")
                        with open(img_path, 'wb') as fimg:
                            fimg.write(image_data)
                        existing.append(f"{base_name}{count}.jpg")
                        saved_images += 1
                    except Exception as e:
                        logging.error("Error decoding image %d: %s", i, e)

        if saved_images == 0:
            message = "Please upload or capture at least one image."
            return render_template('sign.html', message=message)

        # Extract features from saved images
        for img_name in existing[-saved_images:]:
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_app.get(rgb)
            if results:
                known_encodings.append(results[0].embedding)
                known_names.append(f"{base_name}_{phone}")
        with open('encodings.pkl', 'wb') as f:
            pickle.dump((known_encodings, known_names), f)

        # Save user info
        with open('users.csv', 'a') as f:
            f.write(f"{base_name}_{phone},{phone}\n")

        # Remove from unknown.csv if registered
        if os.path.exists("unknown.csv"):
            with open("unknown.csv", "r") as f:
                lines = f.readlines()

            new_lines = [lines[0]]  # Keep header
            # Remove any unknown label whose face_id is now registered
            updated_labels = set(unknown_face_labels.values())
            for line in lines[1:]:
                label, face_id = line.strip().split(',')
                # Remove if label refers to a now-registered user
                # base_name is the registered user's base name
                # Remove if label == unknownX for a face_id now handled, or if face_id in unknown_face_labels and unknown_face_labels[face_id] is now registered
                if label not in known_names:
                    new_lines.append(line)
                else:
                    # Remove label from tracking
                    if face_id in unknown_face_labels:
                        unknown_face_labels.pop(face_id)

            with open("unknown.csv", "w") as f:
                f.writelines(new_lines)

        message = "Registration successful."

        # Remove from unknown.csv if registered
        cleaned_unknowns = []
        if os.path.exists("unknown.csv"):
            with open("unknown.csv", "r") as f:
                lines = f.readlines()
            with open("unknown.csv", "w") as f:
                f.write(lines[0])  # Write header
                for line in lines[1:]:
                    label, face_id = line.strip().split(',')
                    if label not in known_names:
                        f.write(line)
                    else:
                        unknown_face_labels.pop(face_id, None)

    return render_template('sign.html', message=message)


# 🛠 Utility Route: Fix Missing Phone Numbers in users.csv
@app.route('/fix-missing-phones')
def fix_missing_phones():
    phone_dict = {}
    if os.path.exists('users.csv'):
        with open('users.csv', 'r') as f:
            for line in f:
                n, p = line.strip().split(',')
                phone_dict[n] = p

    updated = False
    with open('users.csv', 'a') as f:
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.endswith('.jpg') or file.endswith('.png'):
                name_id = os.path.splitext(file)[0]
                if name_id not in phone_dict:
                    fake_phone = f"99999{name_id[-5:].replace('_','')[:5]:0<5}"
                    f.write(f"{name_id},{fake_phone}\n")
                    updated = True

    if updated:
        return "✅ users.csv has been updated with missing phone numbers."
    else:
        return "✅ All entries in users.csv are already up to date."


# 📸 Route: Capture Image from Webcam
@app.route('/capture_image', methods=['GET'])
def capture_image():
    success, frame = cap.read()
    if success:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if not os.path.exists('Captured'):
            os.makedirs('Captured')
        img_path = os.path.join('Captured', f'captured_{timestamp}.jpg')
        cv2.imwrite(img_path, frame)
        return jsonify({'status': 'success', 'image_path': img_path})
    else:
        return jsonify({'status': 'failed', 'message': 'Camera not accessible'})


# 🌐 Route: General User Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'user' and password == 'user@123':
            session['user_logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = 'Invalid Credentials'
    return render_template('login.html', error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5050, host='0.0.0.0', use_reloader=False)

