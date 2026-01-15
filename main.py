import cv2
import pygame
import numpy as np
from flask import Flask, render_template, Response
import os

# ==========================================
# CONFIGURATION & SETTINGS
# ==========================================
ALARM_FILE = "1.mp3"       # The name of your alarm audio file
SLEEP_THRESHOLD = 15       # Number of frames eyes must be closed to trigger alarm
Face_Scale = 1.3           # Face detection scale factor
Eye_Scale = 1.1            # Eye detection scale factor

# ==========================================
# INITIALIZATION
# ==========================================
app = Flask(__name__)
pygame.mixer.init()

# 1. Setup Alarm Sound
if os.path.exists(ALARM_FILE):
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
    print(f"SUCCESS: Audio file '{ALARM_FILE}' loaded.")
else:
    print(f"WARNING: '{ALARM_FILE}' not found. Alarm will NOT play.")
    alarm_sound = None

alarm_active = False

# 2. Setup Haar Cascades (Face & Eye Detectors)
# Using cv2.data.haarcascades ensures we find the files automatically
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

if face_cascade.empty() or eye_cascade.empty():
    print("CRITICAL ERROR: Could not load Haar Cascade XML files.")
    print("Ensure you have opencv-python installed correctly.")
else:
    print("SUCCESS: AI Models loaded successfully.")

# ==========================================
# CORE LOGIC CLASS
# ==========================================
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.score = 0
        global alarm_active

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global alarm_active
        success, image = self.video.read()
        if not success:
            return None

        # 1. Image Processing for "Major Project" Accuracy
        # Convert to Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # IMPROVEMENT: Histogram Equalization
        # This helps the AI see better in low light (Night Driving Simulation)
        gray = cv2.equalizeHist(gray) 
        
        # 2. Detect Faces
        faces = face_cascade.detectMultiScale(gray, Face_Scale, 5)

        status = "Active"
        color = (0, 255, 0) # Green

        # Safe Fail: If no face is found, reset score so alarm doesn't ring randomly
        if len(faces) == 0:
            self.score = 0
            status = "No Face Found"
        
        for (x, y, w, h) in faces:
            # Draw box around face
            cv2.rectangle(image, (x, y), (x+w, y+h), (100, 100, 100), 2)
            
            # Region of Interest (ROI) - Only look for eyes INSIDE the face
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            # Detect Eyes
            eyes = eye_cascade.detectMultiScale(roi_gray, Eye_Scale, 4)
            
            if len(eyes) == 0:
                # ---------------------------
                # STATE: EYES CLOSED (SLEEP)
                # ---------------------------
                self.score += 1
                status = "Eyes Closed"
                if self.score > SLEEP_THRESHOLD:
                    status = "DROWSINESS ALERT!"
                    color = (0, 0, 255) # Red
                    
                    # Visual Warning
                    cv2.putText(image, "WAKE UP!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Audio Alarm
                    if not alarm_active and alarm_sound:
                        alarm_sound.play(-1) # -1 means loop forever
                        alarm_active = True
            else:
                # ---------------------------
                # STATE: EYES OPEN (AWAKE)
                # ---------------------------
                self.score -= 1
                if self.score < 0:
                    self.score = 0
                status = "Awake"
                
                # Stop Alarm if it was ringing
                if alarm_active and alarm_sound:
                    alarm_sound.stop()
                    alarm_active = False
                
                # Draw boxes around eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # 3. Dashboard Overlay (HUD)
        # Add a dark background bar for text readability
        cv2.rectangle(image, (0, 0), (300, 80), (0, 0, 0), -1) 
        cv2.putText(image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(image, f"Fatigue Score: {self.score}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

# ==========================================
# WEB SERVER ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    # host='0.0.0.0' allows you to view this on your phone if connected to same WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)