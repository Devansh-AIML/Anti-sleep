
from flask import Flask, render_template, Response
import cv2
import pygame
app=Flask(__name__)
camera = cv2.VideoCapture(0)

pygame.init()
pygame.mixer.init()

threads = []

audio_file = "./1.mp3"

def play_warning_sound(flag):
    if(flag):
        
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play(-1)
    
    
    
def gen_frames(): 
    
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    video_capture = cv2.VideoCapture(0)
    count = 0
    flag = 0
     
    while True:

            result, video_frame = video_capture.read()  
            if result is False:
                break 
            font = cv2.FONT_HERSHEY_SIMPLEX
            gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
            if(len(faces)==0):
                print(count)
                if(count > 10):
                    
                    cv2.putText(video_frame, 'Alert! No Face Detected', (50,50), font, 1, (25,0,0), 2, cv2.LINE_AA)
                    play_warning_sound(flag)
                    flag = 0
                    
                else:
                    
                    count+=1
            for (x, y, w, h) in faces:
                count = 0
                flag = 1
                pygame.mixer.music.stop();
                cv2.rectangle(video_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(video_frame, 'Face Detected', (50,50), font, 1, (25,0,0), 2, cv2.LINE_AA)
          
                    
            ret, buffer = cv2.imencode('.jpg', video_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/console1')
def console1():
    return "hello"
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)