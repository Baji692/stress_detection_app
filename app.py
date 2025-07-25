from flask import Flask, render_template, Response, request, redirect, url_for
from deepface import DeepFace
import cv2
import os
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Emotion to stress percentage mapping
stress_map = {
    'happy': 10,
    'neutral': 30,
    'surprise': 40,
    'sad': 70,
    'fear': 90,
    'angry': 95,
    'disgust': 80
}

def calculate_stress(emotion):
    return stress_map.get(emotion.lower(), 50)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}.jpg")
    file.save(filepath)

    try:
        analysis = DeepFace.analyze(img_path=filepath, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        stress = calculate_stress(emotion)
    except:
        emotion = "Unknown"
        stress = 50

    return render_template('result.html', emotion=emotion.capitalize(), stress=stress, image_path=filepath)

# Video streaming
def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = analysis[0]['dominant_emotion']
                stress = calculate_stress(emotion)

                label = f"{emotion.upper()} - Stress: {stress}%"
                cv2.putText(frame, label, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            except:
                pass

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live')
def live():
    return render_template('live.html')

if __name__ == '__main__':
    app.run(debug=True)
