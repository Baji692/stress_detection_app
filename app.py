from flask import Flask, render_template, request, Response
import cv2
from fer import FER
import numpy as np

app = Flask(__name__)

# Globals
camera = None
streaming = False

# Faster detector for live usage
detector = FER(mtcnn=False)

# Desired camera resolution
CAM_W, CAM_H = 1280, 720


def open_camera():
    """Open the webcam if it's not already open."""
    global camera
    if camera is None or not camera.isOpened():
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        camera = cam


def close_camera():
    """Release the webcam properly."""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
    camera = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global streaming
    open_camera()
    streaming = True
    return ('', 204)


@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global streaming
    streaming = False
    close_camera()
    return ('', 204)


@app.route('/video_feed')
def video_feed():
    def generate():
        global streaming, camera
        open_camera()

        while streaming and camera is not None and camera.isOpened():
            success, frame = camera.read()
            if not success:
                break

            result = detector.detect_emotions(frame)
            if result and 'emotions' in result[0]:
                emotions = result[0]['emotions']
                top_emotion = max(emotions, key=emotions.get)
                confidence = round(emotions[top_emotion] * 100, 2)

                stress_emotions = ['angry', 'disgust', 'fear', 'sad']
                stress_percent = round(
                    confidence if top_emotion in stress_emotions else 100 - confidence)

                label = f"{top_emotion.upper()} ({confidence}%) | Stress: {stress_percent}%"
                color = (0, 0, 255) if stress_percent >= 50 else (0, 255, 0)
                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No face detected", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 100), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        # When loop ends, make sure camera is closed
        close_camera()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['image']
        if not file:
            return "No file uploaded", 400

        image_data = file.read()
        npimg = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        result = detector.detect_emotions(img)
        if result and 'emotions' in result[0]:
            emotions = result[0]['emotions']
            top_emotion = max(emotions, key=emotions.get)
            confidence = round(emotions[top_emotion] * 100, 2)
            stress_emotions = ['angry', 'disgust', 'fear', 'sad']
            stress_percent = round(
                confidence if top_emotion in stress_emotions else 100 - confidence, 2)
        else:
            top_emotion, confidence, stress_percent = "No Face Detected", 0, 0

        return render_template("result.html",
                               emotion=top_emotion,
                               confidence=confidence,
                               stress_percent=stress_percent)
    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
