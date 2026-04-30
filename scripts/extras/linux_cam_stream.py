import cv2
from flask import Flask, Response

app = Flask(__name__)

def init_camera(index):
    cap = cv2.VideoCapture(index)
    # Use MJPG format to reduce USB bandwidth and CPU decoding overhead
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

camera0 = init_camera(0)
camera1 = init_camera(2)

def generate_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # Yield the output in byte format for HTTP multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/cam0')
def video_feed_0():
    return Response(generate_frames(camera0), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam1')
def video_feed_1():
    return Response(generate_frames(camera1), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
      <head>
        <title>Camera Streams</title>
      </head>
      <body>
        <div style="display: flex; gap: 20px;">
          <div>
            <img src="/cam0" width="640" height="480">
          </div>
          <div>
            <img src="/cam1" width="640" height="480">
          </div>
        </div>
      </body>
    </html>
    '''

if __name__ == "__main__":
    # Host 0.0.0.0 makes it accessible on your local network
    app.run(host='0.0.0.0', port=5000, threaded=True)
