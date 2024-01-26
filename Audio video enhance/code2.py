import cv2

class FaceDetection:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Example usage for live video streaming with face detection
video_capture = cv2.VideoCapture(0)  # Assuming a webcam feed, replace with your video source

# Load the face detection model (replace with your actual model)
cascade_path = 'haarcascade_frontalface_default.xml'
face_detector = FaceDetection(cascade_path)

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Process the frame for face detection
    face_detector.detect_faces(frame)

    # Display the processed frame
    cv2.imshow('Live Video Analysis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
