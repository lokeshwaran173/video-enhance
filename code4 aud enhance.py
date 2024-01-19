import cv2
import numpy as np
import pyaudio
import audioop

class LiveVideoAnalyzer:
    def __init__(self, cascade_path):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Audio processing parameters
        self.audio_chunk = 1024
        self.audio_format = pyaudio.paInt16
        self.audio_channels = 1
        self.audio_rate = 44100
        self.audio_gain = 1.5

        self.audio_stream = self.init_audio_stream()

    def init_audio_stream(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format,
                        channels=self.audio_channels,
                        rate=self.audio_rate,
                        input=True,
                        frames_per_buffer=self.audio_chunk)
        return stream

    def process_audio(self):
        audio_data = np.frombuffer(self.audio_stream.read(self.audio_chunk), dtype=np.int16)
        # Apply audio processing (e.g., gain adjustment)
        processed_audio = audioop.mul(audio_data.tobytes(), 2, self.audio_gain)
        return np.frombuffer(processed_audio, dtype=np.int16)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    def process_frame(self, frame):
        self.detect_faces(frame)

    def process_and_display(self):
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            # Process video frame
            self.process_frame(frame)

            # Process audio
            processed_audio = self.process_audio()

            # Display the processed frame
            cv2.imshow('Live Video Analysis', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.audio_stream.stop_stream()
        self.audio_stream.close()

video_capture = cv2.VideoCapture(0)  # Assuming a webcam feed, replace with your video source

cascade_path = 'haarcascade_frontalface_default.xml'
analyzer = LiveVideoAnalyzer(cascade_path)

analyzer.process_and_display()

video_capture.release()
cv2.destroyAllWindows()
