import cv2

# Function to detect motion in a video frame
def detect_motion(frame1, frame2):
    # Convert frames to grayscale for processing
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply threshold to highlight areas with significant differences
    _, threshold = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected motion areas
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame1

# Capture video from webcam (you can use a video file as well)
video_capture = cv2.VideoCapture(0)

# Read initial frames
ret, frame1 = video_capture.read()
ret, frame2 = video_capture.read()

while True:
    # Detect motion in consecutive frames
    motion_detected_frame = detect_motion(frame1, frame2)

    # Display the frame with motion detection
    cv2.imshow('Motion Detection', motion_detected_frame)

    # Update frames for the next iteration
    frame1 = frame2
    ret, frame2 = video_capture.read()

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
