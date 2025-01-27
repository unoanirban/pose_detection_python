import cv2
import mediapipe as mp

# Initialize MediaPipe pose class and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

# Initialize the Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB (since MediaPipe works with RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect the pose landmarks
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the original frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # Display the image
        cv2.imshow('Pose Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
