import cv2
import numpy as np
import mediapipe as mp

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

print("Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Black canvas for neural mesh
    neural = np.zeros((h, w, 3), dtype=np.uint8)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Draw neural mesh
            mp.solutions.drawing_utils.draw_landmarks(
                image=neural,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 200, 255),  # neon blue
                    thickness=1
                )
            )

    # Resize if needed (same size safety)
    frame = cv2.resize(frame, (w, h))
    neural = cv2.resize(neural, (w, h))

    # Combine side-by-side
    combined = np.hstack((frame, neural))

    cv2.imshow("Real Camera  |  Neural Face Mesh", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




