import cv2
import mediapipe as mp
import numpy as np
import time
import dlib
from scipy.spatial import distance
from tensorflow.keras.models import model_from_json

# Load Face Detection Model
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load Anti-Spoofing Model graph
json_file = open("antispoofing_models/1antispoofing_model_mobilenet.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load anti-spoofing model weights
model.load_weights("antispoofing_models/1antispoofing_model_99-0.969474.h5")
print("Model loaded from disk")


# Function to calculate EAR
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])  # p2, p6
    B = distance.euclidean(eye[2], eye[4])  # p3, p5
    C = distance.euclidean(eye[0], eye[3])  # p1, p4
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame_rate = 20
eyes_closed_warning_counter = 0
eyes_closed_warning_duration = 60 * frame_rate
closed_eyes_counter = 0
closed_eyes_threshold = 2 * frame_rate  # Number of seconds to detect closed eyes
head_orientation_warning_duration = 30 * frame_rate  # Duration to display head orientation warnings
head_orientation_counter = 0
head_direction_counter = 0
head_direction_threshold = 2 * frame_rate

while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -15:
                text = "Looking Left"
            elif y > 15:
                text = "Looking Right"
            elif x < -15:
                text = "Looking Down"
            elif x > 15:
                text = "Looking Up"
            else:
                text = "Forward"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            # cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if text != "Forward" and text != "Looking Up":
                head_direction_counter += 1
                # print("head_direction_counter " + str(head_direction_counter))
                if head_direction_counter >= head_direction_threshold:
                    head_orientation_counter = head_orientation_warning_duration
                    # print("Focus on the lesson!")
            else:
                head_direction_counter = 0

            # print(text)
            if head_orientation_counter > 0:
                cv2.putText(image, "Focus on the lesson", (20, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                head_orientation_counter -= 1

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                # connections=mp_face_mesh.FACEMESH_CONTOURS,
                # landmark_drawing_spec=drawing_spec,
                # connection_drawing_spec=drawing_spec
                connections=None,
                landmark_drawing_spec=None,
                connection_drawing_spec=None)

    # Face detection and anti-spoofing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face = image[y - 5: y + h + 5, x - 5: x + w + 5]
        resized_face = cv2.resize(face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)
        preds = model.predict(resized_face)[0]

        if preds > 0.5:
            label = "spoof"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h + 50), (0, 0, 255), 2)
        else:
            label = "real"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h + 50), (0, 255, 0), 2)

    # Eye tracking and drowsiness detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(image, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        if EAR < 0.2:
            closed_eyes_counter += 1
            # print("closed_eyes_counter " + str(closed_eyes_counter))
            if closed_eyes_counter >= closed_eyes_threshold:
                eyes_closed_warning_counter = eyes_closed_warning_duration
            print("Drowsy")
        else:
            closed_eyes_counter = 0  # Reset the counter if eyes are open

        print(EAR)

        if eyes_closed_warning_counter > 0:
            cv2.putText(image, "Are you Sleepy?", (20, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            eyes_closed_warning_counter -= 1

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    print("FPS: " + str(fps))
    cv2.imshow('Face_Detection', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
