import cv2
import dlib
from scipy.spatial import distance

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B)/(2.0*C)
    return ear

cap = cv2.VideoCapture(1)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        left_eye = []
        right_eye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            left_eye.append((x,y))
            next_point = n+1

            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y

            right_eye.append((x,y))
            next_point = n+1

            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y

        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)

        EAR = round(((left_EAR + right_EAR) / 2), 2)

        if EAR < .26:
            cv2.putText(frame, "Merem terdeteksi", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)
            cv2.putText(frame, "Ngantuk bang?", (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Checking...")  # Perbaiki indentasi


    cv2.imshow("Ngantuk bang?", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


        
